import math
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from utils import torch_fix_seed


class LaRender():
    def __init__(self,
                 pretrain='IterComp',
                 transmittance_use_bbox=True,
                 transmittance_use_attn_map=True,
                 density_scheduler='unchanged'):
        '''
        Args:
            pretrain ('str', *optional*):
                Pretrain model type, support 'IterComp' or 'GLIGEN' now. 'IterComp' is T2I without location control, while 'GLIGEN' is a pre-trained location contorl model.
            transmittance_use_bbox (bool, *optional*):
                The flag whether to use bounding boxes in approximating transmittance maps. If set to False, objects in the back usually disappear.
            transmittance_use_attn_map (bool, *optional*):
                The flag whether to use attention maps in approximating transmittance maps.
            density_scheduler (str, *optional*):
                The type of density scheduler, support 'opaque', 'unchanged', 'inverse_proportional'.

        '''
        assert pretrain in ['IterComp', 'GLIGEN']
        assert density_scheduler in [
            'opaque', 'unchanged', 'inverse_proportional']
        self.pretrain = pretrain
        self.transmittance_use_bbox = transmittance_use_bbox
        self.density_scheduler = density_scheduler

        if pretrain == 'IterComp':
            self.pipe = DiffusionPipeline.from_pretrained(
                "comin/IterComp",
                torch_dtype=torch.float16,
                use_safetensors=True)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                "jiuntian/gligen-xl-1024", trust_remote_code=True, torch_dtype=torch.float16)
        self.pipe.to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True)
        self.pipe.enable_xformers_memory_efficient_attention()

        self.isvanilla = True
        self.transmittance_use_attn_map = transmittance_use_attn_map

        for name, module in self.pipe.unet.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                module.forward = self.create_larender_forward(module)
                module.name = name

    def _forward_cross_attention(self, module, hidden_states, encoder_hidden_states):
        query = module.to_q(hidden_states)
        key = module.to_k(encoder_hidden_states)
        value = module.to_v(encoder_hidden_states)
        query = module.head_to_batch_dim(query)
        key = module.head_to_batch_dim(key)
        value = module.head_to_batch_dim(value)
        attention_probs = module.get_attention_scores(query, key)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = module.batch_to_head_dim(hidden_states)
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)
        return hidden_states, attention_probs

    def _bboxes_to_mask(self, bboxes, h, w, dtype, device):
        '''
        Args:
            bboxes (List[List[y1, y2, x1, x2]]):
                The bounding boxes in y1, y2, x1, x2 format.
        '''
        masks = []
        for bbox in bboxes:
            M = torch.zeros((h, w), dtype=dtype, device=device)
            upper = int(h * bbox[0])
            lower = int(h * bbox[1])
            left = int(w * bbox[2])
            right = int(w * bbox[3])
            assert lower > upper and right > left, f"Invalid bounding box: {bbox}"
            M[upper:lower, left:right] = 1
            masks.append(M)
        return masks

    def create_larender_forward(self, module):
        def forward(hidden_states, encoder_hidden_states=None, **kwargs):
            return self.larender_core_process(module, hidden_states, encoder_hidden_states)
        return forward

    def _prompt_embedding(self, objects, negative_prompt):
        prompt_embeds_list = []
        negative_prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        negative_pooled_prompt_embeds_list = []
        for p in objects:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=p,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_list.append(prompt_embeds)
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            negative_pooled_prompt_embeds_list.append(
                negative_pooled_prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=1)
        pooled_prompt_embeds = sum(
            pooled_prompt_embeds_list) / len(pooled_prompt_embeds)
        negative_pooled_prompt_embeds = sum(
            negative_pooled_prompt_embeds_list) / len(negative_pooled_prompt_embeds)

        (
            _,
            _,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=', '.join(objects),
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def larender_core_process(self, module, hidden_states, encoder_hidden_states):
        # preparation
        locations = self.larender_inputs['locations']
        indices = self.larender_inputs['indices']
        densities = self.larender_inputs['densities']
        height = self.height
        width = self.width
        latent_h = int(math.sqrt(hidden_states.size()[1] * height / width))
        assert hidden_states.size()[1] % latent_h == 0, \
            "Cannot infer latent_h and latent_w, please check the code."
        latent_w = int(hidden_states.size()[1] / latent_h)
        num_inference_steps = len(self.pipe.scheduler.timesteps)
        assert hasattr(self.pipe.scheduler, 'step_index')
        current_count = self.pipe.scheduler.step_index if self.pipe.scheduler.step_index is not None else 0
        self.current_count = current_count  # only used in visualization
        if self.density_scheduler == 'inverse_proportional':
            density_multiplier = num_inference_steps / (current_count + 1)
            densities_multiplied = [x * density_multiplier for x in densities]
        elif self.density_scheduler == 'opaque':
            densities_multiplied = [x * num_inference_steps for x in densities]
        else:
            densities_multiplied = densities

        token_max_length = self.pipe.tokenizer.model_max_length
        assert encoder_hidden_states.shape[1] % token_max_length == 0

        # object-wise cross-attention and transmittance map
        R = []
        dtype = hidden_states.dtype
        device = hidden_states.device
        if self.transmittance_use_bbox:
            M = self._bboxes_to_mask(
                locations, latent_h, latent_w, dtype, device)
        else:
            M = [torch.ones((latent_h, latent_w), dtype=dtype,
                            device=device) for _ in locations]
        for i in range(self.num_objects):
            context = encoder_hidden_states[:, i *
                                            token_max_length: (i+1) * token_max_length, :]
            R_i, attn_probs = \
                self._forward_cross_attention(module, hidden_states, context)
            R.append(R_i.reshape(R_i.shape[0],
                     latent_h, latent_w, R_i.shape[2]))
            if self.transmittance_use_attn_map:
                attn_map = attn_probs.reshape(
                    attn_probs.shape[0], latent_h, latent_w, -1)
                attn_map = attn_map[..., indices[i]].mean(dim=-1).mean(dim=0)
                min_val = attn_map.min()
                max_val = attn_map.max()
                normalized_attn_map = (
                    attn_map - min_val) / (max_val - min_val)
                M[i] *= normalized_attn_map

        # accumulated transmittance maps (visibility of planar i from the virtual camera)
        T = [torch.ones((latent_h, latent_w), dtype=dtype, device=device)
             for _ in range(self.num_objects)]
        for i in range(self.num_objects - 1, -1, -1):  # top to bottom
            for j in range(i + 1, self.num_objects):
                # TODO
                T[i] *= torch.exp(-torch.tensor(densities_multiplied[j]) * M[j])

        # rendering
        S = 0
        R_out = 0
        for i in range(self.num_objects):
            contrib = T[i] * (1 - math.exp(-densities_multiplied[i])) * M[i]
            contrib = contrib[None, :, :, None]  # unsqueeze 0, -1
            R_out += contrib * R[i]
            S += contrib
        S = torch.clamp(S, min=1e-6)
        R_out /= S
        return R_out.reshape(*hidden_states.shape)

    def run(self,
            objects,
            locations,
            alpha,
            indices,
            num_images_per_prompt=1,
            height=1024,
            width=1024,
            negative_prompt='',
            seed=0):
        '''
        Args:
            objects (List[str]):
                List of object prompts.
            locations (List[List[float]]):
                List of bounding boxes in [top, bottom, left, right] format.
            alpha ([List[float]):
                List of alpha of each object, values must be in [0, 1).
            indices (List[List[int]]):
                List of indices of subjects in each object prompt.
        Output:
            images (List[PIL.Image]):
                List of generated images.
        '''
        for a in alpha:
            assert a < 1 and a >= 0, "alpha must in range of [0, 1)"
        densities = [-np.log(1 - a) for a in alpha]
        self.num_objects = len(objects)
        assert len(locations) == len(objects) and len(
            densities) == len(objects)
        self.larender_inputs = dict(
            locations=locations, indices=indices, densities=densities)

        self.height = height
        self.width = width

        (prompt_embeds,
         negative_prompt_embeds,
         pooled_prompt_embeds,
         negative_pooled_prompt_embeds) = self._prompt_embedding(objects, negative_prompt)

        torch_fix_seed(seed)

        common_kwargs = dict(
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            guidance_scale=7.0,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds)
        if self.pretrain == 'IterComp':
            images = self.pipe(
                num_inference_steps=25, **common_kwargs).images
        else:
            images = self.pipe(
                num_inference_steps=50,
                gligen_scheduled_sampling_beta=0.4,
                gligen_boxes=[[loc[2], loc[0], loc[3], loc[1]]
                              for loc in locations],  # y1y2x1x2 -> x1y1x2y2
                gligen_phrases=objects,
                **common_kwargs).images
        return images
