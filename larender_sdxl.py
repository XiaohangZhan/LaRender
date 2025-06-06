import torch
import numpy as np
import time
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

import math
import xformers
import random
import os
import pdb

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class DependencyParsing():

    def __init__(self):
        # pip install spacy
        # python -m spacy download en_core_web_sm
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def parse(self, text):
        doc = self.nlp(text)
        indices = []
        for token in doc:
            # noun of a sentence
            if token.dep_ in ["nsubj", "nsubjpass"]:
                indices.append(token.i + 1)
            # noun in a phrase
            elif token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]:
                indices.append(token.i + 1) # +1 because of SOS
        return indices


class LaRender():
    def __init__(self, use_attn_map=True):
        self.pipe = DiffusionPipeline.from_pretrained(
            "comin/IterComp",torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
        self.pipe.to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,use_karras_sigmas=True)
        self.pipe.enable_xformers_memory_efficient_attention()

        self.isvanilla = True
        self.use_attn_map = use_attn_map

        for name, module in self.pipe.unet.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                module.forward = self.create_larender_forward(module)

    
    def _memory_efficient_attention_xformers(self, module, query, key, value):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
        hidden_states = module.batch_to_head_dim(hidden_states)
        return hidden_states

    def _main_forward_diffusers(self, module,hidden_states,encoder_hidden_states):
        query = module.to_q(hidden_states)
        key = module.to_k(encoder_hidden_states)
        value = module.to_v(encoder_hidden_states)
        query = module.head_to_batch_dim(query)
        key = module.head_to_batch_dim(key)
        value = module.head_to_batch_dim(value)
        hidden_states= self._memory_efficient_attention_xformers(module, query, key, value)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)
        return hidden_states


    def _get_attention_weights(self, module,hidden_states,encoder_hidden_states):
        query = module.to_q(hidden_states)
        key = module.to_k(encoder_hidden_states)
        query = module.head_to_batch_dim(query)
        key = module.head_to_batch_dim(key)
        attention_probs = module.get_attention_scores(query,key)
        return attention_probs

    def _bboxes_to_mask(self, bboxes, h, w, dtype, device):
        # bboxes: List[List[y1, y2, x1, x2]]
        masks = []
        for i, bbox in enumerate(bboxes):
            M = torch.zeros((h, w), dtype=dtype, device=device)
            upper = int(h * bbox[0])
            lower = int(h * bbox[1])
            left = int(w * bbox[2])
            right = int(w * bbox[3])
            M[upper:lower, left:right]=1
            masks.append(M)
        return masks

    def larender_core_process(self, module, hidden_states, encoder_hidden_states):
        
        # prepare
        locations = self.larender_inputs['locations']
        indices = self.larender_inputs['indices']
        densities = self.larender_inputs['densities']
        height = self.height
        width = self.width

        latent_h, latent_w = split_dims(hidden_states.size()[1], height, width, self.pipe)

        assert hasattr(self.pipe.scheduler, 'step_index')
        current_count = self.pipe.scheduler.step_index if self.pipe.scheduler.step_index is not None else 0
        num_inference_steps = len(self.pipe.scheduler.timesteps)
        density_multiplier = num_inference_steps / (current_count + 1)
        densities_multiplied = [x * density_multiplier for x in densities]

        token_max_length = self.pipe.tokenizer.model_max_length
        assert encoder_hidden_states.shape[1] % token_max_length == 0

        # object-wise cross-attention
        R = []
        for i in range(self.num_objects):
            context = encoder_hidden_states[:, i * token_max_length: (i+1) * token_max_length, :]
            R_i = self._main_forward_diffusers(module, hidden_states, context)
            R.append(R_i.reshape(R_i.shape[0], latent_h, latent_w, R_i.shape[2]))
        dtype = R[0].dtype
        device = R[0].device

        # compute transmittance maps
        M = self._bboxes_to_mask(locations, latent_h, latent_w, dtype, device)
        if self.use_attn_map:
            for i in range(1, self.num_objects): # exclude background
                context = encoder_hidden_states[:, i * token_max_length: (i+1) * token_max_length, :]
                attn_map = self._get_attention_weights(module, hidden_states, context)
                attn_map = attn_map.reshape(attn_map.shape[0], latent_h, latent_w, attn_map.shape[2])
                attn_map = attn_map[..., indices[i]].mean(dim=-1).mean(dim=0)
                min_val = attn_map.min()
                max_val = attn_map.max()
                normalized_attn_map = (attn_map - min_val) / (max_val - min_val)
                
                M[i] *= normalized_attn_map

        # accumulated transmittance maps (visibility of all pixels of object i from the virtual camera)
        T = [torch.ones((latent_h, latent_w), dtype=dtype, device=device) for _ in range(self.num_objects)]
        for i in range(self.num_objects - 1, -1, -1): # top to bottom
            for j in range(i + 1, self.num_objects):
                T[i] *= torch.exp(-torch.tensor(densities_multiplied[j]) * M[j]) # TODO

        # rendering
        S = 0
        R_out = 0
        for i in range(self.num_objects):
            tmp = T[i] * (1 - math.exp(-densities_multiplied[i])) * M[i]
            tmp = tmp[None,:,:,None] # unsqueeze 0, -1
            R_out += tmp * R[i]
            S += tmp
        # pdb.set_trace()
        S = torch.clamp(S, min=1e-6)
        R_out /= S
        return R_out.reshape(*hidden_states.shape)

    def create_larender_forward(self, module):
        def forward(hidden_states, encoder_hidden_states=None, **kwargs):
            if self.isvanilla: # SBM Ddim reverses cond/uncond.这里执行了
                nx, px = hidden_states.chunk(2)  #将张量 x 均匀分割成两个子张量 nx 和 px
                conn,conp = encoder_hidden_states.chunk(2)#将张量 contexts均匀分割成两个子张量,分割之后通道数为1
            else:
                px, nx = hidden_states.chunk(2)
                conp,conn = encoder_hidden_states.chunk(2)
            opx = self.larender_core_process(module, px, conp)
            onx = self.larender_core_process(module, nx, conn)
            if self.isvanilla: # SBM Ddim reverses cond/uncond. 这里执行了
                output_x = torch.cat([onx, opx])  #再cat消极和积极
            else:
                output_x = torch.cat([opx, onx]) 
            return output_x

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
            # pdb.set_trace()
            prompt_embeds_list.append(prompt_embeds)
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            negative_pooled_prompt_embeds_list.append(negative_pooled_prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=1)
        # pooled_prompt_embeds = sum(pooled_prompt_embeds_list) / len(pooled_prompt_embeds)
        # negative_pooled_prompt_embeds = sum(negative_pooled_prompt_embeds_list) / len(negative_pooled_prompt_embeds)

        # (
        # _,
        # _,
        # pooled_prompt_embeds,
        # negative_pooled_prompt_embeds,
        # ) = pipe.encode_prompt(
        #     prompt=', '.join([background] + objects),
        #     do_classifier_free_guidance=True,
        #     negative_prompt=negative_prompt,
        # )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def run(self, objects, locations, alpha, indices, height, width, negative_prompt, seed):
        for a in alpha:
            assert a < 1 and a >= 0, "alpha must in range of [0, 1)"
        densities = [-np.log(1 - a) for a in alpha]
        self.num_objects = len(objects)
        assert len(locations) == len(objects) and len(densities) == len(objects)
        self.larender_inputs = dict(locations=locations, indices=indices, densities=densities)

        self.height = height
        self.width = width
        
        (prompt_embeds,
        negative_prompt_embeds, 
        pooled_prompt_embeds, 
        negative_pooled_prompt_embeds) = self._prompt_embedding(objects, negative_prompt)

        if seed > 0:
            torch_fix_seed(seed)
        images = self.pipe(
            num_inference_steps=25, 
            height = height,
            width = width,
            seed = seed,
            guidance_scale = 7.0,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        ).images[0]
        return images
    


def split_dims(x_t, height, width, pipe=None):
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    assert x_t == latent_h * latent_w, f"Fail split: {x_t} != {latent_h}, {latent_w}"
    # if x_t > latent_h * latent_w and hasattr(pipe, "nei_multi"):
    #     latent_h, latent_w = pipe.nei_multi[1], pipe.nei_multi[0] 
    #     while latent_h * latent_w != x_t:
    #         latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    while y > 0:
        x = math.ceil(x / 2) 
        y = y - 1
    return x



def main():
    negative_prompt = ''
    use_attn_map = True

    save_name = 'refactor'
    objects= ["a large empty living room", "a piano", "a giraffe",  "a sofa",'many cats',"a blue teddy bear"]
    locations = [[0, 1, 0, 1], [0.3, 0.7, 0.2, 0.95], [0, 0.8, 0, 0.5],  [0.6, 0.9, 0.1, 0.9], [0.55,0.7,0.5,0.7],[0.5, 0.9, 0.6, 0.9]] # mode: [y1, y2, x1, x2], range: 0~1
    alpha = [0.6, 0.99, 0.99, 0.6, 0.99,0.6]
    seed = 82

    save_name = 'improve'
    objects = ['forest', 'a cat', 'a dog', 'a branch']
    locations = [[0, 1, 0, 1], [0.4, 0.8, 0.1, 0.4], [0.3, 0.9, 0.5, 0.7], [0.4, 0.5, 0, 1]]
    alpha = [0.6, 0.6, 0.6, 0.6]
    seed = 5
    

    height = 1024
    width = 1024

    if use_attn_map:
        DP = DependencyParsing()
        indices = [DP.parse(obj) for obj in objects]
    else:
        indices = None

    LR = LaRender(use_attn_map=use_attn_map)
    
    
    start_time = time.time()
    images = LR.run(objects, locations, alpha, indices, height, width, negative_prompt, seed)
    print("Elapsed time:", time.time() - start_time, "seconds")
    os.makedirs(f"results/{save_name}", exist_ok=True)
    save_fn = f"results/{save_name}/{save_name}_{seed:03d}_{alpha[-1]:.3f}.png"
    print(f'Save at: {save_fn}')
    images.save(save_fn)

if __name__ == '__main__':
    main()