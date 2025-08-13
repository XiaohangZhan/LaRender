import os
import time
from PIL import ImageDraw, ImageFont
from utils import DependencyParsing
from larender_sdxl import LaRender


if __name__ == "__main__":
    # basic params
    negative_prompt = ''
    pretrain = 'IterComp'  # IterComp or GLIGEN
    num_images_per_prompt = 10
    height = 1024
    width = 1024
    save_bbox_image = False
    seed = 0

    # dependency parsing params
    use_dependency_parsing = True
    ignore_dp_failure = True

    # larender params
    transmittance_use_bbox = True
    transmittance_use_attn_map = True
    density_scheduler = 'inverse_proportional'  # other choices: unchanged, opaque

    #### Reproducible experiments, uncomment each block to proceed. ####

    dual_elements = False

    objects = ['the entrance of a convenience store', 'a clear glass door']
    locations = [[0.1, 0.9, 0.1, 0.9], [0.3, 0.8, 0.25, 0.75]]
    opacity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # objects = ['sky', 'forest', 'lake with crystal clear water', 'reflection']
    # locations = [[0, 0.5, 0, 1], [0.25, 0.5, 0, 1],
    #              [0.5, 1, 0, 1], [0.5, 1, 0, 1]]
    # opacity = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    # objects = ['mountains with forest', 'fog']
    # locations = [[0.2, 1, 0, 1], [0, 0.8, 0, 1]]
    # opacity = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    # objects = ['a night', 'a cathedral', 'lush palm trees']
    # locations = [[0, 1, 0, 1], [0.2, 0.8, 0.2, 0.8], [0.4, 0.9, 0, 1]]
    # opacity = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    # objects = ['seaside reefs and turbulent waves at dusk', 'long exposure effects']
    # locations = [[0, 1, 0, 1], [0, 1, 0, 1]]
    # opacity = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    # dual_elements = True # dual elements for this rain & sunlight example
    # objects = ['city view with street and cars', 'rain', 'sunlight']
    # locations = [[0, 1, 0, 1], [0, 1, 0, 1], [0, 0.4, 0.6, 1]]
    # opacity = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    base_alpha = [0.8 for _ in objects]
    alpha_list = []
    if dual_elements:
        for opa1 in opacity:
            for opa2 in opacity:
                alpha_list.append(base_alpha[:-2] + [opa1, opa2])
    else:
        for opa in opacity:
            alpha_list.append(base_alpha[:-1] + [opa])

    ################ End of experimental settings ################

    assert all(
        0 <= a < 1 for alpha in alpha_list for a in alpha), "alpha values must be in [0, 1)"
    # parse the indices of subject tokens
    if use_dependency_parsing:
        DP = DependencyParsing()
        indices, subjects = zip(
            *[DP.parse(obj, ignore_dp_failure) for obj in objects])
        assert all(len(ind) > 0 for ind in indices), \
            f"Dependency Parsing error: subject indices contain empty cases, please rephrase your prompts or set ignore_dp_failure=True, indices: {indices}"
        indices = [[ind + 1 for ind in ind_list]
                   for ind_list in indices]  # will include a <start> token at 0
        subjects = ['+'.join(sub) for sub in subjects]
    else:
        indices = [[1], [1], [1], [1]]  # manually assigned if not using DP
        # manually assigned if not using DP
        subjects = ['sky', 'forest', 'lake', 'reflection']

    save_name = f"larender_{pretrain}_{'-'.join(subjects)}"

    # init LaRender
    LR = LaRender(pretrain=pretrain,
                  transmittance_use_bbox=transmittance_use_bbox,
                  transmittance_use_attn_map=transmittance_use_attn_map,
                  density_scheduler=density_scheduler)

    # run
    start_time = time.time()
    for alpha in alpha_list:
        images = LR.run(
            objects,
            locations,
            alpha,
            indices,
            num_images_per_prompt,
            height,
            width,
            negative_prompt,
            seed)
        print("Elapsed time:", time.time() - start_time, "seconds")
        os.makedirs(f"results/{save_name}", exist_ok=True)

        # save results
        for i, img in enumerate(images):
            if dual_elements:
                save_fn = f"results/{save_name}/{save_name}_{i:02d}_{alpha[-2]:.2f}_{alpha[-1]:.2f}.jpg"
            else:
                save_fn = f"results/{save_name}/{save_name}_{i:02d}_{alpha[-1]:.2f}.jpg"
            print(f'Save at: {save_fn}')
            img.save(save_fn)

            if save_bbox_image:
                draw = ImageDraw.Draw(img)
                width, height = img.width, img.height
                for obj, loc in zip(objects, locations):
                    draw.rectangle((loc[2] * width, loc[0] * height, loc[3]
                                   * width, loc[1] * height), outline='red', width=2)
                    font = ImageFont.load_default(size=32)
                    text_position = (loc[2] * width + 1, loc[0] * height + 1)
                    draw.text(text_position, obj,
                              font=font, fill="red")  # Text
                if dual_elements:
                    img.save(
                        f"results/{save_name}/{save_name}_box_{i:02d}_{alpha[-2]:.2f}_{alpha[-1]:.2f}.jpg")
                else:
                    img.save(
                        f"results/{save_name}/{save_name}_box_{i:02d}_{alpha[-1]:.2f}.jpg")
