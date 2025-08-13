import os
import time
from PIL import ImageDraw, ImageFont
from utils import DependencyParsing
from larender_sdxl import LaRender


if __name__ == "__main__":
    # basic params
    negative_prompt = ''
    pretrain = 'IterComp'  # IterComp or GLIGEN
    num_images_per_prompt = 10  # set to 1 for speed test
    height = 1024
    width = 1024
    save_bbox_image = True
    seed = 5

    # dependency parsing params
    use_dependency_parsing = True
    ignore_dp_failure = True

    # larender params
    transmittance_use_bbox = True
    transmittance_use_attn_map = True
    density_scheduler = 'inverse_proportional'  # other choices: unchanged, opaque

    #### Reproducible experiments, uncomment each block to proceed. ####

    # Multiple "next to" are used here, because the cat is easily mixed into the giraffe, and the piano easily get lost. Never use "behind" or "in front of" to avoid introducing occlusion hints.
    objects = ["a large empty living room", "a brown piano", "a giraffe is standing next to a piano and a cat",
               "a sofa", "a ginger cat sitting", "a blue teddy bear next to a cat"]
    locations = [[0, 1, 0, 1], [0.4, 0.7, 0.2, 0.9], [0, 0.8, 0, 0.5],  [0.6, 0.9, 0.05, 0.95],
                 [0.5, 0.75, 0.5, 0.7], [0.5, 0.9, 0.7, 0.95]]  # mode: [y1, y2, x1, x2], range: 0~1

    # objects = ["forest", 'a white cat standing on the ground next to a branch',
    #            'a brown dog sitting on the ground next to a branch', 'a long branch']
    # locations = [[0, 1, 0, 1], [0.3, 0.8, 0.1, 0.4],
    #              [0.3, 0.8, 0.5, 0.8], [0.5, 0.6, 0, 1]]

    # objects = ["forest", 'a white cat standing on the ground next to a branch',
    #            'a long branch', 'a brown dog sitting next to a branch']
    # locations = [[0, 1, 0, 1], [0.3, 0.8, 0.1, 0.4],
    #              [0.5, 0.6, 0, 1], [0.3, 0.8, 0.5, 0.8]]

    # objects = ["forest", 'a brown dog sitting',
    #            'a long branch', 'a white cat standing']
    # locations = [[0, 1, 0, 1], [0.3, 0.8, 0.5, 0.8],
    #              [0.5, 0.6, 0, 1], [0.3, 0.8, 0.1, 0.4]]

    # objects = ["forest", 'a long branch', 'a brown dog sitting',
    #             'a white cat standing']
    # locations = [[0, 1, 0, 1],[0.5, 0.6, 0, 1], [0.3, 0.8, 0.55, 0.8],
    #              [0.3, 0.8, 0.2, 0.4]]

    # objects = ['a giraffe', 'an airplane on the ground']
    # locations = [[0.2, 0.8, 0.4, 0.6], [0.5, 0.8, 0, 1]]

    # objects = ['a house', 'lawn', 'a giant moon']
    # locations = [[0.3, 0.7, 0.1, 0.9], [0.7, 1, 0, 1], [0.55, 0.8, 0.2, 0.45]]

    # objects = ['a cyan vase', 'a yellow clock']
    # locations = [[0.1, 0.8, 0.2, 0.5], [0.4, 0.8, 0.3, 0.7]] # try pretrain = 'GLIGEN' if bbox not accurate

    # objects = ['a yellow clock', 'a cyan vase']
    # locations = [[0.4, 0.8, 0.3, 0.7], [0.1, 0.8, 0.2, 0.5]]

    # objects = ['a girl', 'a refrigerator']
    # locations = [[0.2, 0.9, 0.4, 0.7], [0.3, 0.9, 0.4, 0.7]]

    # objects = ['a refrigerator', 'a girl']
    # locations = [[0.2, 0.9, 0.4, 0.7], [0.2, 0.9, 0.3, 0.6]]

    # objects = ['fence', 'a man', 'a cow next to a man']
    # locations = [[0.4, 1, 0, 1], [0.1, 1, 0.1, 0.5], [0.2, 1, 0.3, 0.8]]

    # objects = ['a cow next to a man', 'a man wearing white shirt', 'fence']
    # locations = [[0.2, 1, 0.3, 0.8], [0.1, 1, 0.1, 0.5], [0.4, 1, 0, 1]]

    # objects = ['park', 'trees', 'a fountain', 'a lion statue', 'bush']
    # locations = [[0, 1, 0, 1], [0, 0.7, 0, 1], [0, 0.7, 0.3, 0.7], [0.4, 0.7, 0.5, 0.8], [0.6, 0.9, 0.1, 0.9]]

    # objects = ['a brown teddy bear', 'a computer']
    # locations = [[0.4, 0.8, 0.2, 0.5], [0.3, 0.8, 0.3, 0.8]]

    # objects = ['a computer', 'a brown teddy bear']
    # locations = [[0.3, 0.8, 0.3, 0.8], [0.4, 0.8, 0.2, 0.5]]

    # objects = ['a piano', 'a giant Yamaha guitar']
    # locations = [[0.5, 0.9, 0.2, 0.8], [0.1, 0.7, 0.4, 0.6]]

    # objects = ['a giant Yamaha guitar', 'a piano']
    # locations = [[0.1, 0.7, 0.4, 0.6], [0.5, 0.9, 0.2, 0.8]]

    # objects = ['a boat', 'a bear']
    # locations = [[0.4, 0.8, 0.1, 0.9], [0.2, 0.8, 0.3, 0.6]]

    # objects = ['a bear', 'a boat']
    # locations = [[0.2, 0.8, 0.3, 0.6], [0.4, 0.8, 0.1, 0.9]]

    # objects = ['a girl in yellow dress next to a boy', 'a boy in blue T-shirt']
    # locations = [[0.2, 1, 0.2, 0.6], [0.1, 1, 0.4, 0.8]]

    # objects = ['a boy in blue T-shirt next to a girl', 'a girl in yellow dress']
    # locations = [[0.1, 1, 0.4, 0.8], [0.2, 1, 0.2, 0.6]]

    # for non-transparent objects, we can always set 0.8
    alpha = [0.8 for _ in objects]

    ################ End of experimental settings ################

    assert all(0 <= a < 1 for a in alpha), "alpha values must be in [0, 1)"
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
        save_fn = f"results/{save_name}/{save_name}_{i:02d}.jpg"
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
                draw.text(text_position, obj, font=font, fill="red")  # Text
            img.save(f"results/{save_name}/{save_name}_box_{i:02d}.jpg")
