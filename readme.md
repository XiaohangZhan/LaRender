## LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering

This repository is the official implementation of our [LaRender](https://xiaohangzhan.github.io/projects/larender/) accepted by ICCV 2025 as oral presentation.

> [**LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering**](https://xiaohangzhan.github.io/projects/larender/)

> [Xiaohang Zhan](https://xiaohangzhan.github.io/), 
> Dingming Liu
> <br>**Tencent**<br>

> Project page: https://xiaohangzhan.github.io/projects/larender/

> Paper: https://arxiv.org/pdf/2508.07647

## Introduction

This paper proposes plug and play training-free method to control object occlusion relationships and strength of visual effects in pre-trained text to image models, via simply replacing the cross-attention layers to Latent Rendering layers.


## Get Started

### Environments

```bash
git clone git@github.com:XiaohangZhan/larender.git
cd larender
conda create -n larender python==3.10
conda activate larender
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run

1. Object occlusion control demos.
```shell
python demos_occlusion.py
```

2. Visual effect strength control demos.
```shell
python demos_opacity.py
```

Results saved in ``results/``.

### Tips

1. If multiple objects have similar semantics or appearance (like girl and boy, cat and dog), or with close positioning, the results might suffer from subject neglect or mixing, which has been mentioned in this paper: https://arxiv.org/pdf/2411.18301 . 
Simple workarounds include:
   - **Modify the prompt** to describe the missing object as precisely as possible.  
     For example:  
     ❌ *"a cat"* → ✅ *"a white cat standing"*
     
   - **Mention the missing object** in the prompt of the other object.
     For example:  
     ❌ `["a girl in yellow dress", "a boy in blue T-shirt"]`  
     ✅ `["a girl in yellow dress next to a boy", "a boy in blue T-shirt"]`

2. Unreasonable bounding boxes will affect the occlusion accuracy. If the occlusion is wrong, please adjust the bounding boxes. Note that the bounding boxes are rough hints of positioning, we are not pursuing accurate bounding box control in this paper.