# GLAM: Full Contextual Attention for Multi-resolution Transformers in Semantic Segmentation
 
[Full Contextual Attention for Multi-resolution Transformers in Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2023/papers/Themyr_Full_Contextual_Attention_for_Multi-Resolution_Transformers_in_Semantic_Segmentation_WACV_2023_paper.pdf) official repo.
This repo contains the supported code and configuration files to reproduce semantic segmentaion results of GLAM. It is based on mmsegmentaion.


The 3D medical images segmentation code of this paper is available [here].


### Installation

Please refer to [Get Started](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/overview.md) for installation and dataset preparation.

You can find the pretrained models on the [Swin repo](https://github.com/microsoft/Swin-Transformer)

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# Or
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> --launcher=<LAUNCHER>

```
For example, to train an GLAM-Swin-UperNet Tiny with 10 global tokens on 8 gpus, run:
```
python tools/train.py configs/glam_swin_upernet/glam_swin_upernet_g10_tiny_patch4_window7_512x512_160k_ade20k.py --gpus 8 --options model.pretrained=pretrained_models/swin_tiny_patch4_window7_224.pth
```

## Citing Swin Transformer
```
@InProceedings{Themyr_2023_WACV,
    author    = {Themyr, Loic and Rambour, Cl\'ement and Thome, Nicolas and Collins, Toby and Hostettler, Alexandre},
    title     = {Full Contextual Attention for Multi-Resolution Transformers in Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3224-3233}
}
```