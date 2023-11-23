#from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
from mmcv import Config
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from mmcv.utils import print_log
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
print("aqui")
# config_file = '/users/a/araujofj/original_segformer/SegFormer/local_configs/segformer/B3/segformer.b3.512x512.lc.160k.py'

# config_file = "configs/swin/upernet_swin_base_patch4_window7_512x512_160k_cityscapes.py"
config_file = "configs/swinunet/swinunet_base_patch4_window7_512x512_160k_cityscapes.py"
cfg = Config.fromfile(config_file)


# @DATASETS.register_module()
# class Landcover(CustomDataset):
#   CLASSES = ('background', 'buildings', 'woods', 'water', 'roads')
#   def __init__(self, split, **kwargs):
#     super().__init__(img_suffix='.jpg', seg_map_suffix='_m.png', 
#                      split=split, **kwargs)
#     #assert osp.exists(self.img_dir) and self.split is not None

from mmseg.apis import set_random_seed

# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg

#for i in [0,1,2,3]:
#    cfg.model.auxiliary_head[i].norm_cfg = cfg.norm_cfg
#cfg.model.decode_head.bn = cfg.norm_cfg

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/swindbg'
# cfg.work_dir = './work_dirs/swinunet'

#cfg.runner.max_iters = 200
cfg.log_config.interval = 10
cfg.evaluation.interval = 2000
cfg.checkpoint_config.interval = 2000

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = [2]

#batch size and workers
cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu=2

print("config modified succesfully")

# Let's have a look at the final config used for training
#print(f'Config:\n{cfg.pretty_text}')


datasets = [build_dataset(cfg.data.train)]

# Build the detector


model = build_segmentor(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

print(model)


# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                meta=dict())