# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations
from .custom import CustomDataset
from pycocotools.coco import COCO
import json


@DATASETS.register_module()
class TableSoccerDataset(CustomDataset):

    CLASSES = ('ball',)

    # Colours for Segmentation-Masks
    PALETTE = [[250, 250, 0]]

    def __init__(self, **kwargs):
        super(TableSoccerDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)


"""
    def __init__(self, ann_file, img_dir, **kwargs):
        super(TableSoccerDataset, self).__init__(img_dir=img_dir, **kwargs)
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.coco = COCO(self.ann_file)
        self.img_suffix = '.jpg'
        self.images = self.coco.loadImgs(self.coco.getImgIds())
        print(self.img_infos)



    def get_ann_info(self, idx):
        # print("Length img_infos " + str(len(self.img_infos)))
        print("IDX: " + str(idx))
        print(self.img_infos[idx])
        print(print(self.img_infos[idx]['segmentation']))
        return self.img_infos[idx]['segmentation']
"""

