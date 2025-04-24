import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

ADT_CLS_NAMES = [
    'Aluminum Sheet Defect Detection using YOLO on VOC dataset', 
]
ADT_ROOT = "/home/datasets/zk/AD_test/"

class ADTDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=ADT_CLS_NAMES, aug_rate=0.2, root=ADT_ROOT, training=True):
        super(ADTDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
