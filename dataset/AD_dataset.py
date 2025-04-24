import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

AD_CLS_NAMES = [
    'Aluminum_Sheet_Defect_Detection_using_YOLO_on_VOC_dataset', 'Asphalt_Road', 'battery', 'cloth_black', 'cloth_blue',
    'Concrete_Pavement', 'defects_in_aluminum_materials_blue', 'defects_in_aluminum_materials_white', 'Gravel_Road', 'hollow_iron_filings',
    'hollow_steels', 'NEU_DET', 'pige', 'Prailway', 'RSW_voc',
    'Tiles_white', 'Tiles_yellow', 'Welding_Surface_Defect_Dataset_in_VOC_Format', 'wire_rope', 'wood',
]
AD_ROOT = "/home/datasets/zk/AD_dataset/"

class ADDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=AD_CLS_NAMES, aug_rate=0.2, root=AD_ROOT, training=True):
        super(ADDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
