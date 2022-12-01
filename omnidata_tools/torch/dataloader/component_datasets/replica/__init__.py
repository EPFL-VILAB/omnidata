import logging
import os
import torch
from   torchvision import transforms
from ...segment_instance import random_colors
from ...splits import get_splits
from ...omnidata_dataset import OmnidataDataset, load_subfolder

###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_replica.csv')
flat_split_to_spaces = get_splits(split_file)

###############################################################################
#  Semantic segmentation:
###############################################################################

CLASS_LABELS = [
    'undefined', 'backpack', 'base-cabinet', 'basket', 'bathtub', 'beam', 'beanbag', 'bed', 'bench', 'bike',
    'bin', 'blanket', 'blinds', 'book', 'bottle', 'box', 'bowl', 'camera', 'cabinet', 'candle', 'chair',
    'chopping-board', 'clock', 'cloth', 'clothing', 'coaster', 'comforter', 'computer-keyboard', 'cup',
    'cushion', 'curtain', 'ceiling', 'cooktop', 'countertop', 'desk', 'desk-organizer', 'desktop-computer',
    'door', 'exercise-ball', 'faucet', 'floor', 'handbag', 'hair-dryer', 'handrail', 'indoor-plant',
    'knife-block', 'kitchen-utensil', 'lamp', 'laptop', 'major-appliance', 'mat', 'microwave', 'monitor',
    'mouse', 'nightstand', 'pan', 'panel', 'paper-towel', 'phone', 'picture', 'pillar', 'pillow', 'pipe',
    'plant-stand', 'plate', 'pot', 'rack', 'refrigerator', 'remote-control', 'scarf', 'sculpture', 'shelf',
    'shoe', 'shower-stall', 'sink', 'small-appliance', 'sofa', 'stair', 'stool', 'switch', 'table',
    'table-runner', 'tablet', 'tissue-paper', 'toilet', 'toothbrush', 'towel', 'tv-screen', 'tv-stand',
    'umbrella', 'utensil-holder', 'vase', 'vent', 'wall', 'wall-cabinet', 'wall-plug', 'wardrobe', 'window',
    'rug', 'logo', 'bag', 'set-of-clothing'
]

CLASS_LABEL_TRANSFORM = [
    0, 6, 62, 63, 64, 0, 65, 41, 5, 1, 66, 67, 68, 55, 21, 69, 27, 70, 62, 71, 38, 72, 56, 73, 74, 
    75, 76, 48, 23, 77, 78, 79, 80, 81, 82, 83, 84, 85, 14, 86, 87, 8, 60, 88, 40, 25, 89, 90, 45,
    91, 92, 50, 93, 46, 94, 95, 0, 96, 97, 98, 0, 99, 0, 40, 100, 101, 102, 54, 47, 103, 104, 102,
    105, 106, 53, 107, 39, 108, 109, 110, 42, 73, 111, 96, 43, 61, 112, 44, 113, 7, 114, 57, 115, 116, 62, 
    117, 118, 119, 120, 121, 8, 74
]

CLASS_COLORS = random_colors(len(CLASS_LABELS), bright=True, seed=50)



###############################################################################
# Make dataset 
###############################################################################



def semseg_remap_inplace(res):
    res2 = res.clone() 
    labels = torch.unique(res)
    for old_label in labels:
        if old_label == -1 or old_label == 255: continue
        res[res2 == old_label] = CLASS_LABEL_TRANSFORM[old_label]
    return res

class ReplicaDataset(OmnidataDataset):

    def __init__(self, options: OmnidataDataset.Options, logger: logging.Logger=None):
        self.dataset_name = 'replica'
        super().__init__(options, logger)
        # for task, _transform in self.transform.items():
        #     if task == 'semantic':
        #         _new_transform = transforms.Compose([
        #             _transform,
        #             semseg_remap_inplace
        #         ])
        #         self.transform[task] = _new_transform

    def _folder_in_split(self, folder, split):
        if self.data_amount == 'debug': return (folder == 'frl_apartment_0')
        else: return super()._folder_in_split(folder, split)
