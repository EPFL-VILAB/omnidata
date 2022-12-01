import torch
from   torchvision import transforms
import logging
import os
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
import multiprocessing as mp
from PIL import Image
from ...segment_instance import random_colors
from ...splits import get_splits
from ...omnidata_dataset import OmnidataDataset, load_subfolder

###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_replica_gso.csv')
flat_split_to_spaces = get_splits(split_file)

###############################################################################
#  Semantic segmentation:
###############################################################################

REPLICA_CLASS_LABELS = [
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


# GSO:
# class = 2**8 * r + g
# instance = b 
# number of classes : 102 (replica) + 1032 (google objects)


GSO_NUM_CLASSES = len(REPLICA_CLASS_LABELS) + 1032
CLASS_COLORS = random_colors(GSO_NUM_CLASSES, bright=True, seed=99)


###############################################################################
# Make dataset 
###############################################################################

REPLICA_BUILDINGS = [
    'apartment_0',
    'apartment_1', 
    'apartment_2',
    'frl_apartment_0',
    'frl_apartment_1',
    'frl_apartment_2',
    'frl_apartment_3',
    'frl_apartment_4',
    'frl_apartment_5',
    'office_0',
    'office_1',
    'office_2',
    'office_3',
    'office_4',
    'room_1'
    'room_0',
    'room_2',
    'hotel_0',
]



class GSOReplicaDataset(OmnidataDataset):

    def __init__(self, options: OmnidataDataset.Options, logger: logging.Logger=None):
        self.dataset_name = 'replica_gso'
        super().__init__(options, logger)
        # for task, _transform in self.transform.items():
        #     if task == 'semantic':
        #         _new_transform = transforms.Compose([
        #             _transform,
        #             semseg_remap_inplace
        #         ])
        #         self.transform[task] = _new_transform
