import torch
from   torchvision import transforms
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

    def make_task_dataset(self, task):
        dirpath = self.data_path
        folders = flat_split_to_spaces[f'{self.split}']
        if task == 'segment_semantic': task = 'semantic'

        if folders is None:
            folders = os.listdir(dirpath)
        
        subfolder_paths = [
            os.path.join(dirpath, subfolder, task)
            for subfolder in folders
            if os.path.isdir(os.path.join(dirpath, subfolder, task))
        ]

        dirpath = os.path.expanduser(dirpath)
        if not os.path.isdir(dirpath):
            raise ValueError(f'Expected to find data directory in {dirpath}, but that is not a directory.')
        
        pool_size = self.n_workers
        if pool_size is None:
            pool_size = mp.cpu_count()
        if pool_size == 1:
            images = []
            for subfolder in tqdm.tqdm(subfolder_paths, desc=f'Loading {task} paths'):
                images.append(load_subfolder(subfolder))
            # images = [load_subfolder(subfolder) for subfolder in subfolder_paths]
        else:
            chunksize = min(1, len(subfolder_paths) // pool_size + 1)
            images = process_map(load_subfolder, subfolder_paths,
                desc=f'Loading {task} paths ({pool_size} workers)',
                max_workers=pool_size,
                chunksize=chunksize)
        images = sum(images, start=[])

        return images


    def _get_building(self, url):
        building = url.split('/')[-3]
        return building

    def __init__(self, options: OmnidataDataset.Options):
        super().__init__(options)
        for task, _transform in self.transform.items():
            if task == 'segment_semantic':
                _new_transform = transforms.Compose([
                    _transform,
                    semseg_remap_inplace
                ])
                self.transform[task] = _new_transform

