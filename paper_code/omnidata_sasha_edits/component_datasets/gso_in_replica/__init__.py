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
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_gso.csv')
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
    def make_task_dataset(self, task):
        dirpath = self.data_path
        folders = flat_split_to_spaces[f'{self.split}']
        if task == 'segment_semantic': task = 'semantic'

        if folders is None:
            folders = os.listdir(dirpath)
        
        subfolder_paths = [
            os.path.join(dirpath, folder.split('-')[0], folder.split('-')[1], task)  
            for folder in folders
            if os.path.join(dirpath, folder.split('-')[0], folder.split('-')[1], task) and folder not in REPLICA_BUILDINGS
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
    
    # def make_task_dataset(self, task):
    #     dir = os.path.join(self.data_path, task)
    #     folders = flat_split_to_spaces[f'{self.split}']
    #     if task == 'segment_semantic': task = 'semantic'
    #     #  folders are building names. 
    #     images = []
    #     dir = os.path.expanduser(dir)
    #     if not os.path.isdir(dir):
    #         assert "bad directory"
        
    #     if folders is None:
    #         folders = os.listdir(dir)

    #     for folder in folders:
    #     assert folder not in REPLICA_BUILDINGS # gso dataset e.g. apartment_0-3, apartment_0-6, apartment_0-15
    #     folder_path = os.path.join(dir, folder.split('-')[0], folder.split('-')[1], task)            
    #     for fname in sorted(os.listdir(folder_path)):
    #             path = os.path.join(folder_path, fname)
    #             images.append(path)

    #     return images

    def _get_building(self, url):
        building = url.split('/')[-4] + '-' + url.split('/')[-3]
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
