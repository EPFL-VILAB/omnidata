from   typing import Optional, List, Callable, Union, Dict, Any
import os
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
import multiprocessing as mp
from ...segment_instance import random_colors
from ...splits import get_splits
from ...omnidata_dataset import OmnidataDataset, load_subfolder

###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
subsets = ['debug', 'tiny', 'medium', 'full', 'fullplus']

forbidden_buildings = ['mosquito', 'tansboro', 'tomkins', 'darnestown', 'brinnon'] # We do not have the rgb data for tomkins, darnestown, brinnon
forbidden_buildings += ['rough'] # Contains some wrong viewpoints

taskonomy_split_files = {
        s:  
            os.path.join(os.path.dirname(__file__), 'train_val_test_{}.csv'.format(s.lower()))
        for s in subsets
}

# e.g. taskonomy_split_to_buildings['tiny']['train'] -> List[str] building names
taskonomy_split_to_buildings = {s: 
        get_splits(taskonomy_split_files[s], forbidden_buildings=forbidden_buildings) 
        for s in subsets
}

# e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
flat_split_to_spaces = {}
for subset in taskonomy_split_to_buildings:
    for split, buildings in taskonomy_split_to_buildings[subset].items():
        flat_split_to_spaces[subset + '-' + split] = buildings


###############################################################################
#  Semantic segmentation:
###############################################################################

CLASS_LABELS = [
    '__background__', 'bicycle', 'car', 'motorcycle',
    'boat', 'bench', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

CLASS_COLORS = random_colors(len(CLASS_LABELS), bright=True, seed=50)




###############################################################################
# Make dataset 
###############################################################################

    
class TaskonomyDataset(OmnidataDataset):
    def __init__(self, options: OmnidataDataset.Options):
        super().__init__(options)

    def make_task_dataset(self, task) -> List[str]:
        dirpath = os.path.join(self.data_path, task)
        folders = flat_split_to_spaces[f'{self.data_amount}-{self.split}']
        
        if folders is None:
            folders = os.listdir(dirpath)
        
        subfolder_paths = [os.path.join(dirpath, subfolder) for subfolder in folders if os.path.isdir(os.path.join(dirpath, subfolder))]

        # self.taskonomy_buildings = taskonomy_flat_split_to_buildings[f'{options.taskonomy_variant}-{self.split}']
        # self.replica_buildings = replica_flat_split_to_buildings[self.split]
        # self.gso_buildings = gso_flat_split_to_buildings[self.split]
        # self.hypersim_buildings = hypersim_flat_split_to_buildings[self.split]
    
        # TODO remove later
        if task == 'segment_semantic': dirpath = os.path.join(dirpath, '..', 'segment_panoptic')

        dirpath = os.path.expanduser(dirpath)
        if not os.path.isdir(dirpath):
            raise ValueError('Expected to find data directory in {dirpath}, but that is not a directory.')
        
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
        building = url.split('/')[-2]
        return building
