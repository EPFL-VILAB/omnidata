from   dataclasses import dataclass, field
from   typing import Optional, List, Callable, Union, Dict, Any
import os, sys
import os
import multiprocessing as mp
from ..splits import get_splits, get_all_spaces

###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
subsets = ['debug', 'tiny', 'medium', 'full', 'fullplus']

forbidden_buildings = ['mosquito', 'tansboro', 'tomkins', 'darnestown', 'brinnon', 'rough', 'woodbine'] # We do not have the rgb data for tomkins, darnestown, brinnon
# forbidden_buildings += ['rough'] # Contains some wrong viewpoints
# forbidden_buildings += ['wiconisco'] # Bad texture?
# forbidden_buildings = []

taskonomy_split_files = {
        s: os.path.join( os.path.dirname(__file__), 'train_val_test_{}.csv'.format(s.lower()) )
        for s in subsets
}

# e.g. taskonomy_split_to_buildings['tiny']['train'] -> List[str] building names
taskonomy_split_to_buildings = {
    s: get_splits(taskonomy_split_files[s], forbidden_buildings=forbidden_buildings) 
    for s in subsets
}

# e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
flat_split_to_spaces = {}
for subset in taskonomy_split_to_buildings:
    for split, buildings in taskonomy_split_to_buildings[subset].items():
        flat_split_to_spaces[subset + '-' + split] = buildings

split_to_spaces = taskonomy_split_to_buildings['fullplus']
subset_to_spaces = {subset: get_all_spaces(splits) for subset, splits in taskonomy_split_to_buildings.items()}
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
