import os, sys
from   typing import Optional, List, Callable, Union, Dict, Any
from ..splits import get_splits, get_all_spaces


###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_hypersim.csv')
split_to_spaces = get_splits(split_file)
# subset_to_spaces = {'debug': ['ai_052_001'], 'fullplus': get_all_spaces(split_to_spaces) }

###############################################################################
#  Semantic segmentation:
###############################################################################

CLASS_LABELS = [
    'undefined', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
    'mirror', 'floor-mat', 'clothes', 'ceiling', 'books', 'fridge', 'TV', 'paper', 'towel', 
    'shower-curtain', 'box', 'white-board', 'person', 'night-stand', 'toilet', 'sink', 'lamp',
    'bathtub', 'bag', 'other-struct', 'other-furntr', 'other-prop'
]

CLASS_LABEL_TRANSFORM = [
    0, 116, 87, 62, 41, 38, 39, 42, 85, 119, 122, 98, 123, 68, 82, 102, 78, 124, 99, 125, 92, 74, 
    79, 55, 54, 44, 96, 112, 126, 69, 127, 128, 94, 43, 53, 90, 64, 8, 0, 0, 0
]

NYU40_COLORS = [
    [ 0,    0,   0], [174, 199, 232], [152, 223, 138], [ 31, 119, 180], [255, 187, 120], [188, 189,  34],
    [140,  86,  75], [255, 152, 150], [214,  39,  40], [197, 176, 213], [148, 103, 189], [196, 156, 148],
    [ 23, 190, 207], [178,  76,  76], [247, 182, 210], [ 66, 188, 102], [219, 219, 141], [140,  57, 197],
    [202, 185,  52], [ 51, 176, 203], [200,  54, 131], [ 92, 193,  61], [ 78,  71, 183], [172, 114,  82],
    [255, 127,  14], [ 91, 163, 138], [153,  98, 156], [140, 153, 101], [158, 218, 229], [100, 125, 154],
    [178, 127, 135], [120, 185, 128], [146, 111, 194], [ 44, 160,  44], [112, 128, 144], [ 96, 207, 209],
    [227, 119, 194], [213,  92, 176], [ 94, 106, 211], [ 82,  84, 163], [100,  85, 144]
]
