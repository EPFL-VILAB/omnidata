import os, sys, json
from ..splits import get_splits, get_all_spaces

###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_gso.csv')
split_to_spaces = get_splits(split_file)
subset_to_spaces = {'debug': ['frl_apartment_0-3'], 'fullplus': get_all_spaces(split_to_spaces) }

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

GSO_CLASS_LABELS = [
    'undefined', 'backpack', 'bag', 'bottle', 'cup', 'spoon', 'bowl', 'laptop','mouse', 
    'keyboard', 'toaster', 'clock', 'scissors', 'hair drier', 'basket', 'bin', 'box', 
    'camera', 'desk-organizer', 'kitchen-utensil', 'pan', 'plate', 'shoe', 'small-appliance', 
    'tablet', 'towel', 'utensil-holder', 'boot', 'hat', 'pencil case', 'teapot', 'headphones', 
    'hammer', 'gloves', 'tape', 'file sorter', 'mug', 'slipper', 'pill bottle', 'coffee maker', 
    'hard drive', 'speaker', 'cable', 'pitcher', 'flower pot', 'stuffed animal', 'plastic animal', 
    'toy', 'bucket', 'sponge', 'game console', 'pet food bowl', 'flashlight', 'video game', 
    'USB flash drive', 'dustpan', 'hair straightener', 'dish drainer', 'tube', 'can', 
    'food package', 'medicine package', 'game package', 'random package'
]

REPLICA_LABEL_TRANSFORM = [
    0, 6, 62, 63, 64, 0, 65, 41, 5, 1, 66, 67, 68, 55, 21, 69, 27, 70, 62, 71, 38, 72, 56, 73, 74, 
    75, 76, 48, 23, 77, 78, 79, 80, 81, 82, 83, 84, 85, 14, 86, 87, 8, 60, 88, 40, 25, 89, 90, 45,
    91, 92, 50, 93, 46, 94, 95, 0, 96, 97, 98, 0, 99, 0, 40, 100, 101, 102, 54, 47, 103, 104, 102,
    105, 106, 53, 107, 39, 108, 109, 110, 42, 73, 111, 96, 43, 61, 112, 44, 113, 7, 114, 57, 115, 116, 62, 
    117, 118, 119, 120, 121, 8, 74
]

gso_label_transform_file = os.path.join(os.path.dirname(__file__), 'GSO_LABEL_TRANSFORM.json')
with open(gso_label_transform_file) as file:
    GSO_LABEL_TRANSFORM = json.load(file)


###################
# GSO + replica 
###################
# class = 2**8 * r + g
# instance = b 
# number of classes : 102 (replica) + 1032 (google objects)
# class < 102 -> class in replica
# class >= 102 -> (class - 102) in GSO


GSO_NUM_CLASSES = len(REPLICA_CLASS_LABELS) + 1032

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