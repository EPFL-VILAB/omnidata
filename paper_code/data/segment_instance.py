import torch
import numpy as np
import colorsys
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import patches


TASKONOMY_CLASS_LABELS = [
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


COMBINED_CLASS_LABELS = [
    '__background__', 'bicycle', 'car', 'motorcycle', 'boat', 'bench', 'backpack',
    'umbrella', 'bag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'cabinet', 'basket', 'bathtub',
    'beanbag', 'bin', 'blanket', 'blinds', 'box', 'camera', 'candle', 'chopping-board','cloth', 'clothing',
    'coaster', 'comforter', 'cushion', 'curtain', 'ceiling', 'cooktop', 'countertop', 'desk', 'desk-organizer',
    'desktop-computer', 'door', 'faucet', 'floor', 'handrail', 'kitchen-utensil', 'lamp', 'major-appliance',
    'mat', 'monitor', 'nightstand', 'pan', 'paper', 'phone', 'picture', 'pillow', 'plate', 'pot', 'shelf',
    'scarf', 'sculpture', 'shoe', 'shower-stall', 'small-appliance', 'stair', 'stool', 'switch', 'tablet',
    'towel', 'tv-stand', 'utensil-holder', 'vent', 'wall', 'wall-plug', 'wardrobe', 'window', 'rug','logo', 
    'bookshelf', 'counter', 'dresser', 'mirror', 'shower-curtain', 'white-board', 'person'
]

REPLICA_LABEL_TRANSFORM = [
    0, 6, 62, 63, 64, 0, 65, 41, 5, 1, 66, 67, 68, 55, 21, 69, 27, 70, 62, 71, 38, 72, 56, 73, 74, 
    75, 76, 48, 23, 77, 78, 79, 80, 81, 82, 83, 84, 85, 14, 86, 87, 8, 60, 88, 40, 25, 89, 90, 45,
    91, 92, 50, 93, 46, 94, 95, 0, 96, 97, 98, 0, 99, 0, 40, 100, 101, 102, 54, 47, 103, 104, 102,
    105, 106, 53, 107, 39, 108, 109, 110, 42, 73, 111, 96, 43, 61, 112, 44, 113, 7, 114, 57, 115, 116, 62, 
    117, 118, 119, 120, 121, 8, 74
]

HYPERSIM_CLASS_LABELS = [
    'undefined', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
    'mirror', 'floor-mat', 'clothes', 'ceiling', 'books', 'fridge', 'TV', 'paper', 'towel', 
    'shower-curtain', 'box', 'white-board', 'person', 'night-stand', 'toilet', 'sink', 'lamp',
    'bathtub', 'bag', 'other-struct', 'other-furntr', 'other-prop'
]

HYPERSIM_LABEL_TRANSFORM = [
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
    [227, 119, 194], [213,  92, 176], [ 94, 106, 211], [ 82,  84, 163], [100,  85, 144]]


# GSO:
# class = 2**8 * r + g
# instance = b 
# number of classes : 102 (replica) + 1032 (google objects)
GSO_NUM_CLASSES = len(REPLICA_CLASS_LABELS) + 1032

def random_colors(N, bright=True, seed=0):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    random.seed(seed)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

TASKONOMY_CLASS_COLORS = random_colors(len(TASKONOMY_CLASS_LABELS), bright=True, seed=50)
REPLICA_CLASS_COLORS = random_colors(len(REPLICA_CLASS_LABELS), bright=True, seed=50)
HYPERSIM_CLASS_COLORS = random_colors(len(NYU40_COLORS), bright=True, seed=99)
GSO_CLASS_COLORS = random_colors(GSO_NUM_CLASSES, bright=True, seed=99)
COMBINED_CLASS_COLORS = random_colors(len(COMBINED_CLASS_LABELS), bright=True, seed=99)


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width]. Mask pixels are either 1 or 0.
    Returns: bbox array (y1, x1, y2, x2).
    """
    # Bounding box.
    horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
    vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = torch.Tensor([x1, y1, x2, y2]).to(mask.device)
    return boxes


def extract_instances(img):    
    '''
    
    Returns:
        boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
        labels (Int64Tensor[N]): the class label for each ground-truth box
        masks (BoolTensor[N, H, W]): the segmentation binary masks for each instance
    '''
    #img = torch.Tensor(img)
    img = img.permute(2,0,1) if img.shape[2] == 3 else img
    img[img == 255] = 0
    
    # Return None if image does not contain any segmentations
    if torch.all(img[:,:,0] == 0):
        return None, None, None
    
    # Red channel: Encodes class indices
    # Green & Blue channels: Encode instance indices as 2^8 * G + B
    class_map, G, B = img
    instance_map = 2**8 * G + B

    all_class_idxs = torch.unique(class_map).tolist()
    if 0 in all_class_idxs:
        all_class_idxs.remove(0)

    boxes = []
    labels = []
    masks = []
    
    for class_idx in all_class_idxs:
        class_mask = torch.zeros_like(class_map)
        class_mask[class_map == class_idx] = 1
        
        class_instance_idxs = torch.unique(instance_map[class_mask == 1]).tolist()
        if 0 in class_instance_idxs:
            class_instance_idxs.remove(0)
        
        for instance_idx in class_instance_idxs:
            class_instance_mask = class_mask.clone()
            class_instance_mask[instance_map != instance_idx] = 0
            class_instance_mask = class_instance_mask.bool()
            
            masks.append(class_instance_mask)
            boxes.append(extract_bboxes(class_instance_mask))
            labels.append(class_idx)
    
    if len(boxes) == 0 or len(labels) == 0 or len(masks) == 0:
        return None, None, None

    boxes = torch.stack(boxes)
    labels = torch.LongTensor(labels).to(img.device)
    masks = torch.stack(masks)
    
    return boxes, labels, masks


def apply_mask(image, mask, color, alpha=0.5, mask_threshold=0.5):
    """Apply the given mask to the image.
    """
    if len(np.unique(mask)) > 2:
        mask = (mask >= mask_threshold).astype(int)
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            image[:, :, c]
        )
    return image


def plot_instances_mpl(
        img, boxes, masks, labels, scores=None, 
        alpha=0.5, mask_threshold=0.9, score_threshold=0.1,
        box_color='r'
    ):
    img = img.copy()

    fig, ax = plt.subplots(figsize=(10,10))
    
    for instance_idx in range(len(boxes)):
        if scores is not None and scores[instance_idx] < score_threshold:
            continue
        color = CLASS_COLORS[labels[instance_idx]]
        img = apply_mask(img, masks[instance_idx], color=color)
        xy = [boxes[instance_idx,0], boxes[instance_idx,1]]
        h = boxes[instance_idx,2]-boxes[instance_idx,0]
        w = boxes[instance_idx,3]-boxes[instance_idx,1]
        rect = patches.Rectangle(xy, h, w, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xy[0]+1, xy[1]+3, CLASS_LABELS[labels[instance_idx]])
        
    im = plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

def plot_instances(
        img, boxes, masks, labels, scores=None, 
        alpha=0.5, mask_threshold=0.9, score_threshold=0.1,
        box_color='r', plot_scale_factor=2, return_PIL=False
    ):
    img = img.copy()
    
    for instance_idx in range(len(boxes)):
        if scores is not None and scores[instance_idx] < score_threshold:
            continue
        color = CLASS_COLORS[labels[instance_idx]]
        img = apply_mask(img, masks[instance_idx], color=color)
        
    img = Image.fromarray((255*img).astype(np.uint8))
    width, height = img.size
    img = img.resize((width*plot_scale_factor, height*plot_scale_factor))
    draw = ImageDraw.Draw(img)
    
    for instance_idx in range(len(boxes)):
        if scores is not None and scores[instance_idx] < score_threshold:
            continue
        draw.rectangle(boxes[instance_idx]*plot_scale_factor, outline='red')
        draw.text(boxes[instance_idx,[0,1]]*plot_scale_factor, CLASS_LABELS[labels[instance_idx]], fill='red')
    
    if return_PIL:
        return img
    else:
        return np.array(img) / 255.0