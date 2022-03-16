import torch
import numpy as np
import colorsys
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import patches

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