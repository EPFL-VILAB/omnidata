import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys
import pdb

from data.refocus_augmentation import RefocusImageAugmentation
from data.transforms import get_transform

parser = argparse.ArgumentParser(description='Visualize 3D refocus augmentation')

parser.add_argument('--num_quantiles', dest='num_quantiles', \
    help="number of qualtiles to use in blur stack. More is better, but slower", required=False)
parser.set_defaults(num_quantiles=10)

parser.add_argument('--min_aperture', dest='min_aperture', help="smallest aperture to use", required=False)
parser.set_defaults(min_aperture=0.001)

parser.add_argument('--max_aperture', dest='max_aperture', help="largest aperture to use", required=False)
parser.set_defaults(max_aperture=6)

parser.add_argument('--input_path', dest='input_path', \
    help="path to folder containing rgb and depth_euclidean")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path',\
    help="path to where refocused rgb should be stored")
    
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

transform_rgb = get_transform('rgb', image_size=512)
transform_depth = get_transform('depth_euclidean', image_size=512)
trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {args.output_path}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

refocus_aug = RefocusImageAugmentation(
    args.num_quantiles, float(args.min_aperture), float(args.max_aperture), return_segments=False)


def save_outputs(img_path, output_file_name):
    if not output_file_name.__contains__('rgb'): return
    rgb_path = img_path
    depth_path = img_path.replace('rgb', 'depth_euclidean')
    save_path = args.output_path+'/'+output_file_name+'_refocused'+'.png'

    print(f'Reading input {img_path} ...')
    rgb = Image.open(rgb_path)
    rgb_tensor = transform_rgb(rgb)[:3].unsqueeze(0).to(device)
    depth = Image.open(depth_path)
    depth_tensor = transform_depth(depth)[:1].unsqueeze(0).to(device)

    if rgb_tensor.shape[1] == 1:
        rgb_tensor = rgb_tensor.repeat_interleave(3,1)

    augmented_rgb = refocus_aug(rgb_tensor, depth_tensor)

    print(f'Writing output {save_path} ...')
    trans_topil(augmented_rgb[0]).save(save_path)


input_path = Path(args.input_path)

if input_path.is_dir():
    for f in glob.glob(args.input_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()