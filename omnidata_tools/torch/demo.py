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

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform


def depth_to_heatmap(img):   
    img = (img - np.min(img)) / np.ptp(img)
    cm = plt.get_cmap('viridis', 2**16)
    pixel_colored = np.uint8(np.rint(cm(img) * 255))[:, :, :3]
    return pixel_colored

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# get target task and model
if args.task == 'normal':
    image_size = 512
    pretrained_weights_path = root_dir + 'omnidata_rgb2normal_unet.pth'
    model = UNet(in_channels=3, out_channels=3)
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('model.', '')] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        get_transform('rgb', image_size=None)])

elif args.task == 'depth':
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_rgb2depth_dpt_hybrid.pth'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(512)])


def save_outputs(img_path, output_file_name):
    save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

    print(f'Reading input {img_path} ...')
    img = Image.open(img_path)

    img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

    rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
    trans_rgb(img).save(rgb_path)

    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3,1)

    output = model(img_tensor).clamp(min=0, max=1)

    if args.task == 'depth':
        output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
        output = 1 / (output + 1e-6)
        output = torch.tensor(depth_to_heatmap(output.detach().cpu().squeeze().numpy())).permute(2,0,1).unsqueeze(0)
        output = (output - output.min()) / (output.max() - output.min())

    trans_topil(output[0]).save(save_path)
    print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()