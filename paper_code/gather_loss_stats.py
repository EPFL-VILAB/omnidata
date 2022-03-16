import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import describe
from tqdm import tqdm

from data.taskonomy_replica_gso_dataset import TaskonomyReplicaGsoDataset
from models.unet import UNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(Encoder, self).default(obj)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--datasets', type=str, default='taskonomy-replica-hypersim-gso-blendedMVS',
    help='Dataset names to use, separated by hyphen. (default: taskonomy-replica-hypersim-gso-blendedMVS)')
parser.add_argument(
    '--splits', type=str, default='train-val-test',
    help='Dataset splits to use, separated by hyphen. (default: train-val-test)')
parser.add_argument(
    '--weights', type=str, 
    default='/scratch/roman/pretrained/taskonomy_unet/rgb2normal_omnidata_mvs.pth',
    help='Path to normals UNet weight. (default: omnidata normals)')
parser.add_argument(
    '--save_dir', type=str, default='./loss_stats',
    help='Relative path to save statistics in. (default: ./loss_stats)')
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Batch size. (default: 64)')
parser.add_argument(
    '--num_workers', type=int, default=32,
    help='Number of dataset threads. (default: 32)')
parser.add_argument(
    '--image_size', type=int, default=512,
    help='Image size. (default: 512)')
args = parser.parse_args()


# Save losses and statistics in this folder
os.makedirs(args.save_dir, exist_ok=True)

# Create datasets
dataset_names = args.datasets.split('-')
splits = args.splits.split('-')

datasets = []
for dataset_name in dataset_names:
    for split in splits:
        print(f'Preparing {dataset_name} / {split}')
        options = TaskonomyReplicaGsoDataset.Options(
            taskonomy_variant='fullplus',
            split=split,
            tasks=['rgb', 'normal', 'mask_valid'],
            datasets=[dataset_name],
            transform='DEFAULT',
            image_size=args.image_size,
            normalize_rgb=False,
            randomize_views=False
        )
        dataset = TaskonomyReplicaGsoDataset(options)
        dataset_dict = {
            'dataset': dataset,
            'name': dataset_name,
            'split': split
        }
        datasets.append(dataset_dict)


# Load omnidata normals model
model_normal_omni = UNet(in_channels=3, out_channels=3, downsample=6).to(device).eval()
checkpoint = torch.load(args.weights)
model_normal_omni.load_state_dict(checkpoint)


# Loop over entire dataset/split and compute loss statistics
for dataset_dict in datasets:
    loader = DataLoader(
        dataset_dict['dataset'], batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, pin_memory=False
    )
    
    normal_losses = {}
    
    pbar = tqdm(total=len(loader), desc=f'{dataset_dict["name"]} / {dataset_dict["split"]}')
    for idx, batch in enumerate(loader):
        rgb = batch['positive']['rgb']
        normal_gt = batch['positive']['normal']
        
        mask = 1 - batch['positive']['mask_valid']
        mask = F.max_pool2d(mask, kernel_size=4)
        mask = F.interpolate(mask, (args.image_size, args.image_size), mode='nearest')
        mask = mask == 0
        
        buildings = batch['positive']['building']
        points = batch['positive']['point']
        views = batch['positive']['view']
        
        # Forward pass
        with torch.no_grad():
            normal_preds = model_normal_omni(rgb.to(device)).cpu().detach()
        normal_preds = torch.clamp(normal_preds, 0, 1)
            
        for batch_idx in range(len(rgb)):
            if not torch.all(~mask[batch_idx]).item():
                pixel_losses = abs(normal_gt[batch_idx] - normal_preds[batch_idx])[mask[batch_idx].repeat_interleave(3,0)]
            else:
                pixel_losses = abs(normal_gt[batch_idx] - normal_preds[batch_idx]).flatten()
            stats_normal = describe(pixel_losses.numpy(), nan_policy='omit')
            del pixel_losses

            building, point, view = buildings[batch_idx], points[batch_idx], views[batch_idx]

            if building not in normal_losses:
                normal_losses[building] = {}

            if point not in normal_losses[building]:
                normal_losses[building][point] = {}
            
            normal_losses[building][point][view] = {
                'nobs': stats_normal.nobs,
                'min': stats_normal.minmax[0],
                'max': stats_normal.minmax[1],
                'mean': stats_normal.mean,
                'variance': stats_normal.variance,
                'skewness': stats_normal.skewness,
                'kurtosis': stats_normal.kurtosis
            }
            del stats_normal
                        
        pbar.update(1)
    pbar.close()
    
    json_path = os.path.join(args.save_dir, f'{dataset_dict["name"]}_{dataset_dict["split"]}.json')
    with open(json_path, 'w') as f:
        json.dump(normal_losses, f, cls=Encoder)
    print('Saved loss statistics at', json_path)