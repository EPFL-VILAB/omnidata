import os
import argparse
import numpy as np
import random
import json
import math
from collections import defaultdict
from runstats import Statistics
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from einops import rearrange
import einops as ein
import torch 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.taskonomy_replica_gso_dataset import TaskonomyReplicaGsoDataset
# from data.nyu_dataset import NYUDataset
from models.unet import UNet
from losses import masked_l1_loss, compute_grad_norm_losses
from evaluation_metrics import get_metrics


class DepthTest(pl.LightningModule):
    def __init__(self,
                 pretrained_weights_path,
                 image_size,
                 batch_size,
                 num_workers,
                 taskonomy_variant,
                 taskonomy_root,
                 replica_root,
                 gso_root,
                 hypersim_root,
                 nyu_root,
                 use_taskonomy,
                 use_replica,
                 use_gso,
                 use_hypersim,
                 use_nyu,
                 model_name,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters(
            'image_size', 'batch_size', 'num_workers',
            'taskonomy_variant', 'taskonomy_root', 'replica_root', 'gso_root', 'hypersim_root',
            'use_taskonomy', 'use_replica', 'use_gso', 'use_hypersim',
            'pretrained_weights_path', 'experiment_name', 'restore', 'gpus', 'distributed_backend', 'precision'
        )
        self.pretrained_weights_path = pretrained_weights_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.gpus = kwargs['gpus']

        self.num_workers = num_workers

        self.taskonomy_variant = taskonomy_variant
        self.taskonomy_root = taskonomy_root
        self.replica_root = replica_root
        self.gso_root = gso_root
        self.hypersim_root = hypersim_root
        self.nyu_root = nyu_root
        self.use_taskonomy = use_taskonomy
        self.use_replica = use_replica
        self.use_gso = use_gso
        self.use_hypersim = use_hypersim
        self.use_nyu = use_nyu
        self.model_name = model_name
        self.save_debug_info_on_error = False

        self.setup_datasets()

        self.model = UNet(in_channels=3, out_channels=1)
        if self.pretrained_weights_path is not None:
            checkpoint = torch.load(self.pretrained_weights_path)
            # In case we load a checkpoint from this LightningModule
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k.replace('model.', '')] = v
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)

        self.metrics = defaultdict(Statistics)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--pretrained_weights_path', type=str, default=None,
            help='Path to pretrained UNet weights. Set to None for random init. (default: None)')
        parser.add_argument(
            '--image_size', type=int, default=512,
            help='Input image size. (default: 512)')
        parser.add_argument(
            '--batch_size', type=int, default=4,
            help='Batch size for data loader (default: 4)')
        parser.add_argument(
            '--num_workers', type=int, default=16,
            help='Number of workers for DataLoader. (default: 16)')
        parser.add_argument(
            '--taskonomy_variant', type=str, default='tiny',
            choices=['full', 'fullplus', 'medium', 'tiny', 'debug'],
            help='One of [full, fullplus, medium, tiny, debug] (default: fullplus)')
        parser.add_argument(
            '--taskonomy_root', type=str, default='/datasets/taskonomy',
            help='Root directory of Taskonomy dataset (default: /datasets/taskonomy)')
        parser.add_argument(
            '--replica_root', type=str, default='/scratch/ainaz/replica-taskonomized',
            help='Root directory of Replica dataset')
        parser.add_argument(
            '--gso_root', type=str, default='/scratch/ainaz/replica-google-objects',
            help='Root directory of GSO dataset.')
        parser.add_argument(
            '--hypersim_root', type=str, default='/scratch/ainaz/hypersim-dataset2/evermotion/scenes',
            help='Root directory of hypersim dataset.')
        parser.add_argument(
            '--nyu_root', type=str, default='/scratch/ainaz/replica-google-objects',
            help='Root directory of NYU dataset.')
        parser.add_argument(
            '--use_taskonomy', action='store_true', default=False,
            help='Set to use taskonomy dataset.')
        parser.add_argument(
            '--use_replica', action='store_true', default=False,
            help='Set to use replica dataset.')
        parser.add_argument(
            '--use_gso', action='store_true', default=True,
            help='Set to user GSO dataset.')
        parser.add_argument(
            '--use_hypersim', action='store_true', default=False,
            help='Set to user hypersim dataset.')
        parser.add_argument(
            '--use_nyu', action='store_true', default=False,
            help='Set to user NYU dataset.')
        parser.add_argument(
            '--model_name', type=str, default='taskonomy-tiny',
            help='Name of model used for testing.')
        return parser

    def setup_datasets(self):
        self.num_positive = 1 

        tasks = ['rgb', 'depth_zbuffer', 'mask_valid']
        self.test_datasets = []
        if self.use_taskonomy: self.test_datasets.append('taskonomy')
        if self.use_replica: self.test_datasets.append('replica')
        if self.use_gso: self.test_datasets.append('gso')
        if self.use_hypersim: self.test_datasets.append('hypersim')
        if self.use_nyu: self.test_datasets.append('nyu')

        if self.use_nyu:
            # self.testset = NYUDataset(root=self.nyu_root, type='val', task='depth_zbuffer')
            pass
        else:
            opt_test = TaskonomyReplicaGsoDataset.Options(
                taskonomy_data_path=self.taskonomy_root,
                replica_data_path=self.replica_root,
                gso_data_path=self.gso_root,
                hypersim_data_path=self.hypersim_root,
                split='test',
                taskonomy_variant=self.taskonomy_variant,
                tasks=tasks,
                datasets=self.test_datasets,
                transform='DEFAULT',
                image_size=self.image_size,
                num_positive=self.num_positive,
                normalize_rgb=False,
                load_building_meshes=False,
                force_refresh_tmp = False,
                randomize_views=False
            )
            self.testset = TaskonomyReplicaGsoDataset(options=opt_test)

        # self.testset.randomize_order(seed=10)

        print('Loaded test set:')
        print(f'Test set contains {len(self.testset)} samples.')


    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False
        )

    def forward(self, x):
        return self.model(x)


    def test_step(self, batch, batch_idx):        
        rgb = batch['positive']['rgb']
        depth_gt = batch['positive']['depth_zbuffer']
        # Forward pass
        depth_preds = self(rgb)

        depth_preds = torch.clamp(depth_preds, 0, 1)
        depth_gt = torch.clamp(depth_gt, 0, 1)

        # save samples
        if batch_idx % 4 == 0:
            pred = np.uint8(255 * depth_preds[0].cpu().permute((1, 2, 0)).numpy())
            gt = np.uint8(255 * depth_gt[0].cpu().permute((1, 2, 0)).numpy())
            rgb = np.uint8(255 * rgb[0].cpu().permute((1, 2, 0)).numpy())

            # transform = transforms.Resize(512, Image.BILINEAR)
            im = Image.fromarray(rgb)
            im.save(os.path.join('test_images', 'depth', self.test_datasets[0], f'{batch_idx}_rgb.png'))
            im = Image.fromarray(gt.squeeze(axis=2))
            im.save(os.path.join('test_images', 'depth', self.test_datasets[0], f'{batch_idx}_gt.png'))
            im = Image.fromarray(pred.squeeze(axis=2))
            im.save(os.path.join('test_images', 'depth', self.test_datasets[0], f'{batch_idx}_{self.model_name}_pred.png'))


        # Mask out invalid pixels and compute loss
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid'])
        for pred, target, mask in zip(depth_preds, depth_gt, mask_valid):
            # print("******** ", pred.max(), pred.min(), target.max(), target.min(), mask.max())
            metrics = get_metrics(pred.cpu().unsqueeze(0), target.cpu().unsqueeze(0), \
                masks=mask.cpu().unsqueeze(0), task='depth_zbuffer')
            for metric_name, metric_val in metrics.items(): 
                self.metrics[metric_name].push(metric_val)
    

    def make_valid_mask(self, mask_float, max_pool_size=4, return_small_mask=False):
        '''
            Creates a mask indicating the valid parts of the image(s).
            Enlargens masked area using a max pooling operation.

            Args:
                mask_float: A mask as loaded from the Taskonomy loader.
                max_pool_size: Parameter to choose how much to enlarge masked area.
                return_small_mask: Set to true to return mask for aggregated image
        '''
        if len(mask_float.shape) == 3:
            mask_float = mask_float.unsqueeze(axis=0)
        reshape_temp = len(mask_float.shape) == 5
        if reshape_temp:
            mask_float = rearrange(mask_float, 'b p c h w -> (b p) c h w')
        mask_float = 1 - mask_float
        mask_float = F.max_pool2d(mask_float, kernel_size=max_pool_size)
        mask_float = F.interpolate(mask_float, (self.image_size, self.image_size), mode='nearest')
        mask_valid = mask_float == 0
        if reshape_temp:
            mask_valid = rearrange(mask_valid, '(b p) c h w -> b p c h w', p=self.num_positive)

        return mask_valid



if __name__ == '__main__':
    # Experimental setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Experiment name for Weights & Biases. (default: None)')
    parser.add_argument(
        '--restore', type=str, default=None,
        help='Weights & Biases ID to restore and resume training. (default: None)')
    parser.add_argument(
        '--save-on-error', type=bool, default=True,
        help='Save crash information on fatal error. (default: True)')    
    parser.add_argument(
        '--save-dir', type=str, default='exps',
        help='Directory in which to save this experiments. (default: exps/)')    


    # Add PyTorch Lightning Module and Trainer args
    parser = DepthTest.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = DepthTest(**vars(args))
    # model = DepthTest.load_from_checkpoint(
    # checkpoint_path=args.pretrained_weights_path, **vars(args))

    trainer = Trainer.from_argparse_args(args, gpus=-1, accelerator='ddp')

    result =trainer.test(verbose=True, model=model)
    print(result)

    metrics = {}
    for metric_name, metric_val in model.metrics.items(): 
        print(f"\t{metric_name}: {metric_val.mean()} ({math.sqrt(metric_val.variance())})")
        metrics[metric_name] = metric_val.mean()
        metrics[metric_name + "_std"] = math.sqrt(metric_val.variance())

    print("metrics : ", metrics)

    os.makedirs(os.path.join('results'), exist_ok=True)
    metrics_file = os.path.join('results', f'metrics_depth_{model.test_datasets[0]}_model_{model.model_name}.json')

    with open(metrics_file, 'w') as json_file:
        json.dump(metrics, json_file)