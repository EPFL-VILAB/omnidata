import os
import argparse
import numpy as np
import random
import pickle
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
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

from data.taskonomy_replica_gso_dataset import TaskonomyReplicaGsoDataset, REPLICA_BUILDINGS
from models.unet import UNet
from models.multi_task_model import MultiTaskModel
from losses import masked_l1_loss, compute_grad_norm_losses

def building_in_gso(building):
    return building.__contains__('-') and building.split('-')[0] in REPLICA_BUILDINGS

def building_in_replica(building):
    return building in REPLICA_BUILDINGS

def building_in_hypersim(building):
    return building.startswith('ai_')

def building_in_taskonomy(building):
    return building not in REPLICA_BUILDINGS and not building.startswith('ai_') and not building.__contains__('-')


class ConsistentDepth(pl.LightningModule):
    def __init__(self,
                 pretrained_weights_path,
                 num_positive,
                 image_size,
                 batch_size,
                 num_workers,
                 lr,
                 lr_step,
                 taskonomy_variant,
                 taskonomy_root,
                 replica_root,
                 gso_root,
                 hypersim_root,
                 use_taskonomy,
                 use_replica,
                 use_gso,
                 use_hypersim,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters(
            'num_positive', 'image_size', 'batch_size', 'num_workers', 'lr', 'lr_step',
            'taskonomy_variant', 'taskonomy_root', 'replica_root', 'gso_root', 'use_taskonomy', 'use_replica', 'use_gso',
            'pretrained_weights_path', 'experiment_name', 'restore', 'gpus', 'distributed_backend', 
            'precision', 'val_check_interval', 'max_epochs'
        )
        self.pretrained_weights_path = pretrained_weights_path
        self.num_positive = num_positive
        self.image_size = image_size
        self.batch_size = batch_size
        self.gpus = kwargs['gpus']

        self.num_workers = num_workers
        self.learning_rate = lr
        self.lr_step = lr_step

        self.taskonomy_variant = taskonomy_variant
        self.taskonomy_root = taskonomy_root
        self.replica_root = replica_root
        self.gso_root = gso_root
        self.hypersim_root = hypersim_root
        self.use_taskonomy = use_taskonomy
        self.use_replica = use_replica
        self.use_gso = use_gso
        self.use_hypersim = use_hypersim
        self.save_debug_info_on_error = False

        self.setup_datasets()
        
        self.val_samples = self.select_val_samples_for_datasets()

        # self.model = UNet(in_channels=3, out_channels=1)
        self.model = MultiTaskModel(tasks=['depth_zbuffer'], backbone='hrnet_w48', head='hrnet', pretrained=True, dilated=False)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--pretrained_weights_path', type=str, default=None,
            help='Path to pretrained UNet weights. Set to None for random init. (default: None)')
        parser.add_argument(
            '--num_positive', type=int, default=10,
            help='Number of views to return for each point. (default: 10)')
        parser.add_argument(
            '--image_size', type=int, default=512,
            help='Input image size. (default: 512)')
        parser.add_argument(
            '--lr', type=float, default=1e-3,
            help='Learning rate. (default: 1e-5)')
        parser.add_argument(
            '--lr_step', type=int, default=40,
            help='Number of epochs after which to decrease learning rate. (default: 1)')
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
            '--use_taskonomy', action='store_true', default=True,
            help='Set to use taskonomy dataset.')
        parser.add_argument(
            '--use_replica', action='store_true', default=True,
            help='Set to use replica dataset.')
        parser.add_argument(
            '--use_gso', action='store_true', default=False,
            help='Set to user GSO dataset.')
        parser.add_argument(
            '--use_hypersim', action='store_true', default=True,
            help='Set to user hypersim dataset.')
        return parser

    def setup_datasets(self):
        self.num_positive = 1 

        tasks = ['rgb', 'normal', 'segment_semantic', 'depth_zbuffer', 'mask_valid']

        self.train_datasets = []
        if self.use_taskonomy: self.train_datasets.append('taskonomy')
        if self.use_replica: self.train_datasets.append('replica')
        if self.use_gso: self.train_datasets.append('gso')
        if self.use_hypersim: self.train_datasets.append('hypersim')

        self.val_datasets = ['taskonomy', 'replica', 'hypersim']

        opt_train = TaskonomyReplicaGsoDataset.Options(
            taskonomy_data_path=self.taskonomy_root,
            replica_data_path=self.replica_root,
            gso_data_path=self.gso_root,
            hypersim_data_path=self.hypersim_root,
            tasks=tasks,
            datasets=self.train_datasets,
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            num_positive=self.num_positive,
            normalize_rgb=True,
            randomize_views=True
        )
        self.trainset = TaskonomyReplicaGsoDataset(options=opt_train)

        opt_val = TaskonomyReplicaGsoDataset.Options(
            taskonomy_data_path=self.taskonomy_root,
            replica_data_path=self.replica_root,
            gso_data_path=self.gso_root,
            hypersim_data_path=self.hypersim_root,
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=self.val_datasets,
            transform='DEFAULT',
            image_size=self.image_size,
            num_positive=self.num_positive,
            normalize_rgb=True,
            randomize_views=False
        )
        self.valset = TaskonomyReplicaGsoDataset(options=opt_val)
        
        # Shuffle, so that truncated validation sets are randomly sampled, but the same throughout training
        self.valset.randomize_order(seed=99)

        print('Loaded training and validation sets:')
        print(f'Train set contains {len(self.trainset)} samples.')
        print(f'Validation set contains {len(self.valset)} samples.')

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False
        )

    def forward(self, x):
        return self.model(x)['depth_zbuffer']

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, train=True)
        # Logging
        self.log('train_loss_supervised', res['loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        return {'loss': res['loss']}

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, train=False)
        # Logging
        self.log('val_loss_supervised', res['loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        return {'loss': res['loss']}
    
    
    def register_save_on_error_callback(self, callback):
        '''
            On error, will call the following callback. 
            Callback should have signature:
                callback(batch) -> none
        '''
        self.on_error_callback = callback
        self.save_debug_info_on_error = True
        
    def shared_step(self, batch, train=True):
        try:
            return self._shared_step(batch, train)
        except:
            if self.save_debug_info_on_error:
                self.on_error_callback(batch)
            raise

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
    
    def _shared_step(self, batch, train=True):
        step_results = {}

        rgb = batch['positive']['rgb']
        depth_gt = batch['positive']['depth_zbuffer']

        # Forward pass
        depth_preds = self(rgb)

        # Mask out invalid pixels and compute loss
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid'])
        loss = masked_l1_loss(depth_preds, depth_gt, mask_valid)
        
        step_results.update({
            'loss': loss
        })
        return step_results

    def validation_epoch_end(self, outputs):
        if trainer.global_rank > 0:
             return 
        # Log validation set and OOD debug images using W&B
        self.log_validation_example_images(num_images=30)
        self.log_ood_example_images(num_images=10)


    def select_val_samples_for_datasets(self):
        frls = 0
        val_imgs = defaultdict(list)
        while len(val_imgs['replica']) + len(val_imgs['taskonomy']) + \
             len(val_imgs['hypersim']) + len(val_imgs['gso']) < 95:
            idx = random.randint(0, len(self.valset) - 1)
            example = self.valset[idx]
            building = example['positive']['building']
            print(len(val_imgs['replica']), len(val_imgs['taskonomy']), len(val_imgs['hypersim']), len(val_imgs['gso']), building)
            
            if building_in_hypersim(building) and len(val_imgs['hypersim']) < 40:
                val_imgs['hypersim'].append(idx)

            elif building_in_replica(building) and len(val_imgs['replica']) < 25:
                if building.startswith('frl') and frls > 15:
                    continue
                if building.startswith('frl'): frls += 1
                val_imgs['replica'].append(idx)

            elif building_in_gso(building) and len(val_imgs['gso']) < 20:
                val_imgs['gso'].append(idx)

            elif building_in_taskonomy(building) and len(val_imgs['taskonomy']) < 30:
                val_imgs['taskonomy'].append(idx)
        return val_imgs

    def select_val_samples_for_datasets2(self):
        val_imgs = defaultdict(list)
        while len(val_imgs['taskonomy']) < 25:
            idx = random.randint(0, len(self.valset))
            val_imgs['taskonomy'].append(idx)
        return val_imgs

    def log_validation_example_images(self, num_images=20):
        self.model.eval()
        all_imgs = defaultdict(list)

        for dataset in self.val_datasets:
            for img_idx in self.val_samples[dataset]:
                example = self.valset[img_idx]
                num_positive = self.num_positive
                rgb_pos = example['positive']['rgb'].to(self.device)
                depth_gt_pos = example['positive']['depth_zbuffer']

                mask_valid = self.make_valid_mask(example['positive']['mask_valid']).squeeze(axis=0)

                depth_gt_pos[~mask_valid] = 0

                rgb_pos = rgb_pos.unsqueeze(axis=0)
                depth_gt_pos = depth_gt_pos.unsqueeze(axis=0)

                with torch.no_grad():
                    depth_preds_pos = self.model.forward(rgb_pos)['depth_zbuffer']

                for pos_idx in range(num_positive):
                    rgb = rgb_pos[pos_idx].permute(1, 2, 0).detach().cpu().numpy()
                    rgb = wandb.Image(rgb, caption=f'RGB I{img_idx}')
                    all_imgs[f'rgb-{dataset}'].append(rgb)

                    depth_gt = depth_gt_pos[pos_idx].permute(1, 2, 0).detach().cpu().numpy()
                    depth_gt = wandb.Image(depth_gt, caption=f'GT-Depth I{img_idx}')
                    all_imgs[f'gt-depth-{dataset}'].append(depth_gt)

                    depth_pred = depth_preds_pos[pos_idx].permute(1, 2, 0).detach().cpu().numpy()
                    depth_pred = wandb.Image(depth_pred, caption=f'Pred-Depth I{img_idx}')
                    all_imgs[f'pred-depth-{dataset}'].append(depth_pred)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    def log_ood_example_images(self, data_dir='/datasets/evaluation_ood/real_world/images', num_images=15):
        self.model.eval()

        all_imgs = {'rgb_ood': [], 'pred_ood': []}

        for img_idx in range(num_images):
            rgb = Image.open(f'{data_dir}/{img_idx:05d}.png').convert('RGB')
            rgb = self.valset.transform['rgb'](rgb).to(self.device)

            with torch.no_grad():
                depth_pred = self.model.forward(rgb.unsqueeze(0))['depth_zbuffer'][0]

            rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb, caption=f'RGB OOD {img_idx}')
            all_imgs['rgb_ood'].append(rgb)

            depth_pred = wandb.Image(depth_pred, caption=f'Pred OOD {img_idx}')
            all_imgs['pred_ood'].append(depth_pred)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=2e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40)
        return [optimizer], [scheduler]


def save_model_and_batch_on_error(checkpoint_function, save_path_prefix='.'):
    def _save(batch):
        checkpoint_function(os.path.join(save_path_prefix, "crash_model.pth"))
        print(f"Saving crash information to {save_path_prefix}")
        with open(os.path.join(save_path_prefix, "crash_batch.pth"), 'wb') as f:
            torch.save(batch, f)
        
    return _save



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
        '--save-dir', type=str, default='exps_depth',
        help='Directory in which to save this experiments. (default: exps/)')    


    # Add PyTorch Lightning Module and Trainer args
    parser = ConsistentDepth.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = ConsistentDepth(**vars(args))

    # model = ConsistentDepth.load_from_checkpoint(
    # checkpoint_path=args.pretrained_weights_path, **vars(args))

    if args.experiment_name is None:
        args.experiment_name = 'taskonomy_depth_baseline'

    os.makedirs(os.path.join(args.save_dir, 'wandb'), exist_ok=True)
    wandb_logger = WandbLogger(name=args.experiment_name,
                               project='omnidata', 
                               entity='ainaz',
                               save_dir=args.save_dir,
                               version=args.restore)
    wandb_logger.watch(model, log=None, log_freq=5000)

    # Save best and last model like {args.save_dir}/checkpoints/taskonomy_depth/W&BID/epoch-X.ckpt (or .../last.ckpt)
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints', f'{wandb_logger.name}', f'{wandb_logger.experiment.id}')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, '{epoch}'),
        verbose=True, monitor='val_loss_supervised', mode='min', period=1, save_last=True, save_top_k=3
    )

    if args.restore is None:
        trainer = Trainer.from_argparse_args(args, logger=wandb_logger, \
            checkpoint_callback=checkpoint_callback, gpus=-1, auto_lr_find=False, \
                accelerator='ddp')
    else:
        trainer = Trainer(
            resume_from_checkpoint=os.path.join(
                os.path.join(checkpoint_dir, 'last.ckpt')
            ),
            logger=wandb_logger, checkpoint_callback=checkpoint_callback, accelerator='ddp'
        )
    

    if args.save_on_error:
        model.register_save_on_error_callback(
            save_model_and_batch_on_error(
                trainer.save_checkpoint,
                args.save_dir
            )
        )

    # trainer.tune(model)
    print("!!! Learning rate :", model.learning_rate)
    trainer.fit(model)
