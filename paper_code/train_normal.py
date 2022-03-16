import os
import argparse
import numpy as np
from PIL import Image
import random
import pickle
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
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
from data.refocus_augmentation import RefocusImageAugmentation
from data.augmentation import Augmentation
from data.validate_oasis import OASISValidate
from models.unet import UNet
# from models.seg_hrnet import get_configured_hrnet
from models.multi_task_model import MultiTaskModel
from losses import masked_l1_loss, compute_grad_norm_losses, masked_cosine_angular_loss, masked_angular_distance

def building_in_gso(building):
    return building.__contains__('-') and building.split('-')[0] in REPLICA_BUILDINGS

def building_in_replica(building):
    return building in REPLICA_BUILDINGS

def building_in_hypersim(building):
    return building.startswith('ai_')

def building_in_taskonomy(building):
    return building not in REPLICA_BUILDINGS and not building.startswith('ai_') and not building.__contains__('-')

def building_in_blendedMVS(building):
    return building.startswith('5')

class ConsistentNormal(pl.LightningModule):
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
                 blendedMVS_root,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters(
            'num_positive', 'image_size', 'batch_size', 'num_workers', 'lr', 'lr_step',
            'taskonomy_variant', 'taskonomy_root', 'replica_root', 'gso_root', 'hypersim_root', 'blendedMVS_root',
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
        self.blendedMVS_root = blendedMVS_root
        self.save_debug_info_on_error = False

        self.normalize_rgb = False
        self.setup_datasets()
        self.val_samples = self.select_val_samples_for_datasets()
        self.log_val_imgs_step = 0
        self.oasis_validate = OASISValidate()

        # self.refocus_aug = RefocusImageAugmentation(8, 0.01, 3.0, return_segments=False)
        # self.refocus_aug = RefocusImageAugmentation(10, 0.005, 6.0, return_segments=False)
        self.aug = Augmentation()

        self.model = UNet(in_channels=3, out_channels=3)
        # self.model = MultiTaskModel(tasks=['normal'], n_channels=3,\
        #     backbone='hrnet_w18', head='hrnet', pretrained=True, dilated=False)

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
            '--lr_step', type=int, default=8,
            help='Number of epochs after which to decrease learning rate. (default: 1)')
        parser.add_argument(
            '--batch_size', type=int, default=4,
            help='Batch size for data loader (default: 4)')
        parser.add_argument(
            '--num_workers', type=int, default=16,
            help='Number of workers for DataLoader. (default: 16)')
        parser.add_argument(
            '--taskonomy_variant', type=str, default='fullplus',
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
            '--blendedMVS_root', type=str, default='/scratch/ainaz/BlendedMVS/mvs_low_res_taskonomized',
            help='Root directory of blendedMVS dataset.')


        return parser

    def setup_datasets(self):
        self.num_positive = 1 

        tasks = ['rgb', 'normal', 'mask_valid']        

        self.tasks = tasks

        self.val_datasets = ['taskonomy', 'replica', 'hypersim', 'gso', 'blendedMVS']

        opt_train_taskonomy = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=['taskonomy'],
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=True
        )
        
        self.trainset_taskonomy = TaskonomyReplicaGsoDataset(options=opt_train_taskonomy)

        opt_train_replica = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=['replica'],
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=True
        )
        
        self.trainset_replica = TaskonomyReplicaGsoDataset(options=opt_train_replica)

        opt_train_hypersim = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=['hypersim'],
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=True
        )
        
        self.trainset_hypersim = TaskonomyReplicaGsoDataset(options=opt_train_hypersim)

        opt_train_gso = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=['gso'],
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=True
        )
        
        self.trainset_gso = TaskonomyReplicaGsoDataset(options=opt_train_gso)

        opt_train_blendedMVS = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=['blendedMVS'],
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=True
        )
        
        self.trainset_blendedMVS = TaskonomyReplicaGsoDataset(options=opt_train_blendedMVS)


        opt_val_taskonomy = TaskonomyReplicaGsoDataset.Options(
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=['taskonomy'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=False
        )

        self.valset_taskonomy = TaskonomyReplicaGsoDataset(options=opt_val_taskonomy)
        self.valset_taskonomy.randomize_order(seed=99)

        opt_val_replica = TaskonomyReplicaGsoDataset.Options(
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=['replica'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=False
        )

        self.valset_replica = TaskonomyReplicaGsoDataset(options=opt_val_replica)
        self.valset_replica.randomize_order(seed=99)

        opt_val_hypersim = TaskonomyReplicaGsoDataset.Options(
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=['hypersim'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=False
        )

        self.valset_hypersim = TaskonomyReplicaGsoDataset(options=opt_val_hypersim)
        self.valset_hypersim.randomize_order(seed=99)

        opt_val_gso = TaskonomyReplicaGsoDataset.Options(
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=['gso'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=False
        )

        self.valset_gso = TaskonomyReplicaGsoDataset(options=opt_val_gso)
        self.valset_gso.randomize_order(seed=99)

        opt_val_blendedMVS = TaskonomyReplicaGsoDataset.Options(
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=['blendedMVS'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=False
        )

        self.valset_blendedMVS = TaskonomyReplicaGsoDataset(options=opt_val_blendedMVS)
        self.valset_blendedMVS.randomize_order(seed=99)

        print('Loaded training and validation sets:')
        print(f'Train set (taskonomy) contains {len(self.trainset_taskonomy)} samples.')
        print(f'Train set (replica) contains {len(self.trainset_replica)} samples.')
        print(f'Train set (hypersim) contains {len(self.trainset_hypersim)} samples.')
        print(f'Train set (gso) contains {len(self.trainset_gso)} samples.')
        print(f'Train set (blendedMVS) contains {len(self.trainset_blendedMVS)} samples.')

        print(f'Validation set (taskonomy) contains {len(self.valset_taskonomy)} samples.')
        print(f'Validation set (replica) contains {len(self.valset_replica)} samples.')
        print(f'Validation set (hypersim) contains {len(self.valset_hypersim)} samples.')
        print(f'Validation set (gso) contains {len(self.valset_gso)} samples.')
        print(f'Validation set (blendedMVS) contains {len(self.valset_blendedMVS)} samples.')

        self.train_val_sets = [
            self.trainset_taskonomy, self.trainset_hypersim, self.trainset_replica, 
            self.trainset_gso, self.trainset_blendedMVS,
            self.valset_taskonomy, self.valset_hypersim, self.valset_replica, 
            self.valset_gso, self.valset_blendedMVS]


    # def train_dataloader(self):
    #     return DataLoader(
    #         self.trainset_taskonomy, batch_size=self.batch_size, shuffle=True,
    #         num_workers=self.num_workers, pin_memory=False,
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.valset, batch_size=self.batch_size, shuffle=False,
    #         num_workers=self.num_workers, pin_memory=False
    #     )


    def train_dataloader(self):
        taskonomy_count = len(self.trainset_taskonomy)
        replica_count = len(self.trainset_replica)
        hypersim_count = len(self.trainset_hypersim)
        gso_count = len(self.trainset_gso)
        blendedMVS_count = len(self.trainset_blendedMVS)

        dataset_sample_count = torch.tensor([taskonomy_count, replica_count, hypersim_count, gso_count, blendedMVS_count])
        weight = 1. / dataset_sample_count.float()
        print("!!!!!!!!!!! ", weight, dataset_sample_count)
        samples_weight = torch.tensor(
            [weight[0]] * taskonomy_count + [weight[1]] * replica_count + [weight[2]] * hypersim_count + [weight[3]] * gso_count + [weight[4] * 3] * blendedMVS_count)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        trainset = ConcatDataset(
            [self.trainset_taskonomy, self.trainset_replica, self.trainset_hypersim, self.trainset_gso, self.trainset_blendedMVS])
        return DataLoader(
            trainset, batch_size=self.batch_size, sampler=sampler, 
            num_workers=self.num_workers, pin_memory=False
        )
        
    def val_dataloader(self):
        taskonomy_dl = DataLoader(
            self.valset_taskonomy, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=False
        )
        replica_dl = DataLoader(
            self.valset_replica, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=False
        )
        hypersim_dl = DataLoader(
            self.valset_hypersim, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=False
        )
        gso_dl = DataLoader(
            self.valset_gso, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=False
        )
        blendedMVS_dl = DataLoader(
            self.valset_blendedMVS, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=False
        )
        return [taskonomy_dl, replica_dl, hypersim_dl, gso_dl, blendedMVS_dl]

    def forward(self, x):
        return self.model(x) #['normal']

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, train=True)
        # Logging
        self.log('train_l1_loss', res['l1_loss'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
        self.log('train_cos_loss', res['cos_loss'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
        self.log('train_normal_loss', res['normal_loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        return {'loss': res['normal_loss']}

    def validation_step_combined(self, batch, batch_idx):
        res = self.shared_step(batch, train=False)
        # Logging
        self.log('val_normal_loss', res['loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        return {'loss': res['loss']}
    
    def validation_step(self, batch, batch_idx, dataset_idx):
        res = self.shared_step(batch, train=False)
        dataset = self.val_datasets[dataset_idx]
        res['dataset'] = dataset
        return res
    
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
        h, w = mask_float.shape[2], mask_float.shape[3]
        reshape_temp = len(mask_float.shape) == 5
        if reshape_temp:
            mask_float = rearrange(mask_float, 'b p c h w -> (b p) c h w')
        mask_float = 1 - mask_float
        mask_float = F.max_pool2d(mask_float, kernel_size=max_pool_size)
        # mask_float = F.interpolate(mask_float, (self.image_size, self.image_size), mode='nearest')
        mask_float = F.interpolate(mask_float, (h, w), mode='nearest')
        mask_valid = mask_float == 0
        if reshape_temp:
            mask_valid = rearrange(mask_valid, '(b p) c h w -> b p c h w', p=self.num_positive)

        return mask_valid

    
    def _shared_step(self, batch, train=True):
        # resize augmentation
        batch['positive'] = self.aug.resize_augmentation(batch['positive'], self.tasks)
        # rgb augmentation
        if train:
            augmented_rgb = self.aug.augment_rgb(batch)
        else:
            augmented_rgb = batch['positive']['rgb']

        step_results = {}
        normal_gt = batch['positive']['normal']


        # Forward pass
        normal_preds = self(augmented_rgb)
        # clamp the output
        normal_preds = torch.clamp(normal_preds, 0, 1)

        # Mask out invalid pixels and compute loss
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid']).repeat_interleave(3,1)

        l1_loss = masked_l1_loss(normal_preds, normal_gt, mask_valid)
        cos_loss = masked_cosine_angular_loss(normal_preds, normal_gt, mask_valid)
        loss = cos_loss + 10 * l1_loss

        step_results.update({
            'l1_loss': l1_loss,
            'cos_loss': cos_loss,
            'normal_loss': loss
        })
        return step_results


    def validation_epoch_end(self, outputs):
        counts = {'taskonomy':0, 'replica':0, 'hypersim':0, 'gso':0, 'blendedMVS':0, 'all':0}
        losses = {}
        losses = defaultdict(lambda: 0, losses)
        for dataloader_outputs in outputs:
            for output in dataloader_outputs:
                dataset = output['dataset']
                counts[dataset] += 1
                counts['all'] += 1
                for loss_name in output:
                    if loss_name.__contains__('_loss'):
                        losses[f'{dataset}_{loss_name}'] += output[loss_name]
                        losses[loss_name] += output[loss_name]

        for loss_name in losses:
            if loss_name.split('_')[0] in self.val_datasets:
                losses[loss_name] /= counts[loss_name.split('_')[0]]
            else:
                losses[loss_name] /= counts['all']

            if not loss_name.__contains__('normal') or loss_name == 'normal_loss':
                self.log(f'val_{loss_name}', losses[loss_name], prog_bar=False, logger=True, sync_dist=self.gpus>1)

        # OASIS Validation
        # oasis_results = self.oasis_validate.valid_augmented(self.model, self.device, max_iter=500)
        # self.log(f'oasis_mean_angle_error', oasis_results['MAE'], prog_bar=False, logger=True, sync_dist=self.gpus>1)

        # Log validation set and OOD debug images using W&B
        if self.global_step >= self.log_val_imgs_step + 15999 or self.global_step <= 3000:
            self.log_val_imgs_step = self.global_step
            self.log_validation_example_images(num_images=10)
            self.log_ood_example_images(num_images=10)

    def select_val_samples_for_datasets(self):
        frls = 0
        val_imgs = defaultdict(list)

        while len(val_imgs['hypersim']) < 20:
            idx = random.randint(0, len(self.valset_hypersim) - 1)
            val_imgs['hypersim'].append(idx)
        while len(val_imgs['replica']) < 20:
            idx = random.randint(0, len(self.valset_replica) - 1)
            example = self.valset_replica[idx]
            building = example['positive']['building']
            if building.startswith('frl') and frls > 15:
                continue
            if building.startswith('frl'): frls += 1
            val_imgs['replica'].append(idx)
        while len(val_imgs['taskonomy']) < 20:
            idx = random.randint(0, len(self.valset_taskonomy) - 1)
            val_imgs['taskonomy'].append(idx)
        while len(val_imgs['gso']) < 20:
            idx = random.randint(0, len(self.valset_gso) - 1)
            val_imgs['gso'].append(idx)
        while len(val_imgs['blendedMVS']) < 50:
            idx = random.randint(0, len(self.valset_blendedMVS) - 1)
            val_imgs['blendedMVS'].append(idx)

        return val_imgs
    

    def log_validation_example_images(self, num_images=20):
        self.model.eval()
        all_imgs = defaultdict(list)

        for dataset in self.val_datasets:
            for img_idx in self.val_samples[dataset]:
                if dataset == 'taskonomy': example = self.valset_taskonomy[img_idx]
                elif dataset == 'replica': example = self.valset_replica[img_idx]
                elif dataset == 'hypersim': example = self.valset_hypersim[img_idx]
                elif dataset == 'gso': example = self.valset_gso[img_idx]
                elif dataset == 'blendedMVS': example = self.valset_blendedMVS[img_idx]

                num_positive = self.num_positive
                rgb_pos = example['positive']['rgb'].to(self.device)

                # augmentation
                example['positive'] = self.aug.resize_augmentation(example['positive'], self.tasks)
                rgb_pos = self.aug.augment_rgb(example)[0].to(self.device)

                normal_gt_pos = example['positive']['normal'][0]

                mask_valid = self.make_valid_mask(example['positive']['mask_valid'][0]).squeeze(axis=0).repeat_interleave(3,0)

                normal_gt_pos[~mask_valid] = 0

                rgb_pos = rgb_pos.unsqueeze(axis=0)
                normal_gt_pos = normal_gt_pos.unsqueeze(axis=0)

                with torch.no_grad():
                    normal_preds_pos = self.model.forward(rgb_pos) #['normal']
                    normal_preds_pos = torch.clamp(normal_preds_pos, 0, 1)

                for pos_idx in range(num_positive):
                    rgb = rgb_pos[pos_idx].permute(1, 2, 0).detach().cpu().numpy()
                    rgb = wandb.Image(rgb, caption=f'RGB I{img_idx}')
                    all_imgs[f'rgb-{dataset}'].append(rgb)

                    normal_gt = normal_gt_pos[pos_idx].permute(1, 2, 0).detach().cpu().numpy()
                    normal_gt = wandb.Image(normal_gt, caption=f'GT-Normal I{img_idx}')
                    all_imgs[f'gt-normal-{dataset}'].append(normal_gt)

                    normal_pred = normal_preds_pos[pos_idx].permute(1, 2, 0).detach().cpu().numpy()
                    normal_pred = wandb.Image(normal_pred, caption=f'Pred-Normal I{img_idx}')
                    all_imgs[f'pred-normal-{dataset}'].append(normal_pred)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    
    # /datasets/evaluation_ood/real_world/images
    def log_ood_example_images(self, data_dir='/scratch/ainaz/ood', num_images=15):
        self.model.eval()

        all_imgs = defaultdict(list)

        # for img_idx in range(num_images):
        for img_name in os.listdir(data_dir):
            # rgb = Image.open(f'{data_dir}/{img_idx:05d}.png').convert('RGB')
            rgb = Image.open(f'{data_dir}/{img_name}').convert('RGB')
            # rgb = self.valset_taskonomy.transform['rgb'](rgb).to(self.device)

            transform = transforms.Compose([
                transforms.Resize(384, Image.BILINEAR),
                transforms.CenterCrop(384),
                self.valset_taskonomy.transform['rgb']])

            rgb = transform(rgb).to(self.device)
            

            with torch.no_grad():
                normal_pred = self.model.forward(rgb.unsqueeze(0))[0] #['normal'][0]
                normal_pred = torch.clamp(normal_pred, 0, 1)

            rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb, caption=f'RGB OOD {img_name}')
            all_imgs['rgb_ood'].append(rgb)

            normal_pred = wandb.Image(normal_pred, caption=f'Pred-Normal OOD {img_name}')
            all_imgs['pred-normal-ood'].append(normal_pred)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=2e-6, amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40)     
        return optimizer


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
        '--save-dir', type=str, default='experiments/normal',
        help='Directory in which to save this experiment. (default: exps/)') 
 

    # Add PyTorch Lightning Module and Trainer args
    parser = ConsistentNormal.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = ConsistentNormal(**vars(args))

    if args.experiment_name is None:
        args.experiment_name = 'taskonomy_normal_baseline'

    os.makedirs(os.path.join(args.save_dir, 'wandb'), exist_ok=True)
    wandb_logger = WandbLogger(name=args.experiment_name,
                               project='omnidata', 
                               entity='ainaz',
                               save_dir=args.save_dir,
                               version=args.restore)
    wandb_logger.watch(model, log=None, log_freq=5000)
    wandb_logger.log_hyperparams(model.hparams)

    # Save best and last model like {args.save_dir}/checkpoints/taskonomy_normal/W&BID/epoch-X.ckpt (or .../last.ckpt)
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints', f'{wandb_logger.name}', f'{wandb_logger.experiment.id}')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, '{epoch}'),
        verbose=True, monitor='val_normal_loss', mode='min', period=1, save_last=True, save_top_k=5
    )

    if args.restore is None:
        trainer = Trainer.from_argparse_args(args, logger=wandb_logger, \
            checkpoint_callback=checkpoint_callback, gpus=[0, 1], auto_lr_find=False, gradient_clip_val=10,\
                accelerator='ddp', replace_sampler_ddp=False)
    else:
        checkpoint_path = os.path.join(f'/scratch/ainaz/omnidata2/experiments/normal/checkpoints/{wandb_logger.name}/{args.restore}/last.ckpt')
        trainer = Trainer(
            resume_from_checkpoint=checkpoint_path,
            logger=wandb_logger, checkpoint_callback=checkpoint_callback, accelerator='ddp', 
            gpus=[0, 1], gradient_clip_val=10, replace_sampler_ddp=False

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
