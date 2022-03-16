import os
import argparse
import numpy as np
from PIL import Image
import pickle
from einops import rearrange
import einops as ein
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
import wandb

from data.taskonomy_replica_gso_dataset import TaskonomyReplicaGsoDataset, REPLICA_BUILDINGS
from data.segment_instance import extract_instances, TASKONOMY_CLASS_LABELS, TASKONOMY_CLASS_COLORS, \
    REPLICA_CLASS_LABELS, REPLICA_CLASS_COLORS, HYPERSIM_CLASS_COLORS, NYU40_COLORS, \
        GSO_NUM_CLASSES, GSO_CLASS_COLORS, COMBINED_CLASS_LABELS, COMBINED_CLASS_COLORS, plot_instances, apply_mask
from losses import compute_grad_norm_losses
from losses.masked_losses import masked_l1_loss
from models.unet_semseg import UNetSemSeg, UNetSemSegCombined
from models.seg_hrnet import get_configured_hrnet
from models.multi_task_model import MultiTaskModel
from models.padnet import PADNet

RGB_MEAN = torch.Tensor([0.55312, 0.52514, 0.49313]).reshape(3,1,1)
RGB_STD =  torch.Tensor([0.20555, 0.21775, 0.24044]).reshape(3,1,1)

def building_in_gso(building):
    return building.__contains__('-') and building.split('-')[0] in REPLICA_BUILDINGS

def building_in_replica(building):
    return building in REPLICA_BUILDINGS

def building_in_hypersim(building):
    return building.startswith('ai_')

def building_in_taskonomy(building):
    return building not in REPLICA_BUILDINGS and not building.startswith('ai_') and not building.__contains__('-')

class MuliTask(pl.LightningModule):
    def __init__(self, 
                 pretrained_weights_path,
                 image_size, model_name, batch_size, num_workers, lr, lr_step, loss_balancing,
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
            'image_size', 'model_name', 'batch_size', 'num_workers', 'lr', 'lr_step', 'loss_balancing',
            'taskonomy_variant', 'taskonomy_root',
            'experiment_name', 'restore', 'gpus', 'distributed_backend', 'precision', 'val_check_interval', 'max_epochs',
        )
        self.pretrained_weights_path = pretrained_weights_path
        self.image_size = image_size
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = kwargs['gpus']
        self.num_workers = num_workers
        self.lr = lr
        self.lr_step = lr_step
        self.loss_balancing = loss_balancing
        self.taskonomy_variant = taskonomy_variant
        self.taskonomy_root = taskonomy_root
        self.replica_root = replica_root
        self.gso_root = gso_root
        self.hypersim_root = hypersim_root
        self.use_taskonomy = use_taskonomy
        self.use_replica = use_replica
        self.use_gso = use_gso
        self.use_hypersim = use_hypersim
        
        self.normalize_rgb = True
        self.setup_datasets()
        self.val_samples = self.select_val_samples_for_datasets()
        self.log_val_imgs_step = 0

        # self.model = MultiTaskModel(tasks=['normal', 'segment_semantic', 'depth_zbuffer'], n_channels=3, \
        #     backbone='hrnet_w18', head='hrnet', pretrained=True, dilated=False)

        # PAD-Net
        self.auxiliary_tasks = ['normal', 'segment_semantic']
        # self.auxiliary_tasks = ['normal', 'segment_semantic', 'edge_occlusion']
        # self.auxiliary_tasks = ['normal', 'segment_semantic', 'edge_occlusion', 'edge_texture', 'depth_zbuffer', 'keypoints3d']

        self.tasks = ['segment_semantic']
        self.model = PADNet(self.tasks, self.auxiliary_tasks, backbone='hrnet_w18', pretrained=False)
        

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
            '--image_size', type=int, default=256,
            help='Input image size. (default: 256)')
        parser.add_argument(
            '--lr', type=float, default=0.0001,
            help='Learning rate. (default: 0.00001)')
        parser.add_argument(
            '--lr_step', type=int, default=10,
            help='Number of epochs after which to decrease learning rate. (default: 1)')
        parser.add_argument(
            '--loss_balancing', type=str, default='none',
            choices=['none', 'grad_norm'],
            help='Loss balancing choice. One of [none, grad_norm]. (default: none)')
        parser.add_argument(
            '--batch_size', type=int, default=16,
            help='Batch size for data loader (default: 16)')
        parser.add_argument(
            '--num_workers', type=int, default=16,
            help='Number of workers for DataLoader. (default: 16)')
        parser.add_argument(
            '--taskonomy_variant', type=str, default='tiny',
            choices=['full', 'fullplus', 'medium', 'tiny'],
            help='One of [full, fullplus, medium, tiny] (default: fullplus)')
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
            '--use_taskonomy', action='store_true', default=False,
            help='Set to use taskonomy dataset.')
        parser.add_argument(
            '--use_replica', action='store_true', default=True,
            help='Set to use replica dataset.')
        parser.add_argument(
            '--use_gso', action='store_true', default=False,
            help='Set to user GSO dataset.')
        parser.add_argument(
            '--use_hypersim', action='store_true', default=False,
            help='Set to user hypersim dataset.')
        parser.add_argument(
            '--model_name', type=str, default='mask_rnn',
            help='Semantic segmentation network. (default: mask_rnn)')
        return parser
        
    def setup_datasets(self):
        self.num_positive = 1
        self.train_datasets = []
        if self.use_taskonomy: self.train_datasets.append('taskonomy')
        if self.use_replica: self.train_datasets.append('replica')
        if self.use_gso: self.train_datasets.append('gso')
        if self.use_hypersim: self.train_datasets.append('hypersim')

        self.val_datasets = ['taskonomy', 'replica', 'hypersim']
        # tasks = ['rgb', 'normal', 'segment_semantic', 'edge_occlusion', 'edge_texture', 'depth_zbuffer', 'keypoints3d', 'mask_valid']
        tasks = ['rgb', 'normal', 'segment_semantic', 'mask_valid']

        # opt_train_taskonomy = TaskonomyReplicaGsoDataset.Options(
        #     tasks=tasks,
        #     datasets=['taskonomy'],
        #     split='train',
        #     taskonomy_variant=self.taskonomy_variant,
        #     transform='DEFAULT',
        #     image_size=self.image_size,
        #     normalize_rgb=True,
        #     randomize_views=True
        # )
        
        # self.trainset_taskonomy = TaskonomyReplicaGsoDataset(options=opt_train_taskonomy)

        opt_train_replica = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=['replica'],
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=True,
            randomize_views=True
        )
        
        self.trainset_replica = TaskonomyReplicaGsoDataset(options=opt_train_replica)

        # opt_train_hypersim = TaskonomyReplicaGsoDataset.Options(
        #     tasks=tasks,
        #     datasets=['hypersim'],
        #     split='train',
        #     taskonomy_variant=self.taskonomy_variant,
        #     transform='DEFAULT',
        #     image_size=self.image_size,
        #     normalize_rgb=True,
        #     randomize_views=True
        # )
        
        # self.trainset_hypersim = TaskonomyReplicaGsoDataset(options=opt_train_hypersim)


        opt_val_taskonomy = TaskonomyReplicaGsoDataset.Options(
            split='val',
            taskonomy_variant=self.taskonomy_variant,
            tasks=tasks,
            datasets=['taskonomy'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=True,
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
            normalize_rgb=True,
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
            normalize_rgb=True,
            randomize_views=False
        )

        self.valset_hypersim = TaskonomyReplicaGsoDataset(options=opt_val_hypersim)
        self.valset_hypersim.randomize_order(seed=99)

        print('Loaded training and validation sets:')
        # print(f'Train set contains {len(self.trainset)} samples.')
        # print(f'Validation set (combined) contains {len(self.valset_combined)} samples.')
        # print(f'Train set (taskonomy) contains {len(self.trainset_taskonomy)} samples.')
        print(f'Train set (replica) contains {len(self.trainset_replica)} samples.')
        # print(f'Train set (hypersim) contains {len(self.trainset_hypersim)} samples.')
        print(f'Validation set (taskonomy) contains {len(self.valset_taskonomy)} samples.')
        print(f'Validation set (replica) contains {len(self.valset_replica)} samples.')
        print(f'Validation set (hypersim) contains {len(self.valset_hypersim)} samples.')


    
    def train_dataloader(self):
        return DataLoader(
            self.trainset_replica, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=False
        )

    # def train_dataloader(self):
    #     taskonomy_count = len(self.trainset_taskonomy)
    #     replica_count = len(self.trainset_replica)
    #     hypersim_count = len(self.trainset_hypersim)
    #     dataset_sample_count = torch.tensor([taskonomy_count, replica_count, hypersim_count])
    #     weight = 1. / dataset_sample_count.float()
    #     print("!!!!!!!!!!! ", weight, dataset_sample_count)
    #     samples_weight = torch.tensor([weight[0]] * taskonomy_count + [weight[1]] * replica_count + [weight[2]] * hypersim_count)
    #     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    #     trainset = ConcatDataset([self.trainset_taskonomy, self.trainset_replica, self.trainset_hypersim])
    #     return DataLoader(
    #         trainset, batch_size=self.batch_size, sampler=sampler, 
    #         num_workers=self.num_workers, pin_memory=False
    #     )
        
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
        return [taskonomy_dl, replica_dl, hypersim_dl]

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, train=True)
        # Logging
        self.log('train_loss', res['loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        # , 'edge2d_initial', 'keypoints3d_initial', 'depth_initial'
        for task in ['semantic', 'normal_initial', 'semantic_initial']:
            self.log(f'train_{task}_loss', res[f'{task}_loss'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
            self.log(f'{task}_loss_weight', res[f'{task}_loss_weight'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
        
        return {'loss': res['loss']}
    
    def validation_step_combined(self, batch, batch_idx):
        res = self.shared_step(batch, train=False)
        # Logging
        self.log('val_loss', res['loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1, on_epoch=True)
        for task in ['semantic', 'normal', 'depth']:
            self.log(f'val_{task}_loss', res[f'{task}_loss'], prog_bar=False, logger=True, sync_dist=self.gpus>1, on_epoch=True)
        return {'loss': res['loss']}

    def validation_step(self, batch, batch_idx, dataset_idx):
        res = self.shared_step(batch, train=False)
        dataset = self.val_datasets[dataset_idx]
        res['dataset'] = dataset
        return res


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
        

    def shared_step(self, batch, train=True):
        step_results = {}
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid'])
        mask_valid_semantic = mask_valid.squeeze(1)
        mask_valid_normal = mask_valid.repeat_interleave(3,1)
        # mask_valid_edge3d = mask_valid.clone()
        # mask_valid_edge2d = mask_valid.clone()
        # mask_valid_keypoints3d = mask_valid.clone()
        # mask_valid_depth = mask_valid.clone()
        rgb = batch['positive']['rgb']
        semantic = batch['positive']['segment_semantic']
        normal_gt = batch['positive']['normal']
        # edge_occlusion_gt = batch['positive']['edge_occlusion']
        # edge_texture_gt = batch['positive']['edge_texture']
        # keypoints3d_gt = batch['positive']['keypoints3d']
        # depth_gt = batch['positive']['depth_zbuffer']
        semantic_gt = semantic[:,:,:,0]

        # background and undefined classes are labeled as 0
        semantic_gt[(semantic[:,:,:,0]==255) * (semantic[:,:,:,1]==255) * (semantic[:,:,:,2]==255)] = 0 # background in taskonomy
        semantic_gt[semantic_gt==-1] = 0  # undefined class in hypersim

        # mask out invalid parts of the mesh, background and undefined label
        semantic_gt *= mask_valid_semantic # invalid parts of the mesh also have undefined label (0)
        semantic_gt -= 1  # the model should not predict undefined and background classes

        # Compute loss weights
        if self.loss_balancing == 'grad_norm' and train and len(losses) > 1:
            loss_weights = compute_grad_norm_losses(losses, self.model.backbone.layer4)
        else:
            loss_weights = {'semantic': 1, 'normal_initial':10, 'semantic_initial':1}
            # loss_weights = {
            #     'semantic': 1, 
            #     'normal_initial':10,
            #     'depth_initial':10, 
            #     'edge3d_initial':100, 
            #     'edge2d_initial':10, 
            #     'keypoints3d_initial':10, 
            #     'semantic_initial':1}

        # Forward pass MultiTaskModel
        # preds = self(rgb)
        # normal_preds = preds['normal']
        # normal_preds = torch.clamp(normal_preds, -1, 1)
        # semantic_preds = preds['segment_semantic']
        # depth_preds = preds['depth_zbuffer']
        # depth_preds = torch.clamp(depth_preds, -1, 1)

        # loss_normal = masked_l1_loss(normal_preds, normal_gt, mask_valid_normal)
        # loss_semantic = criterion(semantic_preds, semantic_gt)
        # loss_depth = masked_l1_loss(depth_preds, depth_gt, mask_valid_depth)

        # losses = {'semantic':loss_semantic, 'normal':loss_normal, 'edge3d':loss_edge}
        # total_loss = sum([losses[loss_name] * loss_weights[loss_name] for loss_name in losses.keys()])


        # Forward pass PAD-Net
        preds = self(rgb)

        # Losses initial task predictions (deepsup)
        normal_preds_initial = F.interpolate(preds['initial_normal'], self.image_size, mode='bilinear')
        semantic_preds_initial = F.interpolate(preds['initial_segment_semantic'], self.image_size, mode='bilinear')
        # edge3d_preds_initial = F.interpolate(preds['initial_edge_occlusion'], self.image_size, mode='bilinear')
        # edge2d_preds_initial = F.interpolate(preds['initial_edge_texture'], self.image_size, mode='bilinear')
        # depth_preds_initial = F.interpolate(preds['initial_depth_zbuffer'], self.image_size, mode='bilinear')
        # keypoints3d_preds_initial = F.interpolate(preds['initial_keypoints3d'], self.image_size, mode='bilinear')

        loss_normal_initial = masked_l1_loss(normal_preds_initial, normal_gt, mask_valid_normal)
        loss_semantic_initial = criterion(semantic_preds_initial, semantic_gt)
        # loss_edge3d_initial = masked_l1_loss(edge3d_preds_initial, edge_occlusion_gt, mask_valid_edge3d)
        # loss_edge2d_initial = masked_l1_loss(edge2d_preds_initial, edge_texture_gt, mask_valid_edge2d)
        # loss_keypoints3d_initial = masked_l1_loss(keypoints3d_preds_initial, keypoints3d_gt, mask_valid_keypoints3d)
        # loss_depth_initial = masked_l1_loss(depth_preds_initial, depth_gt, mask_valid_depth)

        # Losses at output  
        semantic_preds = preds['segment_semantic']
        loss_semantic = criterion(semantic_preds, semantic_gt)

        losses = {
            'semantic':loss_semantic, 
            'normal_initial':loss_normal_initial, 
            # 'edge3d_initial':loss_edge3d_initial, 
            # 'edge2d_initial':loss_edge2d_initial, 
            # 'keypoints3d_initial':loss_keypoints3d_initial, 
            # 'depth_initial':loss_depth_initial, 
            'semantic_initial':loss_semantic_initial}
        total_loss = sum([losses[loss_name] * loss_weights[loss_name] for loss_name in losses.keys()])


        for loss_name, loss in losses.items(): 
            step_results.update({f'{loss_name}_loss': loss})   
            if train:
                step_results.update({f'{loss_name}_loss_weight': loss_weights[loss_name]})              
        
        step_results.update({
            'loss': total_loss
        })
        return step_results
 

    def validation_epoch_end(self, outputs):
        counts = {'taskonomy':0, 'replica':0, 'hypersim':0, 'all':0}
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

        # losses['loss'] = losses['semantic_loss'] + losses['normal_loss'] + losses['edge3d_loss']
        for loss_name in losses:
            if loss_name.split('_')[0] in self.val_datasets:
                losses[loss_name] /= counts[loss_name.split('_')[0]]
            else:
                losses[loss_name] /= counts['all']

            if loss_name.__contains__('semantic') and not loss_name.__contains__('initial'):
                self.log(f'val_{loss_name}', losses[loss_name], prog_bar=False, logger=True, sync_dist=self.gpus>1)

        # Log validation set and OOD debug images using W&B
        if self.global_step >= self.log_val_imgs_step + 19999 or self.global_step < 2000:
            self.log_val_imgs_step = self.global_step
            self.log_validation_example_images(num_images=10)
            self.log_ood_example_images(num_images=10)

    def select_val_samples_for_datasets(self):
        frls = 0
        val_imgs = defaultdict(list)
        # with open('/scratch/ainaz/omnidata2/val_samples/hypersim_val_indices.pkl', 'rb') as f:
        #     val_imgs['hypersim'] = pickle.load(f)

        while len(val_imgs['hypersim']) < 40:
            idx = random.randint(0, len(self.valset_hypersim) - 1)
            val_imgs['hypersim'].append(idx)

        while len(val_imgs['replica']) < 25:
            idx = random.randint(0, len(self.valset_replica) - 1)
            example = self.valset_replica[idx]
            building = example['positive']['building']
            if building.startswith('frl') and frls > 12:
                continue
            if building.startswith('frl'): frls += 1
            val_imgs['replica'].append(idx)
        while len(val_imgs['taskonomy']) < 20:
            idx = random.randint(0, len(self.valset_taskonomy) - 1)
            val_imgs['taskonomy'].append(idx)

        return val_imgs


    def log_validation_example_images(self, num_images=10):
        self.model.eval()
        all_imgs = defaultdict(list)
        for dataset in self.val_datasets:
            for img_idx in self.val_samples[dataset]:
                if dataset == 'taskonomy': example = self.valset_taskonomy[img_idx]
                elif dataset == 'replica': example = self.valset_replica[img_idx]
                elif dataset == 'hypersim': example = self.valset_hypersim[img_idx]
                rgb = example['positive']['rgb'].to(self.device)
                semantic = example['positive']['segment_semantic']
                # normal_gt = example['positive']['normal']
                # depth_gt = example['positive']['depth_zbuffer']
                mask_valid = self.make_valid_mask(example['positive']['mask_valid'])
                mask_valid_semantic = mask_valid.squeeze()
                # mask_valid_normal = mask_valid.squeeze(0).repeat_interleave(3,0)
                # mask_valid_depth = mask_valid.squeeze(0)
                # normal_gt[~mask_valid_normal] = 0
                # depth_gt[~mask_valid_depth] = 0

                if dataset == 'gso': semantic_gt = 2**8 * semantic[:,:,0] + semantic[:,:,1]
                else: semantic_gt = semantic[:,:,0]
                
                # background and undefined classes are labeled as 0
                semantic_gt[(semantic[:,:,0]==255) * (semantic[:,:,1]==255) * (semantic[:,:,2]==255)] = 0
                semantic_gt[semantic_gt==-1] = 0

                semantic_gt *= mask_valid_semantic.cpu()  # final labels
                mask_valid_semantic = semantic_gt != 0
                if mask_valid_semantic.sum() == 0: continue

                with torch.no_grad(): 
                    preds = self.model.forward(rgb.unsqueeze(0))
                    semantic_pred = preds['segment_semantic'].squeeze(0)
                    # normal_pred = preds['normal'].squeeze(0)
                    # normal_pred = torch.clamp(normal_pred, 0, 1)
                    # depth_pred = preds['depth_zbuffer'].squeeze(0)
                    CLASS_COLORS = COMBINED_CLASS_COLORS

                # semantic
                mask_preds = F.softmax(semantic_pred, dim=0)
                mask_preds = torch.argmax(mask_preds, dim=0) + 1 # model does not predict background/undefined
                mask_preds *= mask_valid_semantic.to(self.device)
                rgb_masked = (rgb.detach().clone().cpu() * RGB_STD + RGB_MEAN).permute(1,2,0).numpy()

                metadata = MetadataCatalog.get(dataset)
                metadata.stuff_classes = COMBINED_CLASS_LABELS
                metadata.things_classes = []
                metadata.stuff_colors = [[c * 255.0 for c in color] for color in CLASS_COLORS] 
                mask = mask_valid_semantic.unsqueeze(axis=2).repeat_interleave(3,2) * 1.0

                visualizer_gt = Visualizer(rgb_masked * 255.0, metadata=metadata)
                vis_gt = visualizer_gt.draw_sem_seg(semantic_gt.cpu(), area_threshold=None, alpha=0.9)
                gt_im = vis_gt.get_image()
                semantic_gt_im = mask * np.uint8(gt_im) + (1-mask) * np.uint8(rgb_masked * 255.0)

                visualizer_pred = Visualizer(rgb_masked * 255.0, metadata=metadata)
                vis_pred = visualizer_pred.draw_sem_seg(mask_preds.cpu(), area_threshold=None, alpha=0.6)
                pred_im = vis_pred.get_image()
                semantic_pred_im = mask * np.uint8(pred_im) + (1-mask) * np.uint8(rgb_masked * 255.0)
  
                rgb_img = wandb.Image(rgb.permute(1,2,0).detach().cpu().numpy(), caption=f'RGB {img_idx}')
                all_imgs[f'rgb-{dataset}'].append(rgb_img)
                semantic_gt = wandb.Image(np.uint8(semantic_gt_im), caption=f'GT-Semantic {img_idx}')
                all_imgs[f'gt-semantic-{dataset}'].append(semantic_gt)
                semantic_pred = wandb.Image(np.uint8(semantic_pred_im), caption=f'Pred-Semantic {img_idx}')
                all_imgs[f'pred-semantic-{dataset}'].append(semantic_pred)

                # normals
                # normal_gt = normal_gt.permute(1, 2, 0).detach().cpu().numpy()
                # normal_gt = wandb.Image(normal_gt, caption=f'GT-Normal I{img_idx}')
                # all_imgs[f'gt-normal-{dataset}'].append(normal_gt)
                # normal_pred = normal_pred.permute(1, 2, 0).detach().cpu().numpy()
                # normal_pred = wandb.Image(normal_pred, caption=f'Pred-Normal I{img_idx}')
                # all_imgs[f'pred-normal-{dataset}'].append(normal_pred)

                # depth
                # depth_gt = depth_gt.permute(1, 2, 0).detach().cpu().numpy()
                # depth_gt = wandb.Image(depth_gt, caption=f'GT-Depth I{img_idx}')
                # all_imgs[f'gt-depth-{dataset}'].append(depth_gt)
                # depth_pred = depth_pred.permute(1, 2, 0).detach().cpu().numpy()
                # depth_pred = wandb.Image(depth_pred, caption=f'Pred-Depth I{img_idx}')
                # all_imgs[f'pred-depth-{dataset}'].append(depth_pred)


        self.logger.experiment.log(all_imgs, step=self.global_step)

    def log_ood_example_images(self, data_dir='/datasets/evaluation_ood/real_world/images', num_images=15):
        self.model.eval()

        all_imgs = defaultdict(list)

        for img_idx in range(num_images):
            rgb = Image.open(f'{data_dir}/{img_idx:05d}.png').convert('RGB')
            rgb = self.trainset_replica.transform['rgb'](rgb).to(self.device)

            with torch.no_grad():
                preds = self.model.forward(rgb.unsqueeze(0))
                semantic_pred = preds['segment_semantic'].squeeze(0)
                # normal_pred = preds['normal'].squeeze(0)
                # normal_pred = torch.clamp(normal_pred, 0, 1)
                # depth_pred = preds['depth_zbuffer'].squeeze(0)
                CLASS_COLORS = COMBINED_CLASS_COLORS

            rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb_np, caption=f'RGB OOD {img_idx}')
            all_imgs['rgb-ood'].append(rgb)

            # normal_pred = wandb.Image(normal_pred, caption=f'Pred-Normal OOD {img_idx}')
            # all_imgs['pred-normal-ood'].append(normal_pred)

            # depth_pred = wandb.Image(depth_pred, caption=f'Pred-Depth OOD {img_idx}')
            # all_imgs['pred-depth-ood'].append(depth_pred)

            # semantics
            mask_preds = F.softmax(semantic_pred, dim=0)
            mask_preds = torch.argmax(mask_preds, dim=0) + 1 # model does not predict background/undefined
            metadata = MetadataCatalog.get('ood')
            metadata.stuff_classes = COMBINED_CLASS_LABELS
            metadata.things_classes = []
            metadata.stuff_colors = [[c * 255.0 for c in color] for color in CLASS_COLORS]
            visualizer_pred = Visualizer(rgb_np * 255.0, metadata=metadata)
            vis_pred = visualizer_pred.draw_sem_seg(mask_preds.cpu(), area_threshold=None, alpha=0.6)
            semantic_pred_im = np.uint8(vis_pred.get_image())

            semantic_pred = wandb.Image(np.uint8(semantic_pred_im), caption=f'Pred-Semantic OOD {img_idx}')
            all_imgs[f'pred-semantic-ood'].append(semantic_pred)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=0.5)
        # lmbda = lambda epoch: (1 - (epoch/50)) ** 0.9 
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        return optimizer
    
    
if __name__ == '__main__':
    
    # Experimental setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Experiment name for Weights & Biases (default: None)')
    parser.add_argument(
        '--restore', type=str, default=None,
        help='Weights & Biases ID to restore and resume training (default: None)')
    parser.add_argument(
        '--save-dir', type=str, default='experiments/multitask',
        help='Directory in which to save this experiments. (default: experiments/multitask)')   

    # Add PyTorch Lightning Module and Trainer args
    parser = MuliTask.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    model = MuliTask(**vars(args))

    if args.experiment_name is None:
        args.experiment_name = 'multitask'

    os.makedirs(os.path.join(args.save_dir, 'wandb'), exist_ok=True)
    wandb_logger = WandbLogger(name=args.experiment_name,
                               project='omnidata', 
                               entity='ainaz',
                               save_dir=args.save_dir,
                               version=args.restore)
    wandb_logger.watch(model, log=None, log_freq=5000)
    wandb_logger.log_hyperparams(model.hparams)
    
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints', f'{wandb_logger.name}', f'{wandb_logger.experiment.id}')
    # Save weights like ./checkpoints/taskonomy_semseg/W&BID/epoch-X.ckpt
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, '{epoch}'),
        verbose=True, monitor='val_replica_semantic_loss', mode='min', period=1, save_last=True, save_top_k=3
    )
    
    if args.restore is None:
        trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, \
            checkpoint_callback=checkpoint_callback, gpus=[0], auto_lr_find=False, accelerator='ddp', replace_sampler_ddp=False, gradient_clip_val=10)
    else:
        trainer = pl.Trainer(
            resume_from_checkpoint=os.path.join(f'{args.save_dir}/checkpoints/{wandb_logger.name}/{args.restore}/last.ckpt'), 
            logger=wandb_logger, checkpoint_callback=checkpoint_callback, gpus=[0, 1], accelerator='ddp'
        )

    # trainer.tune(model)
    print("!!! Learning rate :", model.lr)
    trainer.fit(model)
    