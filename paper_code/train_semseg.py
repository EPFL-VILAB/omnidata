import os
import pickle
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from einops import rearrange
import einops as ein
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from losses.masked_losses import masked_loss
from models.unet_semseg import UNetSemSeg, UNetSemSegCombined
from models.seg_hrnet import get_configured_hrnet
from models.multi_task_model import MultiTaskModel
from models.unet import UNet

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

class SemanticSegmentation(pl.LightningModule):
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

        self.setup_datasets()
        self.val_samples = self.select_val_samples_for_datasets()
        self.log_val_imgs_step = 0

        # self.train_samples = self.select_train_samples_for_datasets()

        # self.model = get_configured_hrnet(n_classes=len(COMBINED_CLASS_LABELS)-1,\
        #      load_imagenet_model=True, imagenet_ckpt_fpath='/scratch/ainaz/omnidata2/pretrained/hrnet_w48-8ef0771d.pth')
        # self.model = UNet(in_channels=3, out_channels=len(COMBINED_CLASS_LABELS)-1)
        self.model = MultiTaskModel(tasks=['segment_semantic'], backbone='hrnet_w18',\
            head='hrnet', pretrained=True, dilated=False)

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
            '--use_replica', action='store_true', default=False,
            help='Set to use replica dataset.')
        parser.add_argument(
            '--use_gso', action='store_true', default=False,
            help='Set to user GSO dataset.')
        parser.add_argument(
            '--use_hypersim', action='store_true', default=True,
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
        tasks = ['rgb', 'normal', 'segment_semantic', 'depth_zbuffer', 'mask_valid']

        opt_train = TaskonomyReplicaGsoDataset.Options(
            tasks=tasks,
            datasets=self.train_datasets,
            split='train',
            taskonomy_variant=self.taskonomy_variant,
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=True,
            randomize_views=True
        )
        
        self.trainset = TaskonomyReplicaGsoDataset(options=opt_train)

        # opt_val_combined = TaskonomyReplicaGsoDataset.Options(
        #     split='val',
        #     taskonomy_variant=self.taskonomy_variant,
        #     tasks=tasks,
        #     datasets=['taskonomy', 'replica', 'hypersim'],
        #     transform='DEFAULT',
        #     image_size=self.image_size,
        #     normalize_rgb=True,
        #     randomize_views=False
        # )

        # self.valset_combined = TaskonomyReplicaGsoDataset(options=opt_val_combined)
        # self.valset_combined.randomize_order(seed=99)

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
        print(f'Train set contains {len(self.trainset)} samples.')
        # print(f'Validation set (combined) contains {len(self.valset_combined)} samples.')
        print(f'Validation set (taskonomy) contains {len(self.valset_taskonomy)} samples.')
        print(f'Validation set (replica) contains {len(self.valset_replica)} samples.')
        print(f'Validation set (hypersim) contains {len(self.valset_hypersim)} samples.')
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, 
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
        return [taskonomy_dl, replica_dl, hypersim_dl]

    
    def forward(self, x):
        return self.model(x)['segment_semantic']
    
    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, train=True)
        # Logging
        self.log('train_semantic_loss', res['semantic_loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        for dataset in ['taskonomy', 'replica', 'hypersim', 'gso']:
            if f'{dataset}_loss' in res.keys():
                self.log(f'train_{dataset}_loss', res[f'{dataset}_loss'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
            if f'{dataset}_loss_weight' in res.keys():
                self.log(f'{dataset}_loss_weight', res[f'{dataset}_loss_weight'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
        return {'loss': res['semantic_loss']}
    
    def validation_step_combined(self, batch, batch_idx):
        res = self.shared_step(batch, train=False)
        # Logging
        self.log('val_semantic_loss', res['loss'], prog_bar=True, logger=True, sync_dist=self.gpus>1)
        for dataset in ['taskonomy', 'replica', 'hypersim', 'gso']:
            if f'{dataset}_loss' in res.keys():
                self.log(f'val_{dataset}_loss', res[f'{dataset}_loss'], prog_bar=False, logger=True, sync_dist=self.gpus>1)
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
        rgb = batch['positive']['rgb']
        semantic = batch['positive']['segment_semantic']
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid']).squeeze(1)

        ##### GSO classes : 2**8 * r + g
        # building_in_gso_vectorized = np.vectorize(building_in_gso)
        # gso_buildings = 1 * torch.tensor(building_in_gso_vectorized(batch['positive']['building'])).to(semantic.device)
        # gso_buildings = gso_buildings.reshape(gso_buildings.shape[0], 1, 1).repeat_interleave(self.image_size,1).repeat_interleave(self.image_size,2)
        # labels_gt = (1-gso_buildings) * semantic[:,:,:,0] + gso_buildings * (2**8 * semantic[:,:,:,0] + semantic[:,:,:,1])
        labels_gt = semantic[:,:,:,0]

        # background and undefined classes are labeled as 0
        labels_gt[(semantic[:,:,:,0]==255) * (semantic[:,:,:,1]==255) * (semantic[:,:,:,2]==255)] = 0 # background in taskonomy
        labels_gt[labels_gt==-1] = 0  # undefined class in hypersim

        # mask out invalid parts of the mesh, background and undefined label
        labels_gt *= mask_valid # invalid parts of the mesh also have label (0)
        labels_gt -= 1  # the model should not predict undefined and background classes

        # Forward pass 
        labels_preds = self(rgb)

        total_loss = criterion(labels_preds, labels_gt)
        
        step_results.update({
            'semantic_loss': total_loss
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

        for loss_name in losses:
            if loss_name.split('_')[0] in self.val_datasets:
                losses[loss_name] /= counts[loss_name.split('_')[0]]
            else:
                losses[loss_name] /= counts['all']
            self.log(f'val_{loss_name}', losses[loss_name], prog_bar=False, logger=True, sync_dist=self.gpus>1)

        # Log validation set and OOD debug images using W&B
        if self.global_step >= self.log_val_imgs_step + 4000:
            self.log_val_imgs_step = self.global_step
            self.log_validation_example_images(num_images=10)
            self.log_ood_example_images(num_images=10)

    def select_train_samples_for_datasets(self):
        frls = 0
        train_imgs = defaultdict(list)
        while len(train_imgs['hypersim']) < 30:

            idx = random.randint(0, len(self.trainset) - 1)
            example = self.trainset[idx]
            building = example['positive']['building']
            print(len(train_imgs['hypersim']), len(train_imgs['taskonomy']), len(train_imgs['hypersim']), len(train_imgs['gso']), building)
            
            if building_in_hypersim(building) and len(train_imgs['hypersim']) < 30:
                train_imgs['hypersim'].append(idx)

            elif building_in_replica(building) and len(train_imgs['replica']) < 25:
                if building.startswith('frl') and frls > 15:
                    continue
                if building.startswith('frl'): frls += 1
                train_imgs['replica'].append(idx)

            elif building_in_gso(building) and len(train_imgs['gso']) < 20:
                train_imgs['gso'].append(idx)

            elif building_in_taskonomy(building) and len(train_imgs['taskonomy']) < 30:
                train_imgs['taskonomy'].append(idx)
        return train_imgs

    def select_val_samples_for_datasets(self):
        frls = 0
        val_imgs = defaultdict(list)
        with open('/scratch/ainaz/omnidata2/val_samples/hypersim_val_indices.pkl', 'rb') as f:
            val_imgs['hypersim'] = pickle.load(f)

        # if self.val_datasets == ['hypersim']: return val_imgs

        while len(val_imgs['replica']) < 20:
            idx = random.randint(0, len(self.valset_replica) - 1)
            example = self.valset_replica[idx]
            building = example['positive']['building']
            if building.startswith('frl') and frls > 10:
                continue
            if building.startswith('frl'): frls += 1
            val_imgs['replica'].append(idx)
        while len(val_imgs['taskonomy']) < 15:
            idx = random.randint(0, len(self.valset_taskonomy) - 1)
            val_imgs['taskonomy'].append(idx)

        return val_imgs


    def log_validation_example_images(self, num_images=10):
        self.model.eval()
        all_imgs = defaultdict(list)
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)

        for dataset in self.val_datasets:
            for img_idx in self.val_samples[dataset]:
                if dataset == 'taskonomy': example = self.valset_taskonomy[img_idx]
                elif dataset == 'replica': example = self.valset_replica[img_idx]
                elif dataset == 'hypersim': example = self.valset_hypersim[img_idx]

                rgb = example['positive']['rgb'].to(self.device)
                semantic = example['positive']['segment_semantic']

                if dataset == 'gso': labels_gt = 2**8 * semantic[:,:,0] + semantic[:,:,1]
                else: labels_gt = semantic[:,:,0]
                
                # background and undefined classes are labeled as 0
                labels_gt[(semantic[:,:,0]==255) * (semantic[:,:,1]==255) * (semantic[:,:,2]==255)] = 0
                labels_gt[labels_gt==-1] = 0

                mask_valid = self.make_valid_mask(example['positive']['mask_valid']).squeeze()
                labels_gt *= mask_valid.cpu()  # final labels
                mask_valid = labels_gt != 0
                if mask_valid.sum() == 0: continue

                with torch.no_grad(): 
                    preds = self.model.forward(rgb.unsqueeze(0))['segment_semantic'].squeeze(0)

                # compute loss
                gt = labels_gt -1
                loss = criterion(preds.unsqueeze(0), gt.unsqueeze(0).to(preds.device))

                mask_preds = F.softmax(preds, dim=0)
                mask_preds = torch.argmax(mask_preds, dim=0) + 1 # model does not predict background/undefined
                mask_preds *= mask_valid.to(self.device)
                rgb_masked = (rgb.detach().clone().cpu() * RGB_STD + RGB_MEAN).permute(1,2,0).numpy()
                # rgb_masked = (rgb.detach().clone().cpu()).permute(1,2,0).numpy()

                # # error image
                mask = mask_valid.unsqueeze(axis=2).repeat_interleave(3,2) * 1.0
                # err_im = torch.clamp(loss, 0, 1).repeat_interleave(3, 0)
                # err_im = loss.detach().clone().cpu().permute(1,2,0).numpy()
                # err_im = mask * np.uint8(err_im * 255) + (1-mask) * np.uint8(rgb_masked * 255.0)

                # heatmap
                fig=plt.figure()
                ax = plt.gca()
                im = ax.imshow(loss.detach().clone().cpu().squeeze().numpy())
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                fig.canvas.draw()
                w, h = fig.get_size_inches() * fig.get_dpi()
                heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(h),int(w), 3)

                metadata = MetadataCatalog.get(dataset)
                metadata.stuff_classes = COMBINED_CLASS_LABELS
                metadata.things_classes = []
                metadata.stuff_colors = [[c * 255.0 for c in color] for color in COMBINED_CLASS_COLORS] 

                visualizer_gt = Visualizer(rgb_masked * 255.0, metadata=metadata)
                vis_gt = visualizer_gt.draw_sem_seg(labels_gt.cpu(), area_threshold=None, alpha=0.9)
                gt_im = vis_gt.get_image()
                gt_im = mask * np.uint8(gt_im) + (1-mask) * np.uint8(rgb_masked * 255.0)

                visualizer_pred = Visualizer(rgb_masked * 255.0, metadata=metadata)
                vis_pred = visualizer_pred.draw_sem_seg(mask_preds.cpu(), area_threshold=None, alpha=0.6)
                pred_im = vis_pred.get_image()
                pred_im = mask * np.uint8(pred_im) + (1-mask) * np.uint8(rgb_masked * 255.0) 

                rgb_img = wandb.Image(rgb.permute(1,2,0).detach().cpu().numpy(), caption=f'RGB {img_idx}')
                all_imgs[f'rgb-{dataset}'].append(rgb_img)
                anno_gt = wandb.Image(np.uint8(gt_im), caption=f'GT-Semantic {img_idx}')
                all_imgs[f'gt-semantic-{dataset}'].append(anno_gt)
                anno_pred = wandb.Image(np.uint8(pred_im), caption=f'Pred-Semantic {img_idx}')
                all_imgs[f'pred-semantic-{dataset}'].append(anno_pred)

                # anno_error = wandb.Image(np.uint8(err_im), caption=f'Error-Val-Semantic {img_idx}')
                # all_imgs[f'error-val-semantic-{dataset}'].append(anno_error)
                anno_heatmap = wandb.Image(np.uint8(heatmap), caption=f'Heatmap-Val-Semantic {img_idx}')
                all_imgs[f'heatmap-val-semantic-{dataset}'].append(anno_heatmap)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    def log_train_example_images(self, num_images=10):
        self.model.eval()
        all_imgs = defaultdict(list)
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)

        for dataset in self.val_datasets:
            for img_idx in self.train_samples[dataset]:
                example = self.trainset[img_idx]

                rgb = example['positive']['rgb'].to(self.device)
                semantic = example['positive']['segment_semantic']

                if dataset == 'gso': labels_gt = 2**8 * semantic[:,:,0] + semantic[:,:,1]
                else: labels_gt = semantic[:,:,0]
                
                # background and undefined classes are labeled as 0
                labels_gt[(semantic[:,:,0]==255) * (semantic[:,:,1]==255) * (semantic[:,:,2]==255)] = 0
                labels_gt[labels_gt==-1] = 0

                mask_valid = self.make_valid_mask(example['positive']['mask_valid']).squeeze()
                labels_gt *= mask_valid.cpu()  # final labels
                mask_valid = labels_gt != 0
                if mask_valid.sum() == 0: continue

                with torch.no_grad(): 
                    preds = self.model.forward(rgb.unsqueeze(0))['segment_semantic'].squeeze(0)

                # compute loss
                gt = labels_gt -1
                loss = criterion(preds.unsqueeze(0), gt.unsqueeze(0).to(preds.device))

                mask_preds = F.softmax(preds, dim=0)
                mask_preds = torch.argmax(mask_preds, dim=0) + 1 # model does not predict background/undefined
                mask_preds *= mask_valid.to(self.device)
                rgb_masked = (rgb.detach().clone().cpu() * RGB_STD + RGB_MEAN).permute(1,2,0).numpy()
                # rgb_masked = (rgb.detach().clone().cpu()).permute(1,2,0).numpy()

                # error image
                mask = mask_valid.unsqueeze(axis=2).repeat_interleave(3,2) * 1.0
                err_im = torch.clamp(loss, 0, 1).repeat_interleave(3, 0)
                err_im = loss.detach().clone().cpu().permute(1,2,0).numpy()
                err_im = mask * np.uint8(err_im * 255) + (1-mask) * np.uint8(rgb_masked * 255.0)

                # heatmap
                fig=plt.figure()
                ax = plt.gca()
                im = ax.imshow(loss.detach().clone().cpu().squeeze().numpy())
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                fig.canvas.draw()
                w, h = fig.get_size_inches() * fig.get_dpi()
                heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(h),int(w), 3)

                metadata = MetadataCatalog.get(dataset)
                metadata.stuff_classes = COMBINED_CLASS_LABELS
                metadata.things_classes = []
                metadata.stuff_colors = [[c * 255.0 for c in color] for color in COMBINED_CLASS_COLORS] 

                visualizer_gt = Visualizer(rgb_masked * 255.0, metadata=metadata)
                vis_gt = visualizer_gt.draw_sem_seg(labels_gt.cpu(), area_threshold=None, alpha=0.9)
                gt_im = vis_gt.get_image()
                gt_im = mask * np.uint8(gt_im) + (1-mask) * np.uint8(rgb_masked * 255.0)

                visualizer_pred = Visualizer(rgb_masked * 255.0, metadata=metadata)
                vis_pred = visualizer_pred.draw_sem_seg(mask_preds.cpu(), area_threshold=None, alpha=0.6)
                pred_im = vis_pred.get_image()
                pred_im = mask * np.uint8(pred_im) + (1-mask) * np.uint8(rgb_masked * 255.0) 

                rgb_img = wandb.Image(rgb.permute(1,2,0).detach().cpu().numpy(), caption=f'RGB-Train {img_idx}')
                all_imgs[f'rgb-train-{dataset}'].append(rgb_img)
                anno_gt = wandb.Image(np.uint8(gt_im), caption=f'GT-Train-Semantic {img_idx}')
                all_imgs[f'gt-train-semantic-{dataset}'].append(anno_gt)
                anno_pred = wandb.Image(np.uint8(pred_im), caption=f'Pred-Train-Semantic {img_idx}')
                all_imgs[f'pred-train-semantic-{dataset}'].append(anno_pred)
                anno_error = wandb.Image(np.uint8(err_im), caption=f'Error-Train-Semantic {img_idx}')
                all_imgs[f'error-train-semantic-{dataset}'].append(anno_error)

                anno_heatmap = wandb.Image(np.uint8(heatmap), caption=f'Heatmap-train-Semantic {img_idx}')
                all_imgs[f'heatmap-train-semantic-{dataset}'].append(anno_heatmap)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    def log_ood_example_images(self, data_dir='/datasets/evaluation_ood/real_world/images', num_images=15):
        self.model.eval()

        all_imgs = defaultdict(list)

        for img_idx in range(num_images):
            rgb = Image.open(f'{data_dir}/{img_idx:05d}.png').convert('RGB')
            rgb = self.trainset.transform['rgb'](rgb).to(self.device)

            with torch.no_grad():
                semantic_pred = self.model.forward(rgb.unsqueeze(0))['segment_semantic'].squeeze(0)
                CLASS_COLORS = COMBINED_CLASS_COLORS

            rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb_np, caption=f'RGB OOD {img_idx}')
            all_imgs['rgb-ood'].append(rgb)

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
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=0.5)
        lmbda = lambda epoch: (1 - (epoch/50)) ** 0.9 
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        return [optimizer], [scheduler]
    
    
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
        '--save-dir', type=str, default='experiments/semseg',
        help='Directory in which to save this experiments. (default: exps_semseg/)')   

    # Add PyTorch Lightning Module and Trainer args
    parser = SemanticSegmentation.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    model = SemanticSegmentation(**vars(args))

    if args.experiment_name is None:
        args.experiment_name = 'taskonomy_semseg'

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
        verbose=True, monitor='val_semantic_loss', mode='min', period=1, save_last=True, save_top_k=20
    )
    
    if args.restore is None:
        trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, \
            checkpoint_callback=checkpoint_callback, gpus=[0,1], auto_lr_find=False, accelerator='ddp')
    else:
        trainer = pl.Trainer(
            resume_from_checkpoint=os.path.join(f'./checkpoints/{wandb_logger.name}/{args.restore}/last.ckpt'), 
            logger=wandb_logger, checkpoint_callback=checkpoint_callback
        )

    # trainer.tune(model)
    print("!!! Learning rate :", model.lr)
    trainer.fit(model)
    