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
from torch.utils.data import DataLoader, ConcatDataset
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
from data.nyu_dataset import NYUDataset, build_mask_for_eval, mask_val
from data.OASIS_dataset import OASISDataset
from models.unet import UNet
from models.multi_task_model import MultiTaskModel
from models.nips_surface_network import NIPSSurfaceNetwork
from losses import masked_l1_loss, compute_grad_norm_losses
from evaluation_metrics import get_metrics
from data.refocus_augmentation import RefocusImageAugmentation


class NormalTest(pl.LightningModule):
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
                 oasis_root,
                 use_taskonomy,
                 use_replica,
                 use_gso,
                 use_hypersim,
                 use_nyu,
                 use_oasis,
                 model_name,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters(
            'image_size', 'batch_size', 'num_workers',
            'taskonomy_variant', 'taskonomy_root', 'replica_root', 'gso_root', 'hypersim_root',
            'use_taskonomy', 'use_replica', 'use_gso', 'use_hypersim',
            'pretrained_weights_path', 'experiment_name', 'restore', 'gpus', 'distributed_backend', 'precision'
        )
        # self.pretrained_weights_path = pretrained_weights_path
        # no aug
        self.pretrained_weights_path1 = \
            '/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/164bexct/epoch=21.ckpt'
        # aug
        self.pretrained_weights_path2 = \
            '/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/49j20rhe/epoch=26.ckpt'

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
        self.oasis_root = oasis_root
        self.use_taskonomy = use_taskonomy
        self.use_replica = use_replica
        self.use_gso = use_gso
        self.use_hypersim = use_hypersim
        self.use_nyu = use_nyu
        self.use_oasis = use_oasis
        # self.model_name = model_name
        self.model_name1 = 'full-combined-no_aug-size_256_ep_21'
        self.model_name2 = 'full-combined-aug-size_256_ep_26'

        self.save_debug_info_on_error = False
        self.normalize_rgb = False
        self.setup_datasets()
        self.metrics1 = defaultdict(Statistics)
        self.metrics2 = defaultdict(Statistics)
        self.refocus_aug = RefocusImageAugmentation(10, 0.001, 5.0, return_segments=False)

        transform_blind = transforms.Compose([
            transforms.Resize(self.image_size, Image.NEAREST),
            transforms.ToTensor()
        ])
        # self.blind = transform_blind(
        #     Image.open('/scratch/ainaz/omnidata2/blind_guesses/normal_taskonomy.png'))



        #### UNet
        self.model1 = UNet(in_channels=3, out_channels=3)
        if self.pretrained_weights_path1 is not None:
            checkpoint = torch.load(self.pretrained_weights_path1, map_location='cuda:0')
            # In case we load a checkpoint from this LightningModule
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k.replace('model.', '')] = v
            else:
                state_dict = checkpoint
            self.model1.load_state_dict(state_dict)

        self.model2 = UNet(in_channels=3, out_channels=3)
        if self.pretrained_weights_path2 is not None:
            checkpoint = torch.load(self.pretrained_weights_path2, map_location='cuda:0')
            # In case we load a checkpoint from this LightningModule
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k.replace('model.', '')] = v
            else:
                state_dict = checkpoint
            self.model2.load_state_dict(state_dict)


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
            '--nyu_root', type=str, default='/datasets/nyu_official',
            help='Root directory of NYU dataset.')
        parser.add_argument(
            '--oasis_root', type=str, default='/scratch/ainaz/OASIS/OASIS_trainval/image',
            help='Root directory of OASIS dataset.')
        parser.add_argument(
            '--use_taskonomy', action='store_true', default=False,
            help='Set to use taskonomy dataset.')
        parser.add_argument(
            '--use_replica', action='store_true', default=False,
            help='Set to use replica dataset.')
        parser.add_argument(
            '--use_gso', action='store_true', default=False,
            help='Set to use GSO dataset.')
        parser.add_argument(
            '--use_hypersim', action='store_true', default=False,
            help='Set to use hypersim dataset.')
        parser.add_argument(
            '--use_nyu', action='store_true', default=False,
            help='Set to use NYU dataset.')
        parser.add_argument(
            '--use_oasis', action='store_true', default=True,
            help='Set to use OASIS dataset.')
        parser.add_argument(
            '--model_name', type=str, default='taskonomy-tiny',
            help='Name of model used for testing.')
        return parser

    def setup_datasets(self):
        self.num_positive = 1 

        tasks = ['rgb', 'normal', 'depth_euclidean', 'mask_valid']
        self.test_datasets = []
        if self.use_taskonomy: self.test_datasets.append('taskonomy')
        if self.use_replica: self.test_datasets.append('replica')
        if self.use_gso: self.test_datasets.append('gso')
        if self.use_hypersim: self.test_datasets.append('hypersim')
        if self.use_nyu: self.test_datasets.append('nyu')
        if self.use_oasis: self.test_datasets.append('oasis')

        if self.use_nyu:
            self.nyu_dataloader = NYUDataset(root=self.nyu_root, type='val', \
                task='normal', output_size=self.image_size)
            self.testset = self.nyu_dataloader.imgs

        elif self.use_oasis:
            self.oasis_dataloader = OASISDataset(root=self.oasis_root, output_size=self.image_size, normalized=False)
            self.testset = self.oasis_dataloader.imgs

        else:

            opt_test_taskonomy = TaskonomyReplicaGsoDataset.Options(
            split='test',
            taskonomy_variant='tiny',
            tasks=tasks,
            datasets=['taskonomy'],
            transform='DEFAULT',
            image_size=self.image_size,
            normalize_rgb=self.normalize_rgb,
            randomize_views=False
            )

            self.testset_taskonomy = TaskonomyReplicaGsoDataset(options=opt_test_taskonomy)

            opt_test_replica = TaskonomyReplicaGsoDataset.Options(
                split='test',
                taskonomy_variant=self.taskonomy_variant,
                tasks=tasks,
                datasets=['replica'],
                transform='DEFAULT',
                image_size=self.image_size,
                normalize_rgb=self.normalize_rgb,
                randomize_views=False
            )

            self.testset_replica = TaskonomyReplicaGsoDataset(options=opt_test_replica)

            opt_test_hypersim = TaskonomyReplicaGsoDataset.Options(
                split='test',
                taskonomy_variant=self.taskonomy_variant,
                tasks=tasks,
                datasets=['hypersim'],
                transform='DEFAULT',
                image_size=self.image_size,
                normalize_rgb=self.normalize_rgb,
                randomize_views=False
            )

            self.testset_hypersim = TaskonomyReplicaGsoDataset(options=opt_test_hypersim)

            opt_test_gso = TaskonomyReplicaGsoDataset.Options(
                split='test',
                taskonomy_variant=self.taskonomy_variant,
                tasks=tasks,
                datasets=['gso'],
                transform='DEFAULT',
                image_size=self.image_size,
                normalize_rgb=self.normalize_rgb,
                randomize_views=False
            )

            self.testset_gso = TaskonomyReplicaGsoDataset(options=opt_test_gso)


            print('Loaded test set:')
            print(f'Test set (taskonomy) contains {len(self.testset_taskonomy)} samples.')
            print(f'Test set (replica) contains {len(self.testset_replica)} samples.')
            print(f'Test set (hypersim) contains {len(self.testset_hypersim)} samples.')
            print(f'Test set (gso) contains {len(self.testset_gso)} samples.')


    def test_dataloader(self):
        if self.use_nyu:
            return self.nyu_dataloader
        elif self.use_oasis:
            # return self.oasis_dataloader
            return DataLoader(
                self.oasis_dataloader, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=False
            )
        else:
            testset = ConcatDataset([
                 self.testset_taskonomy, self.testset_replica, self.testset_hypersim, self.testset_gso])
            return DataLoader(
                testset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=False
            )

    def forward(self, x):
        return self.model1(x) #['normal']


    def test_step(self, batch, batch_idx):   
        if self.use_nyu:
            rgb = batch[0].unsqueeze(0)
            normal_gt = batch[1].unsqueeze(0)
            mask_valid = build_mask_for_eval(target=normal_gt.cpu(), val=mask_val['normal'])
            normal_preds = self(rgb)
            normal_preds = torch.clamp(normal_preds, 0, 1)

        elif self.use_oasis:
            rgb, normal_gt, mask_valid = batch
            mask_valid = build_mask_for_eval(target=mask_valid.cpu(), val=0.0)
            # Forward pass
            # normal_preds = self(rgb)
            # normal_preds = torch.clamp(normal_preds, 0, 1)
            normal_preds1 = self.model1.forward(rgb)
            normal_preds1 = torch.clamp(normal_preds1, 0, 1)
            normal_preds2 = self.model2.forward(rgb)
            normal_preds2 = torch.clamp(normal_preds2, 0, 1)

            
        else:     
            rgb = batch['positive']['rgb']
            normal_gt = batch['positive']['normal']

            # refocus augmentation
            # depth = batch['positive']['depth_euclidean']
            # if depth[depth < 1.0].shape[0] != 0:
            #     depth[depth >= 1.0] = depth[depth < 1.0].max()
            # else:
            #     depth[depth >= 1.0] = 0.99
            #     print("**")
            # rgb = self.refocus_aug(rgb, depth)

            # Forward pass
            normal_preds1 = self.model1.forward(rgb)
            normal_preds1 = torch.clamp(normal_preds1, 0, 1)
            normal_preds2 = self.model2.forward(rgb)
            normal_preds2 = torch.clamp(normal_preds2, 0, 1)

            #### Blind Guess
            # normal_preds1 = self.blind.unsqueeze(0).repeat_interleave(self.batch_size, 0).to(normal_gt.device)
            # normal_preds2 = self.blind.unsqueeze(0).repeat_interleave(self.batch_size, 0).to(normal_gt.device)
            ##############

            normal_gt = torch.clamp(normal_gt, 0, 1)

            # Mask out invalid pixels and compute loss
            mask_valid = self.make_valid_mask(batch['positive']['mask_valid']).repeat_interleave(3,1)

        # save samples
        if batch_idx % 4 == 0:
            pred1 = np.uint8(255 * normal_preds1[0].cpu().permute((1, 2, 0)).numpy())
            pred2 = np.uint8(255 * normal_preds2[0].cpu().permute((1, 2, 0)).numpy())

            gt = normal_gt[0].cpu().permute((1, 2, 0)).numpy()
            gt[(gt[:,:,0]==0) * (gt[:,:,1]==0) * (gt[:,:,2]==0)] = mask_val['normal']
            gt = np.uint8(255 * gt)

            rgb = np.uint8(255 * rgb[0].cpu().permute((1, 2, 0)).numpy())
            mask = np.uint8(255 * mask_valid[0].cpu().permute((1, 2, 0)).numpy())

            transform = transforms.Resize(512, Image.NEAREST)
            # im = Image.fromarray(rgb)
            # transform(im).save(os.path.join('test_images', 'normal', f'{self.test_datasets[0]}_blur_9', f'{batch_idx}_rgb.png'))
            # im = Image.fromarray(gt)
            # transform(im).save(os.path.join('test_images', 'normal', f'{self.test_datasets[0]}_blur_9', f'{batch_idx}_gt.png'))
            im = Image.fromarray(pred1)
            transform(im).save(os.path.join('test_images', 'normal', f'{self.test_datasets[0]}_full_noaug_ep21', f'{batch_idx}_{self.model_name1}_pred.png'))
            im = Image.fromarray(pred2)
            transform(im).save(os.path.join('test_images', 'normal', f'{self.test_datasets[0]}_full_aug_ep26', f'{batch_idx}_{self.model_name2}_pred.png'))
            # im = Image.fromarray(mask)
            # transform(im).save(os.path.join('test_images', 'normal', self.test_datasets[0], f'{batch_idx}_mask.png'))


        # for pred, target, mask in zip(normal_preds1, normal_gt, mask_valid):
        #     metrics = get_metrics(pred.cpu().unsqueeze(0), target.cpu().unsqueeze(0), \
        #         masks=mask.cpu().unsqueeze(0), task='normal')
        #     if metrics is None:
        #         print("!!!!!!! none mertrics") 
        #         continue
        #     for metric_name, metric_val in metrics.items(): 
        #         self.metrics1[metric_name].push(metric_val)

        # for pred, target, mask in zip(normal_preds2, normal_gt, mask_valid):
        #     metrics = get_metrics(pred.cpu().unsqueeze(0), target.cpu().unsqueeze(0), \
        #         masks=mask.cpu().unsqueeze(0), task='normal')
        #     if metrics is None:
        #         print("!!!!!!! none mertrics") 
        #         continue
        #     for metric_name, metric_val in metrics.items(): 
        #         self.metrics2[metric_name].push(metric_val)
    

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
    parser = NormalTest.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = NormalTest(**vars(args))
    # model = NormalTest.load_from_checkpoint(
    # checkpoint_path=args.pretrained_weights_path, **vars(args))

    trainer = Trainer.from_argparse_args(args, gpus=[0])

    result =trainer.test(verbose=True, model=model)
    print(result)

    # metrics = {}
    # for metric_name, metric_val in model.metrics1.items(): 
    #     print(f"\t{metric_name}: {metric_val.mean()} ({math.sqrt(metric_val.variance())})")
    #     metrics[metric_name] = metric_val.mean()
    #     metrics[metric_name + "_std"] = math.sqrt(metric_val.variance())

    # print("metrics1 : ", metrics)

    # metrics = {}
    # for metric_name, metric_val in model.metrics2.items(): 
    #     print(f"\t{metric_name}: {metric_val.mean()} ({math.sqrt(metric_val.variance())})")
    #     metrics[metric_name] = metric_val.mean()
    #     metrics[metric_name + "_std"] = math.sqrt(metric_val.variance())

    # print("metrics2 : ", metrics)

    # os.makedirs(os.path.join('results'), exist_ok=True)
    # metrics_file = os.path.join('results', f'metrics_normal_{model.test_datasets[0]}_model_{model.model_name}.json')

    # with open(metrics_file, 'w') as json_file:
    #     json.dump(metrics, json_file)