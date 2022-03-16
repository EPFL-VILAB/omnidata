import os
import pickle
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from einops import rearrange
import einops as ein
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchnet import meter
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

class SemSegTest(pl.LightningModule):
    def __init__(self, 
                pretrained_weights_path,
                 image_size, model_name, batch_size, num_workers,
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
            'image_size', 'model_name', 'batch_size', 'num_workers',
            'taskonomy_variant', 'taskonomy_root',
            'experiment_name', 'restore', 'gpus', 'distributed_backend', 'precision', 'val_check_interval', 'max_epochs',
        )
        self.pretrained_weights_path = pretrained_weights_path
        self.image_size = image_size
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = kwargs['gpus']
        self.num_workers = num_workers
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
        self.conf_matrix = meter.ConfusionMeter(len(COMBINED_CLASS_LABELS)-1, normalized=True)


        self.model = get_configured_hrnet(n_classes=len(COMBINED_CLASS_LABELS)-1,\
             load_imagenet_model=False, imagenet_ckpt_fpath='/scratch/ainaz/omnidata2/pretrained/hrnet_w48-8ef0771d.pth')
        # self.model = MultiTaskModel(tasks=['normal', 'segment_semantic', 'depth_zbuffer'], backbone='hrnet_w48',\
        #     head='hrnet', pretrained=False, dilated=False)
        
        # self.model = UNet(in_channels=3, out_channels=len(COMBINED_CLASS_LABELS)-1)

        if self.pretrained_weights_path is not None:
            checkpoint = torch.load(self.pretrained_weights_path, map_location='cuda:0')
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
            '--model_name', type=str, default='',
            help='Semantic segmentation network. (default: )')
        return parser
        
    def setup_datasets(self):
        self.num_positive = 1
        self.train_datasets = []
        if self.use_taskonomy: self.train_datasets.append('taskonomy')
        if self.use_replica: self.train_datasets.append('replica')
        if self.use_gso: self.train_datasets.append('gso')
        if self.use_hypersim: self.train_datasets.append('hypersim')

        # self.val_datasets = ['taskonomy', 'replica', 'hypersim'] #, 'gso']
        self.val_datasets = ['hypersim']

        tasks = ['rgb', 'normal', 'segment_semantic', 'depth_zbuffer', 'mask_valid']
        
        opt_test = TaskonomyReplicaGsoDataset.Options(
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

        self.testset = TaskonomyReplicaGsoDataset(options=opt_test)
        self.testset.randomize_order(seed=99)

        print('Loaded test set:')
        print(f'Test set contains {len(self.testset)} samples.')
    
    def test_dataloader(self):
        # torch.utils.data.Subset(self.testset, list(range(0, 50000)))
        return DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=False
        )
    
    def forward(self, x):
        return self.model(x)   #['segment_semantic']
    

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
        

    def test_step(self, batch, batch_idx):
        
        rgb = batch['positive']['rgb']
        semantic = batch['positive']['segment_semantic']
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid']).squeeze(1)

        labels_gt = semantic[:,:,:,0]

        # background and undefined classes are labeled as 0
        labels_gt[(semantic[:,:,:,0]==255) * (semantic[:,:,:,1]==255) * (semantic[:,:,:,2]==255)] = 0 # background in taskonomy
        labels_gt[labels_gt==-1] = 0  # undefined class in hypersim

        # mask out invalid parts of the mesh, background and undefined label
        labels_gt *= mask_valid # invalid parts of the mesh also have label (0)
        labels_gt -= 1  # the model should not predict undefined and background classes
        

        # Forward pass 
        labels_preds = self(rgb)

        labels_preds = labels_preds.permute(0, 2, 3, 1)     # (b, h, w, c)
        valid_labels_preds = labels_preds[labels_gt != -1]
        valid_labels_gt = labels_gt[labels_gt != -1]

        if valid_labels_gt.shape[0] > 0:
            self.conf_matrix.add(valid_labels_preds, valid_labels_gt)

   
 

    
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
    parser = SemSegTest.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    model = SemSegTest(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, gpus=[0])
    # result =trainer.test(verbose=True, model=model)

    # conf_matrix = model.conf_matrix.value()
    # print(conf_matrix)
    # print(conf_matrix.shape)
    # print(conf_matrix.sum())
    # with open(f'/scratch/ainaz/omnidata2/conf_matrix/{model.model_name}.pkl', 'wb') as f:
    #     pickle.dump(conf_matrix, f)

    # df_cm = pd.DataFrame(conf_matrix, index = COMBINED_CLASS_LABELS[1:], columns = COMBINED_CLASS_LABELS[1:])
    # plt.figure(figsize = (20,14))
    # sn.heatmap(df_cm, annot=False, linewidths=0.001)

    # plt.savefig(f'/scratch/ainaz/omnidata2/conf_matrix/{model.model_name}.png', dpi=300)


    ####
    with open(f'/scratch/ainaz/omnidata2/conf_matrix/{model.model_name}.pkl', 'rb') as f:
        arr = pickle.load(f)
    k = 500
    values = []

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            class1 = COMBINED_CLASS_LABELS[i+1]
            class2 = COMBINED_CLASS_LABELS[j+1]
            values.append([class1, class2, value])
    values = sorted(values, key=lambda x: x[2], reverse=True)
    values = [[c1,c2,str(v)] for [c1,c2,v] in values][:k]

    with open(f"/scratch/ainaz/omnidata2/conf_matrix/{model.model_name}.json", "w") as file:
        json.dump(values, file, indent=2)



    