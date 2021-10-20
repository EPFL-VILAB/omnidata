import os
import argparse
import numpy as np
from PIL import Image
import random
import yaml
import pickle
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from einops import rearrange
import einops as ein
import torch 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.omnidata_dataset import OmnidataDataset, REPLICA_BUILDINGS
from data.augmentation import Augmentation
from modules.unet import UNet
from modules.midas.midas_net import MidasNet
from modules.midas.dpt_depth import DPTDepthModel
from losses import masked_l1_loss, virtual_normal_loss, midas_loss

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

class Depth(pl.LightningModule):
    def __init__(self, config_file, experiment_name):
        super().__init__()

        config = self.load_config(config_file)
        self.experiment_name = experiment_name
        self.pretrained_weights_path = config['pretrained_weights_path']
        self.image_size = config['image_size']
        self.batch_size = config['batch_size']
        self.gpus = config['gpus']
        self.num_workers = config['num_workers']
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']

        self.taskonomy_variant = config['taskonomy_variant']
        self.data_paths = config['data_paths']
        self.train_datasets = [dataset for dataset in config['train_datasets'].keys() if config['train_datasets'][dataset]]
        self.val_datasets = [dataset for dataset in config['val_datasets'].keys() if config['val_datasets'][dataset]]
        self.normalize_rgb = config['normalize_rgb']
        self.normalization_mean = config['normalization_mean']
        self.normalization_std = config['normalization_std']

        self.setup_datasets()
        self.save_debug_info_on_error = False
        self.num_val_images = config['num_val_images']
        self.val_samples = self.select_val_samples_for_datasets()
        self.last_log_step = 0
        self.log_step = config['log_step']
        self.save_dir = config['save_dir']
        self.save_top_k = config['save_top_k']

        self.aug = Augmentation()

        self.vnl_loss = virtual_normal_loss.VNL_Loss(1.0, 1.0, (self.image_size, self.image_size))
        self.midas_loss = midas_loss.MidasLoss(alpha=0.1)

        # self.model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        self.model = DPTDepthModel()    # DPT Hybrid

        if self.pretrained_weights_path is not None:
            checkpoint = torch.load(self.pretrained_weights_path) #, map_location="cuda:1")
            # In case we load a checkpoint from this LightningModule
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    # state_dict[k.replace('model.', '')] = v
                    state_dict[k[6:]] = v

            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)

    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        return config

    def setup_datasets(self):

        self.tasks = ['rgb', 'depth_zbuffer', 'depth_euclidean', 'mask_valid']      

        self.train_options = {}
        self.trainsets = {}
        for dataset in self.train_datasets:
            self.train_options[dataset] = OmnidataDataset.Options(
                tasks=self.tasks,
                datasets=[dataset],
                split='train',
                taskonomy_variant=self.taskonomy_variant,
                transform='DEFAULT',
                image_size=self.image_size,
                normalize_rgb=self.normalize_rgb,
                normalization_mean=self.normalization_mean,
                normalization_std=self.normalization_std,
            )   
            self.trainsets[dataset] = OmnidataDataset(options=self.train_options[dataset])
            print(f'Train set ({dataset}) contains {len(self.trainsets[dataset])} samples.')

        print('Loaded training sets...')


        self.val_options = {}
        self.valsets = {}
        for dataset in self.val_datasets:
            self.val_options[dataset] = OmnidataDataset.Options(
                split='val',
                taskonomy_variant=self.taskonomy_variant,
                tasks=self.tasks,
                datasets=[dataset],
                transform='DEFAULT',
                image_size=self.image_size,
                normalize_rgb=self.normalize_rgb,
                normalization_mean=self.normalization_mean,
                normalization_std=self.normalization_std,
            )
            self.valsets[dataset] = OmnidataDataset(options=self.val_options[dataset])
            self.valsets[dataset].randomize_order()
            print(f'Val set ({dataset}) contains {len(self.valsets[dataset])} samples.')

        print('Loaded validation sets...')


    def train_dataloader(self):
        trainsets = self.trainsets.values()
        trainsets_counts = [len(trainset) for trainset in trainsets]

        dataset_sample_count = torch.tensor(trainsets_counts)
        weights = 1. / dataset_sample_count.float()
        samples_weight = []
        for w, count in zip(weights, dataset_sample_count):
            samples_weight += [w] * count
        samples_weight = torch.tensor(samples_weight)
        
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_dataset = ConcatDataset(trainsets)
        return DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler, 
            num_workers=self.num_workers, pin_memory=False
        )

    def val_dataloader(self):
        val_dls = []
        for valset in self.valsets.values():
            dl = DataLoader(
                    valset, batch_size=self.batch_size, shuffle=False, 
                    num_workers=self.num_workers, pin_memory=False
                )

            val_dls.append(dl)    
        return val_dls   


    def forward(self, x):
        return self.model(x).unsqueeze(1) 

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, train=True)
        # Logging
        self.log('train_ssi_loss', res['ssi_loss'], prog_bar=True, logger=True, sync_dist=len(self.gpus)>1)
        self.log('train_reg_loss', res['reg_loss'], prog_bar=True, logger=True, sync_dist=len(self.gpus)>1)
        self.log('train_vn_loss', res['vn_loss'], prog_bar=True, logger=True, sync_dist=len(self.gpus)>1)
        self.log('train_depth_loss', res['depth_loss'], prog_bar=True, logger=True, sync_dist=len(self.gpus)>1)
        return {'loss': res['depth_loss']}
    
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
        elif len(mask_float.shape) == 2:
            mask_float = mask_float.unsqueeze(axis=0).unsqueeze(axis=0)

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
        
        if train:
            # resize augmentation
            batch['positive'] = self.aug.resize_augmentation(batch['positive'], self.tasks, fixed_size=384)
            # rgb augmentation
            augmented_rgb = self.aug.augment_rgb(batch)
        else:
            # rgb augmentation
            augmented_rgb = batch['positive']['rgb']

        step_results = {}
        depth_gt = batch['positive']['depth_zbuffer']
        # augmented_rgb = batch['positive']['rgb']

        # Forward pass
        depth_preds = self(augmented_rgb)
        # clamp the output
        depth_preds = torch.clamp(depth_preds, 0, 1)

        # Mask out invalid pixels and compute loss
        mask_valid = self.make_valid_mask(batch['positive']['mask_valid'])

        # MiDaS Loss
        midas_loss, ssi_loss, reg_loss = self.midas_loss(depth_preds, depth_gt, mask_valid)

        # Virtual Normal Loss
        vn_loss = self.vnl_loss(depth_preds, depth_gt)

        if train and self.global_step < 15000:
            loss = ssi_loss
            reg_loss = 0
            vn_loss = 0
        else:
            loss = ssi_loss + 0.1 * reg_loss + 10 * vn_loss

        step_results.update({
            'ssi_loss': ssi_loss,
            'reg_loss': reg_loss,
            'vn_loss': vn_loss,
            'depth_loss': loss
        })
        return step_results


    def validation_epoch_end(self, outputs):
        loss_counts = 0
        loss = 0
        for dataloader_outputs in outputs:
            for output in dataloader_outputs:
                loss_counts += 1
                loss += output['depth_loss']
 
        loss /= loss_counts
        self.log(f'val_depth_loss', loss, prog_bar=False, logger=True, sync_dist=len(self.gpus)>1)
       
        # Log validation set images 
        if self.global_step >= self.last_log_step + self.log_step or self.global_step<1000:
            self.last_log_step = self.global_step
            self.log_validation_example_images()

    def select_val_samples_for_datasets(self):
        frls = 0
        val_imgs = defaultdict(list)
        for dataset, valset in self.valsets.items():
            while len(val_imgs[dataset]) < self.num_val_images:
                idx = random.randint(0, len(valset) - 1)
                val_imgs[dataset].append(idx)

        return val_imgs
    
    def depth_to_rgb(self, img):   
        img = (img - np.min(img)) / np.ptp(img)
        cm = plt.get_cmap('inferno', 2**16)
        # pixel_colored = np.uint8(np.rint(cm(1-img) * 255))[:, :, :3]
        pixel_colored = np.uint8(img * 255)
        return pixel_colored

    def inverse_normalize(self, y):
        mean = self.normalization_mean
        std = self.normalization_std
        x = y.new(*y.size())
        x[:, 0, :, :] = y[:, 0, :, :] * std[0] + mean[0]
        x[:, 1, :, :] = y[:, 1, :, :] * std[1] + mean[1]
        x[:, 2, :, :] = y[:, 2, :, :] * std[2] + mean[2]
        return x

    def log_validation_example_images(self):

        self.model.eval()
        all_imgs = defaultdict(list)

        for dataset in self.valsets.keys():
            save_path = os.path.join(self.save_dir, 'images', self.experiment_name, dataset)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for img_idx in self.val_samples[dataset]:
                example = self.valsets[dataset][img_idx]

                rgb_pos = example['positive']['rgb'].to(self.device)

                depth_gt_pos = example['positive']['depth_zbuffer'] #.squeeze(0)

                mask_valid = self.make_valid_mask(example['positive']['mask_valid'][0]).squeeze(axis=0)
                # mask_valid = self.make_valid_mask(example['positive']['mask_valid']).squeeze(axis=0)

                depth_gt_pos[~mask_valid] = 0

                rgb_pos = rgb_pos.unsqueeze(axis=0)
                depth_gt_pos = depth_gt_pos.unsqueeze(axis=0)

                with torch.no_grad():
                    depth_preds_pos = self.model.forward(rgb_pos).unsqueeze(1) 
                    depth_preds_pos = torch.clamp(depth_preds_pos, 0, 1)
                    depth_preds_pos[~mask_valid.unsqueeze(0)] = 0

                rgb = self.inverse_normalize(rgb_pos)
                rgb = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
                rgb_im = Image.fromarray(np.uint8(255 * rgb))
                rgb_im.save(os.path.join(save_path, f'{img_idx}-rgb.png'))

                mask = mask_valid.permute(1, 2, 0).detach().cpu().numpy()
                depth_gt = depth_gt_pos[0].permute(1, 2, 0).detach().cpu().numpy()
                depth_gt = self.depth_to_rgb(depth_gt)
                depth_gt[~mask] = 0
                depth_gt_im = Image.fromarray(depth_gt.squeeze())
                depth_gt_im.save(os.path.join(save_path, f'{img_idx}-gt-depthzbuffer.png'))

                depth_pred = depth_preds_pos[0].permute(1, 2, 0).detach().cpu().numpy()
                depth_pred = self.depth_to_rgb(depth_pred)
                depth_pred[~mask] = 0
                depth_pred_im = Image.fromarray(depth_pred.squeeze())
                depth_pred_im.save(os.path.join(save_path, f'{img_idx}-pred-depthzbuffer.png'))
                

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
        '--config_file', type=str, default='config/depth.yml',
        help='Path to the config file. (default: config/depth.yml)')
    parser.add_argument(
        '--experiment_name', type=str, default='exp1',
        help='Experiment name for logging and saving checkpoints. (default: exp1)')

    # Add PyTorch Lightning Module and Trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = Depth(args.config_file, args.experiment_name)


    # Save best and last model 
    checkpoint_dir = os.path.join(model.save_dir, 'checkpoints', f'{args.experiment_name}')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, '{epoch}'),
        verbose=True, monitor='val_depth_loss', mode='min', period=1, save_last=True, save_top_k=model.save_top_k
    )

    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback,  \
                                        gpus=model.gpus, auto_lr_find=False, gradient_clip_val=10,\
                                        accelerator='ddp', replace_sampler_ddp=False)


    model.register_save_on_error_callback(
        save_model_and_batch_on_error(
            trainer.save_checkpoint,
            model.save_dir
        )
    )

    trainer.fit(model)