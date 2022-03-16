import argparse
import itertools as it
import os
import json
from tqdm import tqdm
import torch
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import cv2
from collections import namedtuple
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from collections import defaultdict
from PIL import Image
import math
import pickle
import sys

import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
import torchvision.transforms as transforms

root_dir = os.getcwd()
os.chdir('/scratch/ainaz/omnidata2')
sys.path.insert(1, '/scratch/ainaz/omnidata2')
from models.unet import UNet

os.chdir('/scratch/sasha/')
sys.path.insert(1, '/scratch/sasha/')
import soda as tta

os.chdir('/scratch/sasha/omnidata/')
sys.path.insert(2, '/scratch/sasha/omnidata/')
from transform_params import ALL_TRANSFORMS

os.chdir(root_dir)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# Experimental setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--save-dir', type=str, default=None,
    help='Save directory. (default: None)')
parser.add_argument(
    '--exp_name', type=str, default='default',
    help='Name of the experiment (default: default = all)') 
parser.add_argument(
    '--transform-name', type=str, default='baseline',
    help='File to load that will export the transforms (`transforms`) to run. (default: baseline)')

parser.add_argument(
    '--short_side_size', type=int, default=512,
    help='In the dataloader, resize short side to this. (default: 512)')    
parser.add_argument(
    '--long_side_size', type=int, default=None,
    help='In the dataloader, resize long side to this. (default: None)')    


parser.add_argument(
    '--fixed_input_height', type=int, default=512,
    help='In the eval function, resize height to this. (default: 512)')    
parser.add_argument(
    '--fixed_input_width', type=int, default=512,
    help='In the dataloader, resize width to this. (default: 512)')    
parser.add_argument(
    '--use_fixed_input_size', type=boolean_string , default=True,
    help='In the dataloader, resize width to this. (default: True)')    


parser.add_argument(
    '--n_ims_to_eval', type=int, default=1000,
    help='How many validation images to evaluate. (default: 1000)')
parser.add_argument(
    '--weights_file', type=str, default='/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/3i8uv0k8/backup9.ckpt',
    help='Weights file to use. (default: -1 = all)') 


#######################
# OASIS
#######################
class OASISNormalDataset(data.Dataset):
    def __init__(self, csv_filename, data_aug=False, img_size=256):
        super(OASISNormalDataset, self).__init__()
        print("=====================================================")
        print("Using OASISNormalDataset...")
        print("csv file name: %s" % csv_filename)

        img_names = []
        normal_names = []

        with open(csv_filename) as infile:
            next(infile) # skip header
            for line in infile:
                # Filenames are absolute directories
                img_name,_,_,normal_name,_,_,_,_,_,_,_,_,_,_ = line.split(',')
                if len(normal_name) == 0:
                    continue
                img_names.append(os.path.join('/scratch/ainaz', img_name.strip()))
                normal_names.append(os.path.join('/scratch/ainaz', normal_name.strip()))
    
        self.img_names = img_names
        self.normal_names = normal_names
        self.width = img_size #320
        self.height = img_size #240
        self.n_sample = len(self.img_names)

        self.data_aug = data_aug
        print("Network input width = %d, height = %d" % (self.width, self.height))
        print("%d samples" % (self.n_sample))
        print("Data augmentation: {}".format(self.data_aug))
        print("=====================================================")

    def __getitem__(self, index):
        color = cv2.imread(self.img_names[index]).astype(np.float32)
        normal_file = open(self.normal_names[index], 'rb')
        normal_dict = pickle.load(normal_file)


        h,w,c = color.shape
        mask = np.zeros((h,w))
        normal = np.zeros((h,w,c))

        # Stuff ROI normal into bounding box
        min_y = normal_dict['min_y']
        max_y = normal_dict['max_y']
        min_x = normal_dict['min_x']
        max_x = normal_dict['max_x']
        roi_normal = normal_dict['normal']
        try:
            normal[min_y:max_y+1, min_x:max_x+1, :] = roi_normal
            normal = normal.astype(np.float32)
        except Exception as e:
            print("Error:", self.normal_names[index])
            print(str(e))
            return 

        # Make mask
        roi_mask = np.logical_or(np.logical_or(roi_normal[:,:,0] != 0, roi_normal[:,:,1] != 0), roi_normal[:,:,2] != 0).astype(np.float32)

        mask[min_y:max_y+1, min_x:max_x+1] = roi_mask


        orig_height = color.shape[0]
        orig_width = color.shape[1]
        # Downsample training images
        color = cv2.resize(color, (self.width, self.height))
        mask = cv2.resize(mask, (self.width, self.height))
        normal = cv2.resize(normal, (self.width, self.height))

        # Data augmentation: randomly flip left to right
        if self.data_aug:
            if random.random() < 0.5:
                color = cv2.flip(color, 1)
                normal = cv2.flip(normal, 1)
                # make sure x coordinates of each vector get flipped
                normal[:,:,0] *= -1

        color = np.transpose(color, (2, 0, 1)) / 255.0  # HWC to CHW.
        normal = np.transpose(normal, (2, 0, 1))   # HWC to CHW.
        # Add one channel b/c Pytorch interpolation requires 4D tensor
        mask = mask[np.newaxis, :, :]

        return color, normal, mask, (orig_height, orig_width)

    def __len__(self):
        return self.n_sample


class OASISNormalDatasetVal(OASISNormalDataset):
    def __init__(self, csv_filename, data_aug=False, short_side_size=512, long_side_size=None):
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print("Using OASISNormalDatasetVal...")
        print("csv file name: %s" % csv_filename)
        OASISNormalDataset.__init__(self, csv_filename, data_aug=data_aug)
        self.short_side_size = short_side_size
        self.long_side_size = long_side_size

    def __getitem__(self, index):
#         color = cv2.imread(self.img_names[index]).astype(np.float32)
        color = np.array(Image.open(self.img_names[index])).astype(np.float32) # my code
        normal_file = open(self.normal_names[index], 'rb')
        normal_dict = pickle.load(normal_file)
        
        if len(color.shape) < 3:
            print("://")
            color = np.expand_dims(color, axis=2)
            color = np.repeat(color, 3, axis=2)

        h,w,c = color.shape
        mask = np.zeros((h,w))
        normal = np.zeros((h,w,c))

        # Stuff ROI normal into bounding box
        min_y = normal_dict['min_y']
        max_y = normal_dict['max_y']
        min_x = normal_dict['min_x']
        max_x = normal_dict['max_x']
        roi_normal = normal_dict['normal']
        try:
            normal[min_y:max_y+1, min_x:max_x+1, :] = roi_normal
            normal = normal.astype(np.float32)
        except Exception as e:
            print("Error:", self.normal_names[index])
            print(str(e))
            return 

        # Make mask
        roi_mask = np.logical_or(np.logical_or(roi_normal[:,:,0] != 0, roi_normal[:,:,1] != 0), roi_normal[:,:,2] != 0).astype(np.float32)

        mask[min_y:max_y+1, min_x:max_x+1] = roi_mask

        orig_height = color.shape[0]
        orig_width = color.shape[1]

        if orig_width > orig_height:
            self.height, self.width = self.short_side_size, int(self.short_side_size * orig_width / orig_height)
            if self.long_side_size is not None:
                self.width = self.long_side_size
        else:
            self.height, self.width = int(self.short_side_size * orig_height / orig_width), self.short_side_size
            if self.long_side_size is not None:
                self.height = self.long_side_size
        ####################
  
        # Downsample training images
        color = cv2.resize(color, (self.width, self.height))
        color = np.transpose(color, (2, 0, 1)) / 255.0  # HWC to CHW.
        normal = np.transpose(normal, (2, 0, 1))   # HWC to CHW.
        
        # Add one channel b/c Pytorch interpolation requires 4D tensor
        mask = mask[np.newaxis, :, :]

        return color, normal, mask, (orig_height, orig_width), self.img_names[index]

    def __len__(self):
        return self.n_sample

#######################
# Evaluation
#######################
def vis_normal(rgb, pred, gt):

    rgb = rgb * 255.0
    gt = gt * 255.0
    pred = pred * 255.0
    
    im = np.concatenate([rgb, gt, pred], axis=1)
    
#     if not is_color:
#         # BGR -> RGB
#         im = im[:,:,[2,1,0]]
    plt.figure(figsize=(16, 8))
    plt.imshow(np.uint8(im))
    plt.show()

def valid_normals(model, type, coord_change, data_loader, max_iter, verbal, front_facing = False, b_vis_normal = False, 
                  use_fixed_input_size=None, fixed_input_width=512, fixed_input_height=512):
    def angle_error(preds, truths):
        '''
        preds and truths: Nx3 pytorch tensor
        '''
        preds_norm =  torch.nn.functional.normalize(preds, p=2, dim=1)
        truths_norm = torch.nn.functional.normalize(truths, p=2, dim=1)
        angles = torch.sum(preds_norm * truths_norm, dim=1)
                
        # Clip values so that max is 1 and min is -1, but don't change intermediate values
        angles = torch.clamp(angles, -1, 1)
        angles = torch.acos(angles)
        return angles

    # In degrees
    def mean(errors):
        error_sum = 0
        total_pixels = 0
        for matrix in errors:
            error_sum += np.sum(matrix)
            total_pixels += matrix.size
        return math.degrees(error_sum / total_pixels)

    # In degrees
    def median(errors):
        return math.degrees(np.median(np.concatenate(errors)))
  
    # 11.25, 22.5, 30
    def below_threshold(errors, thresh_angle):
        num = 0
        total_pixels = 0
        for matrix in errors:
            num += np.sum(matrix < math.radians(thresh_angle))
            total_pixels += matrix.size
        return num / total_pixels

  
    print("####################################")
    print("Evaluating...")
    print("\tfront_facing = %s"  %  front_facing)
    print("\tuse_fixed_input_size = %s"  %  use_fixed_input_size)
    print(f"\tsize = ({fixed_input_height}, {fixed_input_width})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iter = 0
    with torch.no_grad():
        errors = []
        for step, data_tuple in tqdm(enumerate(data_loader), total=max_iter):
            if len(data_tuple) == 3:
                inputs, targets, target_res = data_tuple
            elif len(data_tuple) == 5:
                inputs, targets, masks, target_res, _ = data_tuple
                masks = masks.to(device)
            iter += 1
            if iter == 1:
                print(inputs.shape)
            if iter > max_iter:
                break

            input_var = Variable(inputs.to(device))
            if use_fixed_input_size:
                size = (fixed_input_height, fixed_input_width) #(512, 640) #
                input_var = torch.nn.functional.interpolate(input_var, size=size, mode='bilinear')

            output_var = model(input_var)

            targets = targets.to(device)

            orig_height = target_res[0]
            orig_width = target_res[1]
            output_var = torch.nn.functional.interpolate(output_var, size=(orig_height, orig_width), mode='bilinear')
            targets = torch.nn.functional.interpolate(targets, size=(orig_height, orig_width), mode='bilinear')
            masks = torch.nn.functional.interpolate(masks, size=(orig_height, orig_width), mode='bilinear')
            mask = masks.byte().squeeze(1) # remove color channel        
            output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
            output = output_var.permute(0,2,3,1)[mask, :]
            target = targets.permute(0,2,3,1)[mask, :]

            if front_facing :
                output[:,0] = 0 # output: torch.Size([N, 3])
                output[:,1] = 0
                output[:,2] = 1

        
            error = angle_error(output, target)

            errors.append(error.data.cpu().detach().numpy())


              
        MAE = mean(errors)
        print("Mean angle error: {} degs".format(MAE))
        below_1125 = below_threshold(errors, 11.25)
        print("% below 11.25 deg: {}".format(below_1125))
        below_225 = below_threshold(errors, 22.5)
        print("% below 22.5 deg: {}".format(below_225))
        below_30 = below_threshold(errors, 30)
        print("% below 30 deg: {}".format(below_30))
        MDAE = median(errors)
        print("Median angle error: {} degs".format(MDAE))
        sys.stdout.flush()

        results = {}
        results['MAE'] = MAE
        results['MDAE'] = MDAE
        results['11.25'] = below_1125
        results['22.5'] = below_225
        results['30'] = below_30
        return results



    
def valid(model, coord_change, data_loader, dataset_name, max_iter=1400, 
          verbal=False, b_vis_normal=False, in_thresh = None, front_facing = False, 
          use_fixed_input_size=None, fixed_input_width=512, fixed_input_height=512):
    print("Evaluation on {}".format(dataset_name))

    # NYU: x points left, y points down, z points toward us
    # SNOW: x points right, y points up, z points toward us
    # OASIS: x points right, y points down, z points toward us

    return valid_normals(model, 'OASIS', coord_change, data_loader, max_iter, verbal, front_facing, b_vis_normal,
        use_fixed_input_size=use_fixed_input_size, fixed_input_width=fixed_input_width, fixed_input_height=fixed_input_width)




#######################
# Logging
#######################
def maybe_save(q, checkpoint, best=None, save_dir='/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/3i8uv0k8/'):
    lower_better = {'MAE', 'MDAE'}
    higher_better = {''}
    if best is None:
        best = {
             'MAE': 26.0,
             'MDAE': 19.0,
             '11.25': 0.30,
             '22.5': 0.58,
             '30': 0.70,
        }
    for k in q:
        save = False
        if k in lower_better:
            save = (q[k] < best[k])
        else:
            save = (q[k] > best[k])
        if save:
            save_path = os.path.join(save_dir, f'best_{k}.pth')
            print(f"{k} of {q[k]} better than previous best ({best[k]}). Saving to {save_path}...")
            torch.save(checkpoint, save_path)
            best[k] = q[k]
    return best

def write_results(args, results):
    return json.dumps({'args': args.__dict__, 'results': results})

def evaluate_normals(pretrained_weights_path, transforms, test_data_loader, num_iters=10000, normalize_output=True,
        use_fixed_input_size=None, fixed_input_width=512, fixed_input_height=512, name='Running dataset', return_model=False):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = UNet(in_channels=3, out_channels=3)
        checkpoint = torch.load(pretrained_weights_path, map_location='cuda:0')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k.replace('model.', '')] = v
        else:
              state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval().cuda()

        def model_fn(x):
            output_var = (model(x))* 2 - 1
            output_var[:,2,:,:] *= -1
            if normalize_output:
                output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
            return output_var

        wrapper = tta.SurfaceNormalsTTAWrapper(model=model_fn, transforms=transforms, run_mode='serial', merger_fn=tta.MedianMerger)
        wrapper_fn = lambda x: torch.nn.functional.normalize(wrapper(x), p=2, dim=1)

        print("\n\n")
        print(f'+----------------------------------+')
        print(f'|  OASISNormalDatasetVal ')
        print(f'+----------------------------------+')

        test_rel_error = valid(wrapper_fn, [1., 1., -1.], test_data_loader, dataset_name='OASISNormalDatasetVal', 
                                       max_iter = num_iters, in_thresh=0.0, b_vis_normal=True, 
                                       verbal=True, front_facing=False, 
                                       use_fixed_input_size=use_fixed_input_size, fixed_input_width=fixed_input_width, fixed_input_height=fixed_input_width)

        if return_model:
            return test_rel_error, checkpoint
        return test_rel_error

def load_data_and_evaluate(
   pretrained_weights_path, transforms,
   dataset_file='/scratch/ainaz/OASIS/OASIS_trainval/OASIS_val.csv', num_iters=10000,
   short_side_size=512, long_side_size=None, 
   use_fixed_input_size=None, fixed_input_width=512, fixed_input_height=512,
   return_model=False):
    test_dataset = OASISNormalDatasetVal(csv_filename = dataset_file,
        short_side_size=short_side_size, long_side_size=long_side_size)

    test_data_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = default_collate)
    
    result = evaluate_normals(pretrained_weights_path, transforms, test_data_loader, num_iters=num_iters, 
        use_fixed_input_size=use_fixed_input_size,
        fixed_input_width=fixed_input_width,
        fixed_input_height=fixed_input_width,
        return_model=return_model)
    return result



if __name__ == '__main__':
    args = parser.parse_args()

    # small_crop_transforms = tta.Product(
    #     [
    # #         tta.SquarifyCrop(),
    #         tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
    #         tta.FiveCrops(0.8),
    #         tta.ResizeShortestEdge([512]),
    # #         tta.ResizeShortestEdge([512]),
    #     ]
    # )

    # crop_transforms = tta.Product(
    #     [
    # #         tta.SquarifyCrop(),
    #         tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
    #         tta.FiveCrops(0.9),
    #         tta.ResizeShortestEdge([512, 256]),
    # #         tta.ResizeShortestEdge([512]),
    #     ]
    # )

    # whole_transforms = tta.Product(
    #     [
    # #         tta.SquarifyCrop(),
    #         tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
    #         tta.ResizeShortestEdge([512, 256, 320, 384, 448]),
    # #         tta.ResizeShortestEdge([512], interpolation='bilinear'),

    #     ]
    # )

    # square_transforms = tta.Product(
    #     [
    #         tta.SquarifyCrop(),
    #         tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
    # #         tta.ResizeShortestEdge([512, 256, 320, 384, 448]),
    # #         tta.ResizeShortestEdge([512], interpolation='bilinear'),

    #     ]
    # )




    # transforms = whole_transforms

    # # transforms = list(it.chain(crop_transforms, whole_transforms))

    # transforms = list(it.chain(small_crop_transforms, crop_transforms, whole_transforms))

    # transforms = tta.Product(
    #     [
    #     ]
    # )
    transforms = ALL_TRANSFORMS[args.transform_name]
    
    result = load_data_and_evaluate(
        pretrained_weights_path=args.weights_file,
        transforms=transforms,
        short_side_size=args.short_side_size,
        long_side_size=args.long_side_size,
        use_fixed_input_size=args.use_fixed_input_size, fixed_input_width=args.fixed_input_width, fixed_input_height=args.fixed_input_height,
        dataset_file='/scratch/ainaz/OASIS/OASIS_trainval/OASIS_val.csv',
        num_iters=args.n_ims_to_eval  
    )
    print(result)
    
    # with open(f'{args.save_dir}/{args.exp_name}.pt', 'w') as f:
    #     json.dump({'args':args.__dict__, 'result': result}, f) 
    # torch.save({'args':args, 'result': result},  './tta_results/{config_name}.pt')
