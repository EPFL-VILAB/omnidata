import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from collections import defaultdict, namedtuple
from PIL import Image
import math
import pickle
import sys
import itertools as it

import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
import torchvision.transforms as transforms

# os.chdir('/scratch/sasha/')
sys.path.append('/scratch/sasha/')
import soda as tta


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
        self.width = 320 #img_size #320
        self.height = 256 #img_size #240
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
    def __init__(self, csv_filename, data_aug=False):
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print("Using OASISNormalDatasetVal...")
        print("csv file name: %s" % csv_filename)
        OASISNormalDataset.__init__(self, csv_filename, data_aug=data_aug)

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

        ############ my code
        # if orig_height < orig_width:
        #     if orig_width < 1.5 * orig_height:
        #         self.height, self.width = 448, 512
        #     else:
        #         self.height, self.width = 384, 512
        # elif orig_width < orig_height:
        #     if orig_height < 1.5 * orig_width:
        #         self.height, self.width = 512, 448
        #     else:
        #         self.height, self.width = 512, 384
        # else:
        #     self.height, self.width = 512, 512

        self.height, self.width = 512, int(512 * orig_width / orig_height)
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


class OASISValidate:
    def __init__(self):
        test_file = '/scratch/ainaz/OASIS/OASIS_trainval/OASIS_val.csv'
        collate_fn = default_collate
        DataSet = OASISNormalDatasetVal
        dataset_name = 'OASISNormalDatasetVal'
        test_dataset = DataSet(csv_filename = test_file)
        self.test_data_loader = data.DataLoader(
            test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = collate_fn)

    
    def valid(self, model, device, max_iter=1400):

        # NYU: x points left, y points down, z points toward us
        # SNOW: x points right, y points up, z points toward us
        # OASIS: x points right, y points down, z points toward us

        return self.valid_normals(model, 'OASIS', self.test_data_loader, max_iter, device)

    def valid_augmented(self, model, device, max_iter=1400):
        crop_transforms = tta.Product(
            [
                tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
                # tta.HorizontalFlip(),
                tta.FiveCrops(0.9),
                tta.ResizeShortestEdge([512, 256]),
                # tta.ResizeShortestEdge([512]),
            ]
        )
        whole_transforms = tta.Product(
            [
                tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
                tta.ResizeShortestEdge([512, 256, 320, 384, 448]),
                # tta.ResizeShortestEdge([512], interpolation='nearest'),
            ]
        )
        transforms = whole_transforms
        transforms = list(it.chain(crop_transforms, whole_transforms))

        def model_fn(x):
            output_var = model(x) * 2 - 1
            output_var[:,2,:,:] *= -1 
            output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
            return output_var
        
        wrapper = tta.SurfaceNormalsTTAWrapper(model=model_fn, transforms=transforms, run_mode='parallel_apply', merger_fn=tta.MedianMerger)
        wrapper_fn = lambda x: torch.nn.functional.normalize(wrapper(x), p=2, dim=1)

        return self.valid_normals(wrapper_fn, 'OASIS', self.test_data_loader, max_iter, device)


    def valid_normals(self, model, type, data_loader, max_iter, device):
        
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

    

        # assert not model.training

    #     iter = 0
    #     with torch.no_grad():
    #         errors = []
    #         for step, data_tuple in enumerate(data_loader):
    #             if len(data_tuple) == 3:
    #                 inputs, targets, target_res = data_tuple
    #             elif len(data_tuple) == 5:
    #                 inputs, targets, masks, target_res, _ = data_tuple
    #                 masks = masks.to(device)
    #             iter += 1
    #             if iter > max_iter:
    #                 break
                    
    #             input_var = Variable(inputs.to(device))
    #             output_var = model(input_var)
    #             output_var = torch.clamp(output_var, 0, 1)
    #             targets = targets.to(device)

    #             ##### my code , change output                
    #             output_var = output_var * 2 - 1
    #             output_var[:,2,:,:] *= -1   
    #             ############

    #             orig_height = target_res[0]
    #             orig_width = target_res[1]
    #             output_var = torch.nn.functional.interpolate(output_var, size=(orig_height, orig_width), mode='nearest')
    #             targets = torch.nn.functional.interpolate(targets, size=(orig_height, orig_width), mode='bilinear')
    #             masks = torch.nn.functional.interpolate(masks, size=(orig_height, orig_width), mode='bilinear')
    #             mask = masks.byte().squeeze(1) # remove color channel        
    # #             output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
    #             output = output_var.permute(0,2,3,1)[mask, :]
    #             target = targets.permute(0,2,3,1)[mask, :]
            
    #             error = angle_error(output, target)
    #             errors.append(error.data.cpu().detach().numpy())

                
    #         MAE = mean(errors)
    #         # print("Mean angle error: {} degs".format(MAE))
    #         below_1125 = below_threshold(errors, 11.25)
    #         # print("% below 11.25 deg: {}".format(below_1125))
    #         below_225 = below_threshold(errors, 22.5)
    #         # print("% below 22.5 deg: {}".format(below_225))
    #         below_30 = below_threshold(errors, 30)
    #         # print("% below 30 deg: {}".format(below_30))
    #         MDAE = median(errors)
    #         # print("Median angle error: {} degs".format(MDAE))
    #         # sys.stdout.flush()

    #         results = {}
    #         results['MAE'] = MAE
    #         results['MDAE'] = MDAE
    #         results['11.25'] = below_1125
    #         results['22.5'] = below_225
    #         results['30'] = below_30
    #         return results



        iter = 0
        with torch.no_grad():
            errors = []
            for step, data_tuple in tqdm(enumerate(data_loader), total=max_iter):
                if len(data_tuple) == 3:
                    inputs, targets, target_res = data_tuple
                elif len(data_tuple) == 5:
                    inputs, targets, masks, target_res, _ = data_tuple
                    masks = masks.to(device)
                    targets = targets.to(device)
                iter += 1
                if iter > max_iter:
                    break
                
                orig_height = target_res[0]
                orig_width = target_res[1]
                input_var = Variable(inputs.to(device))

                size = 512
                input_var = torch.nn.functional.interpolate(input_var, size=(size, size), mode='bilinear')
                output_var = model(input_var)
                output_var = torch.nn.functional.interpolate(output_var, size=(orig_height, orig_width), mode='nearest')
                output_var = torch.clamp(output_var, -1, 1)

                targets = torch.nn.functional.interpolate(targets, size=(orig_height, orig_width), mode='bilinear')
                masks = torch.nn.functional.interpolate(masks, size=(orig_height, orig_width), mode='bilinear')
                mask = masks.byte().squeeze(1) # remove color channel  
                

                output = output_var.permute(0,2,3,1)[mask, :]
                target = targets.permute(0,2,3,1)[mask, :]


                error = angle_error(output, target)
                errors.append(error.data.cpu().detach().numpy())

        
            MAE = mean(errors)
            below_1125 = below_threshold(errors, 11.25)
            below_225 = below_threshold(errors, 22.5)
            below_30 = below_threshold(errors, 30)
            MDAE = median(errors)

            results = {}
            results['MAE'] = MAE
            results['MDAE'] = MDAE
            results['11.25'] = below_1125
            results['22.5'] = below_225
            results['30'] = below_30
            return results