import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
import functools

from PIL import Image
import os
import os.path
import numpy as np
from numpy import linalg as LA
import csv
import pandas as pd
import pickle


def make_dataset(dir):
    val_csv_file = '/scratch/ainaz/OASIS/OASIS_trainval/OASIS_val.csv'
    d = pd.read_csv(val_csv_file)
    imgs = d['Image'].tolist()
    normals = d['Normal'].tolist()
    masks = d['Mask'].tolist()
    paths = []
    for i in range(len(imgs)):
        img_path = os.path.join('/scratch/ainaz', imgs[i])
        normal_path = os.path.join('/scratch/ainaz', normals[i])
        mask_path = os.path.join('/scratch/ainaz', masks[i])
        paths.append([img_path, normal_path, mask_path])

    return paths

def rgb_normal_mask_loader(path):
    rgb_path, normal_path, mask_path = path
    rgb = np.array(Image.open(rgb_path))
    with open(normal_path, 'rb') as f:
        normal = pickle.load(f)
    mask = np.array(Image.open(mask_path))

    return rgb, normal, mask


to_tensor = transforms.ToTensor() 
RGB_MEAN = torch.Tensor([0.55312, 0.52514, 0.49313]).reshape(3,1,1)
RGB_STD =  torch.Tensor([0.20555, 0.21775, 0.24044]).reshape(3,1,1)


class OASISDataset(data.Dataset):

    def __init__(self, root, output_size, normalized=False):

        imgs = make_dataset(root)
        print(len([im for im in imgs if im[1] == 1]))

        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))

        self.root = root
        self.imgs = imgs
        self.output_size = output_size

        self.transform_rgb = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.output_size, Image.BILINEAR), 
            transforms.ToTensor()])

        if normalized:
            self.transform_rgb = transforms.Compose(
                self.transform_rgb.transforms + [transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)]
            )

        self.transform_normal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.output_size, Image.NEAREST), 
            transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.imgs[index]
        rgb, gt, mask = rgb_normal_mask_loader(path)

        if len(rgb.shape) < 3:
            print("://")
            rgb = np.expand_dims(rgb, axis=2)
            rgb = np.repeat(rgb, 3, axis=2)

        gt_normal = gt['normal']
        gt_normal[:,:,2] *= -1
        gt_normal = np.uint8((gt_normal + 1) * 0.5 * 255.0)

        normal = np.zeros_like(rgb)
        normal[gt['min_y']:gt['max_y']+1, gt['min_x']:gt['max_x']+1] = gt_normal


        center_x = (gt['min_x'] + gt['max_x']) // 2
        center_y = (gt['min_y'] + gt['max_y']) // 2
        h, w = rgb.shape[0], rgb.shape[1]
        if h < w:
            if center_x > w // 2: 
                cropped_rgb = rgb[0:h, w-h:w]
                cropped_mask = mask[0:h, w-h:w]
                cropped_normal = normal[0:h, w-h:w]
            else: 
                cropped_rgb = rgb[0:h, 0:h]
                cropped_mask = mask[0:h, 0:h]
                cropped_normal = normal[0:h, 0:h]
        else:
            if center_y > h // 2: 
                cropped_rgb = rgb[h-w:h, 0:w]
                cropped_mask = mask[h-w:h, 0:w]
                cropped_normal = normal[h-w:h, 0:w]
            else: 
                cropped_rgb = rgb[0:w, 0:w]
                cropped_mask = mask[0:w, 0:w]
                cropped_normal = normal[0:w, 0:w]

        return self.transform_rgb(cropped_rgb), self.transform_normal(cropped_normal), self.transform_normal(cropped_mask)

    def __len__(self):
        return len(self.imgs)



    

    
