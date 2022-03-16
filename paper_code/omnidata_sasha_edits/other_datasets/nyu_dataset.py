import torch

import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import functools

from PIL import Image

# Dataloader adapted from https://github.com/dontLoveBugs/DORN_pytorch

# Normals from https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz
iheight, iwidth = 480, 640  # raw image size
alpha, beta = 0.02, 10.0  # NYU Depth, min depth is 0.02m, max depth is 10.0m
K = 68  # NYU is 68, but in paper, 80 is good
mask_val = {'normal':0.502, 'depth_zbuffer':1.0}

import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py

IMG_EXTENSIONS = ['.h5', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def h5_loader_with_normals(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    
    head, tail = os.path.split(path)
    im_number = int(tail.split(".")[0]) - 1
    normal_path = head.replace('train/', '').replace('val/', '').replace('/official', '/normals_gt/normals')
    normal = np.array(Image.open(os.path.join(normal_path, f'{im_number:04}.png')))
    mask_path = head.replace('train/', '').replace('val/', '').replace('/official', '/normals_gt/masks')
    mask = Image.open(os.path.join(mask_path, f'{im_number:04}.png'))
    return rgb, depth, normal, mask



def h5_loader_only_normals(path, mask_val=0.5):
    rgb, depth, normal, mask = h5_loader_with_normals(path)
    normal = np.array(normal)

    mask = torch.tensor(np.array(mask))
    #mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
    mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), torch.ones(1, 1, 3, 3, device=mask.device), padding=1) != 0
    #print('loaded', mask.float().mean(), mask_val)
    normal[~mask[0, 0]] = int(np.ceil(mask_val * 255))
    return rgb, normal


# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader_only_normals):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        print(len([im for im in imgs if im[1] == 1]))


        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
    
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                                                  "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = input_np
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)

        depth_tensor = depth_np
        if self.task == 'depth_zbuffer':
            depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)



'''
In this paper, all the images are reduced to 288 x 384 from 480 x 640,
And the model are trained on random crops of size 257x353.
'''

class NYUDataset(MyDataloader):
    def __init__(self, root, type, output_size=512, sparsifier=None, modality='rgb', task='normal'):
        loader = h5_loader if task == 'depth_zbuffer' else functools.partial(h5_loader_only_normals, mask_val=mask_val[task])
        type_ = 'val' if type == 'orig_geonet' else type
        super(NYUDataset, self).__init__(root, type_, sparsifier, modality, loader=loader)
        if type == 'orig_geonet':
            self.transform = self.orig_geonet_transform_transform
        self.task = task
        #self.output_size = (257, 353)
        self.output_size = (output_size, output_size)
        self.mask_val = mask_val[self.task]

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        scaled_size = int(s * iheight)
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            # transforms.Resize(288.0 // iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(-5.0, 5.0)),
            transforms.Resize(scaled_size),
            transforms.CenterCrop(self.output_size),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        rgb_np = transforms.ToTensor()(rgb_np).float()

        normal_np = transforms.ToTensor()(depth_np).squeeze().float()
        normal_np = transform(normal_np)
        mask = build_mask(transforms.ToTensor()(normal_np), mask_val[self.task], tol=0.01)[0]
        z = transforms.ToTensor()(normal_np).clone()
        # for x1, x2 in [(1, 2)]:
        #     z[x1], z[x2] = z[x2].clone(), z[x1].clone()
        # for k in [1]:
        #     z[k] = 1 - z[k]
        depth_np = z
        depth_np[~mask] = mask_val[self.task]

        # depth_np = transform(depth_np.float())

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(min(self.output_size)),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
            
        rgb_np = transform(rgb)
        #rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        if self.task == 'depth_zbuffer':
            depth_np = np.int32(depth_np / 128 * (2**16 - 1)) # Convert to taskonomy units
            depth_np = transform(depth_np)
            depth_np = (depth_np.float() / 8000).clamp(min=0, max=1).squeeze(0)
        elif self.task == 'normal':
            normal_np = transforms.ToTensor()(depth_np).squeeze().float()
            normal_np = transform(normal_np)
            mask = build_mask(normal_np, mask_val[self.task], tol=0.01)[0]

            z = normal_np.clone()
            for x1, x2 in [(1, 2)]:
                z[x1], z[x2] = z[x2].clone(), z[x1].clone()
            for k in [1]:
                z[k] = 1 - z[k]
            depth_np = z
            depth_np[~mask] = mask_val[self.task]

        else:
            raise NotImplementedError(f'NYU Dataset: Unrecognized task label {self.task}')
        #print(depth_np.squeeze(0))
        #print('dminmax', depth_np.min(), depth_np.max())
        return rgb_np, depth_np

    def orig_geonet_transform_transform(self, rgb, target):
        target_np = target
        input_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(481),
            transforms.CenterCrop((481, 641)),
            transforms.ToTensor()
        ])
            
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(min(self.output_size)),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
            
        rgb_np = input_transform(rgb)
        #rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        if self.task == 'depth_zbuffer':
            target_np = np.int32(target_np / 128 * (2**16 - 1)) # Convert to taskonomy units
            target_np = transform(target_np)
            target_np = (target_np.float() / 8000).clamp(min=0, max=1).squeeze(0)
        elif self.task == 'normal':
            target_np = transforms.ToTensor()(target_np).squeeze().float()
            target_np = transform(target_np)
            mask = build_mask(target_np, mask_val[self.task], tol=0.01)[0]

            z = target_np.clone()
            for x1, x2 in [(1, 2)]:
                z[x1], z[x2] = z[x2].clone(), z[x1].clone()
            for k in [1]:
                z[k] = 1 - z[k]
            target_np = z
            target_np[~mask] = mask_val[self.task]

        else:
            raise NotImplementedError(f'NYU Dataset: Unrecognized task label {self.task}')

        return rgb_np, target_np
    
def build_mask(target, val=0.0, tol=1e-3):
        target = target.unsqueeze(0)
        if target.shape[1] == 1:
            mask = ((target >= val - tol) & (target <= val + tol))
            mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
            return (~mask).expand_as(target)

        mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
        mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
        mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
        mask = (mask1 & mask2 & mask3).unsqueeze(1)
        mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
        return (~mask).expand_as(target)  


def build_mask_for_eval(target, val=0.0, tol=1e-3):
    if target.shape[1] == 1:
        mask = ((target >= val - tol) & (target <= val + tol))
        mask = F.conv2d(mask.float(), torch.ones(1, 1, 15, 15, device=mask.device), padding=7) != 0
        return (~mask).expand_as(target)

    mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
    mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
    mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
    mask = (mask1 & mask2 & mask3).unsqueeze(1)
    mask = F.conv2d(mask.float(), torch.ones(1, 1, 15, 15, device=mask.device), padding=7) != 0
    return (~mask).expand_as(target)
    
