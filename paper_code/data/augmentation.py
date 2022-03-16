import random
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from refocus_augmentation import RefocusImageAugmentation
try:
    from kornia.augmentation import *
except:
    print("Error importing kornia augmentation")


class Augmentation:
    def __init__(self):
        self.refocus_aug = RefocusImageAugmentation(10, 0.005, 6.0, return_segments=False)

    def augment_rgb(self, batch):
            rgb = batch['positive']['rgb']
            p = random.random()

            # color jitter
            jitter = ColorJitter(
                (0.8, 1), 
                (0.8, 1.5), 
                (0.8, 1.5), 
                (-0.05, 0.05), 
                p=.6)
            
            augmented_rgb = jitter(rgb)
            
            # blur
            if p < 0.2:
                aug = RandomSharpness(.5, p=1.)
            elif p < 0.5:
                aug = RandomMotionBlur((3, 5), random.uniform(10., 30.), 0.5, p=1.)
            elif p < 0.8:
                aug = GaussianBlur((3, 3), (0.1, 2.0), p=1.)
            # elif p < 0.8:
            #     depth = batch['positive']['depth_euclidean']
            #     if depth[depth < 1.0].shape[0] != 0:
            #         depth[depth >= 1.0] = depth[depth < 1.0].max()
            #     else:
            #         depth[depth >= 1.0] = 0.99
            #         print("**")
            #     augmented_rgb = self.refocus_aug(augmented_rgb, depth)
            
            if p < 0.8: augmented_rgb = aug(augmented_rgb)

            return augmented_rgb

    def resize_augmentation(self, batch, tasks):
        p = random.random()
        if p < 0.4:
            resize_method = 'centercrop'
        elif p < 0.7:
            resize_method = 'randomcrop'
        else:
            resize_method = 'resize'

        img_sizes = [256, 320, 384, 448, 512]
        while True:
            h = random.choice(img_sizes)
            w = random.choice(img_sizes)
            if resize_method == 'resize':
                if h < 1.5 * w and w < 1.5 * h: break
            else:   
                if h < 2 * w and w < 2 * h: break


        if resize_method == 'randomcrop':
            min_x, min_y = 0, 0
            size_x, size_y = batch[tasks[0]].shape[-2], batch[tasks[0]].shape[-1]
            if size_x != h:
                min_x = random.randrange(0, size_x - h - 2)
            if size_y != w:
                min_y = random.randrange(0, size_y - w - 2)

        for task in tasks:
            if len(batch[task].shape) == 3:
                batch[task] = batch[task].unsqueeze(axis=0)

            if resize_method == 'centercrop':
                centercrop = CenterCrop((h, w), p=1.)
                batch[task] = centercrop(batch[task])

            elif resize_method == 'randomcrop':
                batch[task] = batch[task][:, :, min_x:min_x + h, min_y:min_y + w]

            elif resize_method == 'resize':

                if task == 'rgb':
                    batch[task] = F.interpolate(batch[task], (h, w), mode='bilinear')
                else:
                    batch[task] = F.interpolate(batch[task], (h, w), mode='nearest')

        return batch