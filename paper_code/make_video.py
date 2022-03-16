

import numpy as np
import os, sys, math, random, glob, time, itertools
from fire import Fire

#os.environ['CUDA_VISIBLE_DEVICES']='6,7'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
#os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'
import itertools
from functools import partial
import PIL
from PIL import Image
import itertools as it

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as tvf
import torch.utils.data as data
import torchvision.transforms as transforms

from tqdm import tqdm
from models.unet import UNet

# os.chdir('/scratch/sasha/')
sys.path.append('/scratch/sasha/')
import soda as tta


# torch.manual_seed(229) # cpu  vars
# torch.cuda.manual_seed_all(229) # gpu vars

import IPython

# no aug
# pretrained_weights_path = \
#             '/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/164bexct/epoch=9.ckpt'
# aug
# pretrained_weights_path = \
#     '/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/49j20rhe/epoch=11.ckpt'

# tiny + combined
# pretrained_weights_path = \
#             '/scratch/ainaz/omnidata2/exps/checkpoints/omnidata/3ef56j3n/epoch=157.ckpt'

# baseline
# pretrained_weights_path = \
#     '/scratch/ainaz/XTConsistency/models/rgb2normal_baseline.pth'


pretrained_weights_path = \
    '/scratch/ainaz/omnidata2/experiments/normal/checkpoints/omnidata/3i8uv0k8/backup9.ckpt'

video_name = 'food1'
# 'loving_vincent' '03' WhiteHouse_Tour Trump Manouchehri St_Peter 01 Air_Force oval_office Hail_Chief1


FRAME_DIR = f'/scratch/ainaz/videos/frames/{video_name}/rgb'  
OUTPUT_DIR = f'/scratch/ainaz/videos/frames/{video_name}/normal_l1_cos_loss'  

IMG_SIZE = 512


class ImageDataset(data.Dataset):

    def __init__(self, data_dir=f"data/ood_images", files=None,):

        self.files = files \
            or sorted(
                glob.glob(f"{data_dir}/*.png") 
                + glob.glob(f"{data_dir}/*.jpg") 
                + glob.glob(f"{data_dir}/*.jpeg")
            )
        size = IMG_SIZE
        self.crop_rgb_transform = transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.NEAREST), 
            transforms.CenterCrop(size)]
        )

        print("num files = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def file_loader(self, path, resize=IMG_SIZE, crop=None, seed=0):
        image_transform = self.load_image_transform(resize=resize, crop=crop, seed=seed)

        im = Image.open(open(path, 'rb'))
        self.crop_rgb_transform(im).save(path.replace('rgb', 'rgb_cropped'))

        return image_transform(Image.open(open(path, 'rb')))[0:3]

    def load_image_transform(self, resize=IMG_SIZE, crop=None, seed=0):

        size = resize
        random.seed(seed)
        crop_transform = lambda x: x
        if crop is not None:
            i = random.randint(0, size - crop)
            j = random.randint(0, size - crop)
            crop_transform = TF.crop(x, i, j, crop, crop)
        
        # blur = [GaussianBulr(self.blur_radius)] if self.blur_radius else []
        # if self.jpeg_quality is not None:
        #     blur += [partial(jpeg_compress, quality=self.jpeg_quality)]

        return transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.NEAREST), 
            transforms.CenterCrop(size), 
            crop_transform, 
            transforms.ToTensor()]
        )

    def __getitem__(self, idx):

        file = self.files[idx]
        seed = random.randint(0, 1e10)
        image = self.file_loader(file, seed=seed)
        if hasattr(image, 'shape') and image.shape[0] == 1: image = image.expand(3, -1, -1)

        return image
    
# Reduce flickering
def crop_fw(frames, crop_width=0.9):
    left, top = int(frames.shape[-2] * (1 - crop_width) / 2), int(frames.shape[-1] * (1 - crop_width) / 2)
    width, height = int(frames.shape[-2] * crop_width), int(frames.shape[-1] * crop_width)
    cropped = []
    for f in frames:
        f = tvf.to_pil_image(f)
        f = tvf.resized_crop(f, top, left, height, width, frames.shape[-2:], Image.BICUBIC)
        f = tvf.to_tensor(f)
        cropped.append(f)
    cropped = torch.stack(cropped, dim=0)
    return cropped

def crop_bw(new_pred, old_pred, crop_width=0.9):
    left, top = int(old_pred.shape[-2] * (1 - crop_width) / 2), int(old_pred.shape[-1] * (1 - crop_width) / 2)
    width, height = int(old_pred.shape[-2] * crop_width), int(old_pred.shape[-1] * crop_width)
    cropped = []
    for f in new_pred:
        f = tvf.to_pil_image(f)
        f = tvf.resize(f, (width, height))
        f = tvf.to_tensor(f)
        cropped.append(f)
    cropped_pred = torch.stack(cropped, dim=0)
    pred = old_pred.detach().clone()
    pred[:,:, left:left+width, top:top+width] = cropped_pred
    return pred
    
def flip_fw(frames, x_dir_channel=None):
    flipped = []
    for f in frames:
        f = tvf.to_pil_image(f)
        f = tvf.hflip(f)
        f = tvf.to_tensor(f)
        flipped.append(f)
    flipped = torch.stack(flipped, dim=0)
    return flipped

def flip_bw(new_pred, old_pred, x_dir_channel=None):
    if x_dir_channel is not None:
        new_pred[:, x_dir_channel, ...] = 1 - new_pred[:, x_dir_channel, ...]
    return flip_fw(new_pred, x_dir_channel)

def colorjitter_fw():
    pass

def identity(frames, *args, **kwargs):
    return frames



DEFAULT_REDUCE_FLICKER_KWARGS=dict(
            use_flip=True,
            use_colorjitter=False,
            use_crop=True,
        )

def make_predict_fn(model, percep_model, multitask_slice=None):
    def thunk_(data):
        
        with torch.no_grad():
            # preds = model.forward(data.cuda()).clamp(min=0, max=1)
            preds = model(data.cuda()).clamp(min=0, max=1)

        return preds.detach().cpu()
    return thunk_

def run_viz_suite(name, data_loader, dest_task='normal',
                  graph_file=None, model_file=None, model_fn=None, model_fn_kwargs={},
                  old=False, multitask=False, multitask_slice=None,
                  percep_mode=None, percep_model_file=None, percep_model_kwargs={}, 
                  downsample=6, out_channels=3, final_task='normal', oldpercep=False, 
                  just_return_model=False, reduce_flicker_kwargs=DEFAULT_REDUCE_FLICKER_KWARGS,
                  ):
    
    unet = UNet(in_channels=3, out_channels=3).cuda()
    checkpoint = torch.load(pretrained_weights_path, map_location='cuda:0')
    # In case we load a checkpoint from this LightningModule
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('model.', '')] = v
    else:
        state_dict = checkpoint
    unet.load_state_dict(state_dict)


    ######### TTA
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
        output_var = unet(x) #* 2 - 1
        # output_var[:,2,:,:] *= -1 
        # output_var = torch.nn.functional.normalize(output_var, p=2, dim=1)
        return output_var
    
    wrapper = tta.SurfaceNormalsTTAWrapper(model=model_fn, transforms=transforms, run_mode='parallel_apply', merger_fn=tta.MedianMerger)
    model = lambda x: torch.nn.functional.normalize(wrapper(x), p=2, dim=1)
    #############


    # DATA LOADING 1
    results = []
    
    percep_model = None
    final_task = dest_task
    
    predict_fn = make_predict_fn(model, percep_model, multitask_slice=multitask_slice)

    print("Converting...")
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    i = -1
    for data in tqdm(data_loader):

        i += 1
        preds = predict_fn(data)

        with torch.no_grad():
            preds = reduce_flicker(predict_fn, data, preds,
                                   is_surface_normals=(final_task=='normal'),
                                   **reduce_flicker_kwargs)
        results.append(preds)
        continue


    return torch.cat(results, dim=0).clamp(min=0, max=1)




def predict_flipped(model, inputs, is_surface_normals=False, X_DIR_CHANNEL=0):
        flipped = torch.stack([tvf.to_tensor(tvf.hflip(tvf.to_pil_image(f))) 
             for f in inputs],
            dim=0)
        converted_flipped = model.predict_on_batch(flipped).clamp(min=0, max=1).cpu()
        if is_surface_normals:
            converted_flipped[:, X_DIR_CHANNEL, ...] = 1 - converted_flipped[:, X_DIR_CHANNEL, ...]
        converted_flipped = torch.stack([tvf.to_tensor(tvf.hflip(tvf.to_pil_image(f))) 
             for f in converted_flipped],
            dim=0)
        return converted_flipped

def reduce_flicker(model, inputs, original_preds, use_flip=False, use_colorjitter=False, use_crop=False, is_surface_normals=True,
                   n_jitter=4, crop_widths=[0.9, 0.93, 0.95, 0.98]):
    
    if not use_flip and not use_colorjitter and not use_crop:
        return  original_preds

    #print('Reducing flicker')
    to_combine = [original_preds]

    flips = [(identity, identity)]
    if use_flip:
        if is_surface_normals:
            flips.append((flip_fw, partial(flip_bw, x_dir_channel=0)))
        else:
            flips.append((flip_fw, flip_bw))

    crop_widths = crop_widths if use_crop else []
    crops = [(partial(crop_fw, crop_width=c),
              partial(crop_bw, crop_width=c))
             for c in crop_widths]
    crops.append((identity, identity))

    if use_colorjitter:
        raise NotImplementedError()
        pass
    
    for transforms in itertools.product(flips, crops):
        # Expects that:
        # Backward(Model(Forward(input))) should == Model(input)
        forward_transforms, backward_transforms = zip(*transforms)
        backward_transforms = list(backward_transforms)[::-1]

        transformed_frames = inputs
        for forward_fn in forward_transforms:
            transformed_frames = forward_fn(transformed_frames)

        predictions = model(transformed_frames).clamp(min=0, max=1).cpu()
        
        for backward_fn in list(backward_transforms)[::-1]:
            predictions = backward_fn(predictions, old_pred=original_preds)
        
        to_combine.append(predictions.to(original_preds.device))

    converted = torch.stack(to_combine, dim=0).mean(dim=0)
    return converted.to(original_preds.device)

  
    # Won't see this

    if use_flip:
        converted_flipped = predict_flipped(model, inputs, is_surface_normals)
        to_combine.append(converted_flipped.to(original_preds.device))
    
    if use_colorjitter:
        SATURATION_LIM = 0.8 #0.5
        CONTRAST_LIM = 0.6
        BRIGHTNESS_LIM = 0.7
        HUE_LIM = 0.15
        tf = torchvision.transforms.ColorJitter(brightness=[BRIGHTNESS_LIM, 1/BRIGHTNESS_LIM],
                                                contrast=[CONTRAST_LIM, 1/CONTRAST_LIM],
                                                saturation=[SATURATION_LIM, 1/SATURATION_LIM],
                                                hue=[-HUE_LIM, HUE_LIM]
                                               )
        for i in range(n_jitter):
            colorjittered = torch.stack([tvf.to_tensor(
                                   tf(
                                   tvf.to_pil_image(f))) 
                 for f in inputs],
                dim=0)
            converted_jittered = model.predict_on_batch(colorjittered).clamp(min=0, max=1)
            to_combine.append(converted_jittered)
            if use_flip:
                converted_flipped = predict_flipped(model, colorjittered, is_surface_normals)
                to_combine.append(converted_flipped.to(original_preds.device))

    converted = torch.stack(to_combine, dim=0).median(dim=0)[0]
    return converted.to(original_preds.device)


from multiprocessing import dummy as mp

def translate(x):
    return torchvision.utils.save_image(*x)


def make_video(frame_dir=FRAME_DIR,
               batch_size = 8,
               n_workers  = 16,
               config_to_run='rgb2principal_curvature',
               output_dir=OUTPUT_DIR):

    frame_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir=frame_dir),
        batch_size=batch_size,
        num_workers=n_workers, shuffle=False, pin_memory=True
    )
    print(f"Loaded {len(frame_loader)} batches ({batch_size} each) from directory: {frame_dir}")
    
    with torch.no_grad():

        results = run_viz_suite("imagepercep", frame_loader)


    start = time.time()
    print("Finished conversion: Saving...")
    for i in tqdm(range(results.size(0))):
        torchvision.utils.save_image(results[i], f"{output_dir}/output{i:05}.png")
    elapsed = time.time() - start
    print(f"Single = {elapsed}")

if __name__ == "__main__":
    Fire(make_video)
    