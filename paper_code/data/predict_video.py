import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from .segment_instance import plot_instances


def predict_instanceseg_video(video_in_path, video_out_path, model, device='cuda', normalize=False, image_size=224, batch_size=16, score_threshold=0.5):

    mean = torch.Tensor([0.485, 0.456, 0.406]) if normalize else torch.Tensor([0,0,0])
    std = torch.Tensor([0.229, 0.224, 0.225]) if normalize else torch.Tensor([1,1,1])

    image_transforms = transforms.Compose([
        transforms.Resize(image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    

    # 1. Loop over video to make batched predictions

    cap = cv2.VideoCapture(video_in_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch = []

    boxes = []
    masks = []
    labels = []
    scores = []

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        batch.append(image_transforms(img))
        
        if len(batch) >= batch_size or frame_idx >= num_frames-1:
            batch = torch.stack(batch)

            with torch.no_grad():
                model_preds = model.forward(batch.to(device))
            
            for preds in model_preds:
                boxes_pred = preds['boxes'].detach().cpu().numpy()
                masks_pred = preds['masks'][:,0].detach().cpu().numpy()
                labels_pred = preds['labels'].detach().cpu().numpy()
                scores_pred = preds['scores'].detach().cpu().numpy()

                boxes.append(boxes_pred)
                masks.append(masks_pred)
                labels.append(labels_pred)
                scores.append(scores_pred)

            batch = []

    cap.release()


    # 2. Write predictions into new video

    cap = cv2.VideoCapture(video_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dir_name = os.path.dirname(video_out_path)
    os.makedirs(dir_name, exist_ok=True)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_size, image_size))

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        img = np.array(img.resize((image_size, image_size), Image.BILINEAR)) / 255.0
        anno_pred = plot_instances(
            img, boxes[frame_idx], masks[frame_idx], labels[frame_idx], scores[frame_idx], 
            plot_scale_factor=1, score_threshold=score_threshold, return_PIL=True
        )
        anno_pred = np.array(anno_pred)
        out.write(anno_pred)
        
    cap.release()
    out.release()

    print(f'Saved annotated video under: {video_out_path}')


def predict_depth_video(video_in_path, video_out_path, model, device='cuda', image_size=256, batch_size=16):
    image_transforms = transforms.Compose([
        transforms.Resize(image_size, Image.BILINEAR),
        transforms.ToTensor()
    ])
    
    # 1. Loop over video to make batched predictions

    cap = cv2.VideoCapture(video_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch = []
    preds = []

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        batch.append(image_transforms(img))
        
        if len(batch) >= batch_size or frame_idx >= num_frames-1:
            batch = torch.stack(batch)
            with torch.no_grad():
                model_pred = model(batch.to(device))
            preds.append(model_pred.detach().cpu())
            batch = []

    preds = torch.cat(preds, dim=0).permute(0,2,3,1) # B x H x W x 1
    preds = preds.repeat_interleave(3,-1).numpy() # B x H x W x 3

    cap.release()


    # 2. Write predictions into new video    

    dir_name = os.path.dirname(video_out_path)
    os.makedirs(dir_name, exist_ok=True)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_size, image_size))

    for frame_idx in range(num_frames):
        img = (preds[frame_idx] * 255).astype(np.uint8)
        out.write(img)
        
    out.release()

    print(f'Saved annotated video under: {video_out_path}')


def predict_normal_video(video_in_path, video_out_path, model, device='cuda', image_size=256, batch_size=16):
    image_transforms = transforms.Compose([
        transforms.Resize(image_size, Image.BILINEAR),
        transforms.ToTensor()
    ])
    
    # 1. Loop over video to make batched predictions

    cap = cv2.VideoCapture(video_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch = []
    preds = []

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        batch.append(image_transforms(img))
        
        if len(batch) >= batch_size or frame_idx >= num_frames-1:
            batch = torch.stack(batch)
            with torch.no_grad():
                model_pred = model(batch.to(device))
            preds.append(model_pred.detach().cpu())
            batch = []

    preds = torch.cat(preds, dim=0).permute(0,2,3,1).numpy() # B x H x W x 3
    preds = np.clip(preds, 0, 1)

    cap.release()


    # 2. Write predictions into new video    

    dir_name = os.path.dirname(video_out_path)
    os.makedirs(dir_name, exist_ok=True)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_size, image_size))

    for frame_idx in range(num_frames):
        img = (preds[frame_idx] * 1).astype(np.uint8)
        out.write(img)
        
    out.release()

    print(f'Saved annotated video under: {video_out_path}')
