import os
import glob
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models.unet import UNet
from data.predict_video import predict_normal_video
from train_normal import ConsistentNormal

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def predict_test_videos(model, model_name, image_size=512, batch_size=16):
    for video_in_path in glob.glob('/datasets/evaluation_ood/real_world/videos/*.mp4'):    
        video_id = os.path.basename(video_in_path).split('.')[0]
        video_out_path = f'./outputs/normal/{model_name}/{model_name}_{video_id}.mp4'
        predict_normal_video(
            video_in_path, video_out_path, model, device=device, 
            image_size=image_size, batch_size=batch_size
        )


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights_path', type=str, default=None,
        help='Path of model weights to load. (default: None)')
    parser.add_argument(
        '--model_name', type=str, default=None,
        help='Name to save videos with. (default: None)')
    parser.add_argument(
        '--image_size', type=int, default=512,
        help='Image size. (default: 512)')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size. (default: 16)')
    args = parser.parse_args()

    model = UNet(in_channels=3, out_channels=3)
    checkpoint = torch.load(args.weights_path)
    state_dict = {}
    checkpoint = checkpoint.get('state_dict', checkpoint)
    for k, v in checkpoint.items():
        state_dict[k.replace('model.', '')] = v
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    predict_test_videos(model, args.model_name, image_size=args.image_size, batch_size=args.batch_size)
    #run_validation(model)