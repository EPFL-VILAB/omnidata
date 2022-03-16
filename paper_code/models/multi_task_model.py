import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from seg_hrnet_multitask import hrnet_w18, hrnet_w32, hrnet_w48, HighResolutionHead, HighResolutionFuse
from resnet import resnet18, resnet50
from resnet_dilated import ResnetDilated
from aspp import DeepLabHead
from data.taskonomy_replica_gso_dataset import N_OUTPUTS


def get_backbone(name, n_channels=3, pretrained=True, dilated=False, fuse_hrnet=False):
    if name == 'resnet18':
        backbone = resnet18(pretrained=pretrained)
        backbone_channels = 512
    
    elif name == 'resnet50':
        backbone = resnet50(pretrained=pretrained)
        backbone_channels = 2048

    elif name == 'hrnet_w18':
        backbone = hrnet_w18(n_channels=n_channels, pretrained=pretrained)
        backbone_channels = [18, 36, 72, 144]
    
    elif name == 'hrnet_w32':
        backbone = hrnet_w32(pretrained=pretrained)
        backbone_channels = [32, 64, 128, 256]

    elif name == 'hrnet_w48':
        backbone = hrnet_w48(n_channels=n_channels, pretrained=pretrained)
        backbone_channels = [48, 96, 192, 384]

    else:
        raise NotImplementedError

    if dilated: # Add dilated convolutions
        assert(name in ['resnet18', 'resnet50'])
        backbone = ResnetDilated(backbone)

    if fuse_hrnet: # Fuse the multi-scale HRNet features
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels

def get_head(name, backbone_channels, task):
    """ Return the decoder head """
    if name == 'deeplab':
        return DeepLabHead(backbone_channels, N_OUTPUTS[task])

    elif name == 'hrnet':
        return HighResolutionHead(backbone_channels, N_OUTPUTS[task])

    else:
        raise NotImplementedError


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, tasks: list, n_channels, backbone, head, pretrained, dilated):
        super(MultiTaskModel, self).__init__()
        backbone, backbone_channels = get_backbone(backbone, n_channels, pretrained, dilated, fuse_hrnet=False)
        heads = torch.nn.ModuleDict({
            task: get_head(name=head, backbone_channels=backbone_channels, task=task) for task in tasks
            })
        self.backbone = backbone
        self.decoders = heads
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}

  
