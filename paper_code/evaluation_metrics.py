import numpy as np
import pandas as pd
import copy, os, sys, math, random, glob, time, itertools
from collections import namedtuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm 


def get_metrics(pred, target, task=None, masks=None):
    """ Gets standard set of metrics for predictions and targets """

    #original_pred = original_pred.astype('float64')
    #original_target = original_target.astype('float64')
    original_pred = pred.data.permute((0, 2, 3, 1)).double()
    original_target = target.data.permute((0, 2, 3, 1)).double()
    masks = masks.data.permute((0, 2, 3, 1))[:, :, :, 0]
    num_examples, width, height, num_channels = original_pred.shape
    _, _, _, num_channels_targ = original_pred.shape

    pred = original_pred.reshape([-1, num_channels])
    target = original_target.reshape([-1, num_channels_targ])
    flat_masks = masks.reshape(-1)
    num_valid_pixels = masks.sum()
    ratio_inverse_valid = (masks.numel() * 1.0) / (num_valid_pixels * 1.0)

    if num_valid_pixels < 1.0:
        return None
    
    if task == 'normal':

        # See https://discuss.pytorch.org/t/torch-norm-3-6x-slower-than-manually-calculating-sum-of-squares/14684/3
        norm = lambda a: np.sqrt((a * a).sum(axis=1)) 
        
        def cosine_similarity(x1, x2, dim=1, eps=1e-8):
            w12 = torch.sum(x1 * x2, dim)
            w1, w2 = norm(x1), norm(x2)
            return (w12 / (w1 * w2).clamp(min=eps)).clamp(min=-1.0, max=1.0)

        ang_errors_per_pixel_unraveled = torch.acos(cosine_similarity(pred, target)) * 180 / math.pi
        ang_errors_per_pixel = ang_errors_per_pixel_unraveled.reshape(num_examples, width, height)
        ang_errors_per_pixel_masked = ang_errors_per_pixel * masks
        ang_error_mean = torch.sum(ang_errors_per_pixel_masked) / num_valid_pixels
        ang_error_without_masking = torch.mean(ang_errors_per_pixel).item()
        
        ang_error_median = ang_errors_per_pixel_masked.flatten()
        ang_error_median = np.median(ang_error_median[flat_masks])

        threshold_1125 = (torch.sum(ang_errors_per_pixel[masks] <= 11.25).double() / num_valid_pixels).item()
        threshold_225 = (torch.sum(ang_errors_per_pixel[masks] <= 22.5).double() / num_valid_pixels).item()
        threshold_30 = (torch.sum(ang_errors_per_pixel[masks] <= 30).double() / num_valid_pixels).item()
    
    
        normed_pred = pred / (norm(pred)[:, None] + 2e-2)
        normed_target = target / (norm(target)[:, None] + 2e-2)
        diff = (normed_pred - normed_target).abs() * flat_masks.unsqueeze(1)
    else:
        diff = (pred - target).abs() * flat_masks.unsqueeze(1)

    if task == 'depth_zbuffer':
        log10 = (torch.log(1 + 64 * diff) * flat_masks.unsqueeze(1))
        log10[~flat_masks.unsqueeze(1)] = 0
        log10_diff = log10.mean() * ratio_inverse_valid
        
        log10 = ((torch.log(1 + 64 * pred) - torch.log(1 + 64 * target)) * flat_masks.unsqueeze(1)).abs().mean() * ratio_inverse_valid
        
        si_log = ((torch.log(1 + 64 * pred) - torch.log(1 + 64 * target)) * flat_masks.unsqueeze(1)).abs()
        si_log = (si_log ** 2).sum() / num_valid_pixels - (si_log.sum() ** 2) / (num_valid_pixels**2)
        rel_error = ((diff / (target)) * flat_masks.unsqueeze(1)).mean() * ratio_inverse_valid

        irmse = (((1. / (1. + 64. * pred) - 1. / (1. + 64. * target)) ** 2) * flat_masks.unsqueeze(1)).mean() * ratio_inverse_valid


    l1 = diff.abs().mean() * ratio_inverse_valid
    mse = diff ** 2
    mse = mse.mean() * ratio_inverse_valid

    #rmse = np.sqrt(np.mean(mse ** 2)) * 255.0

    return_dict = {
        "eval_mse": mse * 100,
        "eval_L1": l1 * 100,
    }

    if task == 'normal':
        return_dict.update({
            'percentage_within_11.25_degrees': threshold_1125,
            'percentage_within_22.5_degrees': threshold_225,
            'percentage_within_30_degrees': threshold_30,
            "ang_error_without_masking": ang_error_without_masking,
            "ang_error_mean": ang_error_mean,
            "ang_error_median": ang_error_median,
        })
    if task == 'depth_zbuffer':
        return_dict.update({
            'log10_diff': log10_diff,
            'log10': log10,
            'rel_error': rel_error,
            'irmse': irmse,
            'si_log': si_log,
        })

    
    return return_dict