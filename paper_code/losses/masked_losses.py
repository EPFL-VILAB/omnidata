import torch
import numpy as np

def masked_l1_loss(preds, target, mask_valid):
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()

def masked_mse_loss(preds, target, mask_valid):
    element_wise_loss = (preds - target)**2
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()

def masked_cosine_angular_loss(preds, target, mask_valid):
    preds = (2 * preds - 1).clamp(-1, 1)
    target = (2 * target - 1).clamp(-1, 1)
    mask_valid = mask_valid[:,0,:,:].bool().squeeze(1)
    preds = preds.permute(0,2,3,1)[mask_valid, :]
    target = target.permute(0,2,3,1)[mask_valid, :]
    preds_norm =  torch.nn.functional.normalize(preds, p=2, dim=1)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1)
    loss = torch.mean(-torch.sum(preds_norm * target_norm, dim = 1))
    return loss


def masked_angular_distance(preds, target, mask_valid):
    # preds = 2 * preds - 1
    # target = 2 * target - 1
    
    mask_valid = mask_valid[:,0,:,:].bool().squeeze(1)
    preds = preds.permute(0,2,3,1)[mask_valid, :]
    target = target.permute(0,2,3,1)[mask_valid, :]
    preds_norm =  torch.nn.functional.normalize(preds, p=2, dim=1)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1)
    similarity = torch.sum(preds_norm * target_norm, dim=1)    

    eps = 1e-7
    similarity = similarity.clamp(-1 + eps, 1 - eps)
    anglular_distance = 2 * torch.acos(similarity) / np.pi
    return anglular_distance.mean()

def masked_loss(element_wise_loss, mask_valid):
    element_wise_loss[~mask_valid] = 0
    if mask_valid.sum() == 0:
        return torch.tensor(0.0).to(element_wise_loss.device)
    return element_wise_loss.sum() / mask_valid.sum()