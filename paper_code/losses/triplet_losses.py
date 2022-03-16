import torch
import torch.nn.functional as F
from .divergences import jensen_shannon_div, symmetric_kl_div


def mse_triplet_loss(anchor, positive, negative, margin=0.5):
    '''
    Computes the MSE triplet loss between three batches.

    Args:
        anchor: Logits Tensor of dimension (batch x classes)
        positive: Logits Tensor of dimension (batch x classes)
        negative: Logits Tensor of dimension (batch x classes)
    Returns:
        Triplet loss value, averaged over batch dimension.
    '''
    triplet_loss = 0

    mse_AP  = F.mse_loss(anchor, positive, reduction='none').mean(dim=1)
    mse_AN = F.mse_loss(anchor, negative, reduction='none').mean(dim=1)
    triplet_loss += torch.max(mse_AP - mse_AN + margin, torch.zeros_like(mse_AP)).mean()

    return triplet_loss

def kl_triplet_loss(anchor, positive, negative, margin=0.5, divergence='jensen_shannon', symmetric=False):
    '''
    Computes the triplet loss between three logits batches.

    Args:
        anchor: Logits Tensor of dimension (batch x classes)
        positive: Logits Tensor of dimension (batch x classes)
        negative: Logits Tensor of dimension (batch x classes)
        divergence: Type of divergence to compute. Either "jensen_shannon" or "symmetric_kl".
        symmetric: Set to True to return the average of the triplet loss applied to the anchor and the positive.
    Returns:
        Triplet loss value, averaged over batch dimension.
    '''
    if divergence == 'jensen_shannon':
        div = jensen_shannon_div
    elif divergence == 'symmetric_kl':
        div = symmetric_kl_div
    else:
        raise ValueError(f'{divergence} is not a supported divergence.')

    triplet_loss = 0

    div_AP  = div(anchor, positive, reduction='none').sum(dim=1)
    div_AN = div(anchor, negative, reduction='none').sum(dim=1)
    triplet_loss += torch.max(div_AP - div_AN + margin, torch.zeros_like(div_AP)).mean()

    if symmetric:
        div_PN = div(positive, negative, reduction='none').sum(dim=1)
        triplet_loss += torch.max(div_AP - div_PN + margin, torch.zeros_like(div_AP)).mean()
        triplet_loss /= 2

    return triplet_loss