import torch
import torch.nn as nn
import numpy as np

from .masked_losses import masked_l1_loss


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1


def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    depth_preds_nan = depth_preds.clone()
    depth_gt_nan = depth_gt.clone()
    depth_preds_nan[~mask_valid] = np.nan
    depth_gt_nan[~mask_valid] = np.nan

    mask_diff = mask_valid.view(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    t_gt = depth_gt_nan.view(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_gt[torch.isnan(t_gt)] = 0
    diff_gt = torch.abs(depth_gt - t_gt)
    diff_gt[~mask_valid] = 0
    s_gt = (diff_gt.view(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)


    t_pred = depth_preds_nan.view(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_pred[torch.isnan(t_pred)] = 0
    diff_pred = torch.abs(depth_preds - t_pred)
    diff_pred[~mask_valid] = 0
    s_pred = (diff_pred.view(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

    return depth_pred_aligned, depth_gt_aligned


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)



def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)



class SSIMAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_preds, depth_gt, mask_valid):
        depth_pred_aligned, depth_gt_aligned = masked_shift_and_scale(depth_preds, depth_gt, mask_valid)
        ssi_mae_loss = masked_l1_loss(depth_pred_aligned, depth_gt_aligned, mask_valid)
        return ssi_mae_loss


class GradientMatchingTerm(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class MidasLoss(nn.Module):
    def __init__(self, alpha=0.1, scales=4, reduction='image-based'):
        super().__init__()

        self.__ssi_mae_loss = SSIMAE()
        self.__gradient_matching_term = GradientMatchingTerm(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        prediction_inverse = 1 / (prediction.squeeze(1)+1e-6)
        target_inverse = 1 / (target.squeeze(1)+1e-6)
        ssi_loss = self.__ssi_mae_loss(prediction, target, mask)

        scale, shift = compute_scale_and_shift(prediction_inverse, target_inverse, mask.squeeze(1))
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction_inverse + shift.view(-1, 1, 1)
        reg_loss = self.__gradient_matching_term(self.__prediction_ssi, target_inverse, mask.squeeze(1))
        if self.__alpha > 0:
            total = ssi_loss + self.__alpha * reg_loss

        return total, ssi_loss, reg_loss
