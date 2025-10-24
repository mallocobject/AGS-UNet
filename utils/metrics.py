"""Evaluation metrics for segmentation"""

import torch
import torch.nn.functional as F


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient (F1 score) for binary segmentation
    
    Args:
        pred: Predicted mask (B, C, H, W)
        target: Ground truth mask (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) score
    
    Args:
        pred: Predicted mask (B, C, H, W)
        target: Ground truth mask (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred: Predicted mask (B, C, H, W)
        target: Ground truth mask (B, C, H, W)
    
    Returns:
        Pixel accuracy
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    
    return correct / total
