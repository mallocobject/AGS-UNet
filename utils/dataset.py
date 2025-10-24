"""Dataset utilities for image segmentation"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    """
    Dataset class for image segmentation tasks
    
    Args:
        image_dir: Directory containing input images
        mask_dir: Directory containing segmentation masks
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to masks
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Try different mask extensions
        mask_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(self.mask_dir, mask_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image {img_name}")
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        
        return image, mask


def get_default_transforms(image_size=256):
    """
    Get default transforms for training
    
    Args:
        image_size: Size to resize images to
    
    Returns:
        Tuple of (image_transform, mask_transform)
    """
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform
