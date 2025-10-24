"""
Example training script for DDU-Net

This script demonstrates how to train the DDU-Net model on a segmentation dataset.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DDUNet
from utils import SegmentationDataset, Trainer, get_default_transforms


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data transforms
    image_transform, mask_transform = get_default_transforms(args.image_size)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(args.data_dir, 'train', 'images'),
        mask_dir=os.path.join(args.data_dir, 'train', 'masks'),
        transform=image_transform,
        target_transform=mask_transform
    )
    
    val_dataset = SegmentationDataset(
        image_dir=os.path.join(args.data_dir, 'val', 'images'),
        mask_dir=os.path.join(args.data_dir, 'val', 'masks'),
        transform=image_transform,
        target_transform=mask_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    model = DDUNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        init_features=args.init_features,
        growth_rate=args.growth_rate
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create trainer
    trainer = Trainer(model, optimizer, device=device)
    
    # Training loop
    best_dice = 0.0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
              f'Dice: {train_metrics["dice"]:.4f}, '
              f'IoU: {train_metrics["iou"]:.4f}')
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f'Val - Loss: {val_metrics["loss"]:.4f}, '
              f'Dice: {val_metrics["dice"]:.4f}, '
              f'IoU: {val_metrics["iou"]:.4f}')
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            trainer.save_checkpoint(
                os.path.join(args.output_dir, 'best_model.pth'),
                epoch,
                best_dice
            )
            print(f'Saved best model with Dice: {best_dice:.4f}')
    
    print(f'\nTraining completed! Best Dice: {best_dice:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDU-Net')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Path to save checkpoints')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1,
                       help='Number of output channels')
    parser.add_argument('--init_features', type=int, default=32,
                       help='Number of initial features')
    parser.add_argument('--growth_rate', type=int, default=16,
                       help='Growth rate for dense blocks')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
