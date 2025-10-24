"""
Complete usage example for DDU-Net

This example demonstrates:
1. Creating a simple synthetic dataset
2. Training the model
3. Evaluating performance
4. Making predictions

Note: This is a minimal example for demonstration purposes.
For real-world usage, prepare your own dataset following the structure described in README.md
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

# Add parent directory to path for standalone execution
# Note: For production use, install the package with `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DDUNet
from utils import Trainer, dice_coefficient, iou_score


class SyntheticDataset(Dataset):
    """Simple synthetic dataset for demonstration"""
    
    def __init__(self, num_samples=50, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Pre-create coordinate grids for efficiency
        self.y, self.x = torch.meshgrid(
            torch.arange(self.image_size), 
            torch.arange(self.image_size), 
            indexing='ij'
        )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Create synthetic mask (random circles)
        mask = torch.zeros(1, self.image_size, self.image_size)
        
        # Add random shapes
        for _ in range(np.random.randint(1, 4)):
            # Random circle
            cx, cy = np.random.randint(0, self.image_size, 2)
            radius = np.random.randint(10, 40)
            circle = ((self.x - cx) ** 2 + (self.y - cy) ** 2) <= radius ** 2
            mask[0][circle] = 1.0
        
        return image, mask


def main():
    print("=" * 60)
    print("DDU-Net Complete Usage Example")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    num_epochs = 5
    batch_size = 4
    learning_rate = 1e-3
    
    # Create synthetic datasets
    print("\n1. Creating synthetic datasets...")
    train_dataset = SyntheticDataset(num_samples=40)
    val_dataset = SyntheticDataset(num_samples=10)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\n2. Creating DDU-Net model...")
    model = DDUNet(
        in_channels=3,
        out_channels=1,
        init_features=16,  # Smaller for faster demo
        growth_rate=8
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer
    trainer = Trainer(model, optimizer, device=device)
    
    # Training loop
    print(f"\n3. Training for {num_epochs} epochs...")
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n   Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}")
        
        # Track best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            print(f"   ✓ New best Dice score: {best_dice:.4f}")
    
    # Make predictions
    print("\n4. Making predictions on validation set...")
    model.eval()
    
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5
        
        dice = dice_coefficient(outputs, masks)
        iou = iou_score(outputs, masks)
        
        print(f"   Batch Dice: {dice:.4f}")
        print(f"   Batch IoU: {iou:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print("=" * 60)
    
    # Summary
    print("\nNext steps:")
    print("1. Prepare your own dataset following the structure in README.md")
    print("2. Use examples/train.py for full training pipeline")
    print("3. Use examples/inference.py for making predictions")
    print("4. Adjust hyperparameters in configs/default_config.py")


if __name__ == '__main__':
    main()
