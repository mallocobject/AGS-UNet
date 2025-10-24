"""
Quick demo of DDU-Net model

This script demonstrates basic usage of the DDU-Net model without requiring a dataset.
"""

import torch
import numpy as np
from models import DDUNet


def demo_basic_usage():
    """Demonstrate basic model usage"""
    print("=" * 60)
    print("DDU-Net Quick Demo")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating DDU-Net model...")
    model = DDUNet(
        in_channels=3,
        out_channels=1,
        init_features=32,
        growth_rate=16
    )
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input
    print("\n2. Creating dummy input image (batch_size=1, 3 channels, 256x256)...")
    batch_size = 1
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, 3, height, width)
    print(f"   ✓ Input shape: {dummy_input.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Output shape: {output.shape}")
    
    # Apply sigmoid to get probability map
    print("\n4. Applying sigmoid activation...")
    prob_map = torch.sigmoid(output)
    print(f"   ✓ Probability map shape: {prob_map.shape}")
    print(f"   ✓ Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    
    # Binarize predictions
    print("\n5. Binarizing predictions (threshold=0.5)...")
    binary_mask = (prob_map > 0.5).float()
    print(f"   ✓ Binary mask shape: {binary_mask.shape}")
    print(f"   ✓ Positive pixels: {binary_mask.sum().item():.0f} / {binary_mask.numel()}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def demo_different_configurations():
    """Demonstrate different model configurations"""
    print("\n" + "=" * 60)
    print("Testing Different Configurations")
    print("=" * 60)
    
    configs = [
        {"name": "Small", "init_features": 16, "growth_rate": 8},
        {"name": "Default", "init_features": 32, "growth_rate": 16},
        {"name": "Large", "init_features": 64, "growth_rate": 32},
    ]
    
    for config in configs:
        print(f"\n{config['name']} configuration:")
        model = DDUNet(
            in_channels=3,
            out_channels=1,
            init_features=config['init_features'],
            growth_rate=config['growth_rate']
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"  - Init features: {config['init_features']}")
        print(f"  - Growth rate: {config['growth_rate']}")
        print(f"  - Parameters: {params:,}")


def demo_multi_class_segmentation():
    """Demonstrate multi-class segmentation"""
    print("\n" + "=" * 60)
    print("Multi-class Segmentation Example")
    print("=" * 60)
    
    num_classes = 5
    print(f"\nCreating model for {num_classes}-class segmentation...")
    
    model = DDUNet(
        in_channels=3,
        out_channels=num_classes,
        init_features=32,
        growth_rate=16
    )
    
    dummy_input = torch.randn(2, 3, 256, 256)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of classes: {num_classes}")
    
    # Apply softmax for multi-class
    probabilities = torch.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    print(f"✓ Probability shape: {probabilities.shape}")
    print(f"✓ Predicted classes shape: {predicted_classes.shape}")


if __name__ == '__main__':
    demo_basic_usage()
    demo_different_configurations()
    demo_multi_class_segmentation()
    
    print("\n✓ All demos completed successfully!\n")
