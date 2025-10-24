"""
Simple test to verify DDU-Net model can be instantiated and run forward pass
"""

import torch
from models import DDUNet


def test_model_creation():
    """Test that model can be created"""
    model = DDUNet(in_channels=3, out_channels=1)
    assert model is not None
    print("✓ Model created successfully")


def test_forward_pass():
    """Test forward pass with dummy data"""
    model = DDUNet(in_channels=3, out_channels=1)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1, height, width), \
        f"Expected shape {(batch_size, 1, height, width)}, got {output.shape}"
    print("✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_different_input_sizes():
    """Test with different input sizes"""
    model = DDUNet(in_channels=3, out_channels=1)
    model.eval()
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    
    for h, w in sizes:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 1, h, w), \
            f"Expected shape {(1, 1, h, w)}, got {output.shape}"
        print(f"✓ Test passed for size {h}x{w}")


def test_parameter_count():
    """Test parameter count"""
    model = DDUNet(in_channels=3, out_channels=1)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    assert total_params > 0
    assert trainable_params == total_params


if __name__ == '__main__':
    print("Running DDU-Net tests...\n")
    
    test_model_creation()
    test_forward_pass()
    test_different_input_sizes()
    test_parameter_count()
    
    print("\n✓ All tests passed!")
