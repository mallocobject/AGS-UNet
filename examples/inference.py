"""
Example inference script for DDU-Net

This script demonstrates how to use a trained DDU-Net model for inference.
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms

from models import DDUNet


def load_model(checkpoint_path, in_channels=3, out_channels=1, device='cuda'):
    """Load trained model from checkpoint"""
    model = DDUNet(in_channels=in_channels, out_channels=out_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=256):
    """Preprocess input image"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image.size


def postprocess_mask(output, original_size):
    """Postprocess model output to mask"""
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize to original size
    mask_image = Image.fromarray(mask)
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    
    return mask_image


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}')
    model = load_model(
        args.checkpoint,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        device=device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    image_files = [f for f in os.listdir(args.input_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f'Processing {len(image_files)} images...')
    
    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        
        # Preprocess
        image_tensor, original_size = preprocess_image(image_path, args.image_size)
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
        
        # Postprocess
        mask = postprocess_mask(output, original_size)
        
        # Save result
        output_path = os.path.join(args.output_dir, image_file)
        mask.save(output_path)
        print(f'Saved prediction to {output_path}')
    
    print('Inference completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDU-Net Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directory to save predictions')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1,
                       help='Number of output channels')
    
    args = parser.parse_args()
    
    main(args)
