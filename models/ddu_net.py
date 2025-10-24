"""
DDU-Net: Dense Dilated U-Net for Image Segmentation

This module implements the DDU-Net architecture, which combines:
- Dense blocks for feature reuse
- Dilated convolutions for larger receptive fields
- U-Net structure with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Dense block with dilated convolutions"""
    
    def __init__(self, in_channels, growth_rate, num_layers, dilation=1):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                self._make_layer(in_channels + i * growth_rate, growth_rate, dilation)
            )
    
    def _make_layer(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=dilation, dilation=dilation, bias=False)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionDown(nn.Module):
    """Transition down with pooling"""
    
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.conv(x)


class TransitionUp(nn.Module):
    """Transition up with upsampling"""
    
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
    
    def forward(self, x, skip=None):
        x = self.conv(x)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return x


class DDUNet(nn.Module):
    """
    Dense Dilated U-Net for Image Segmentation
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        init_features: Number of features in the first layer
        growth_rate: Growth rate for dense blocks
        num_layers_per_block: Number of layers in each dense block
        dilations: List of dilation rates for each encoder level
    """
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32, 
                 growth_rate=16, num_layers_per_block=4, dilations=None):
        super(DDUNet, self).__init__()
        
        if dilations is None:
            dilations = [1, 1, 2, 4, 8]
        
        features = init_features
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        self.encoder1 = DenseBlock(features, growth_rate, num_layers_per_block, dilations[0])
        features_after_dense1 = features + growth_rate * num_layers_per_block
        self.down1 = TransitionDown(features_after_dense1, features_after_dense1 // 2)
        
        features = features_after_dense1 // 2
        self.encoder2 = DenseBlock(features, growth_rate, num_layers_per_block, dilations[1])
        features_after_dense2 = features + growth_rate * num_layers_per_block
        self.down2 = TransitionDown(features_after_dense2, features_after_dense2 // 2)
        
        features = features_after_dense2 // 2
        self.encoder3 = DenseBlock(features, growth_rate, num_layers_per_block, dilations[2])
        features_after_dense3 = features + growth_rate * num_layers_per_block
        self.down3 = TransitionDown(features_after_dense3, features_after_dense3 // 2)
        
        features = features_after_dense3 // 2
        self.encoder4 = DenseBlock(features, growth_rate, num_layers_per_block, dilations[3])
        features_after_dense4 = features + growth_rate * num_layers_per_block
        self.down4 = TransitionDown(features_after_dense4, features_after_dense4 // 2)
        
        # Bottleneck
        features = features_after_dense4 // 2
        self.bottleneck = DenseBlock(features, growth_rate, num_layers_per_block, dilations[4])
        features_after_bottleneck = features + growth_rate * num_layers_per_block
        
        # Decoder path
        self.up4 = TransitionUp(features_after_bottleneck, features_after_bottleneck)
        decoder4_in = features_after_bottleneck + features_after_dense4
        self.decoder4 = DenseBlock(decoder4_in, growth_rate, num_layers_per_block, dilations[3])
        features_after_decoder4 = decoder4_in + growth_rate * num_layers_per_block
        
        self.up3 = TransitionUp(features_after_decoder4, features_after_decoder4)
        decoder3_in = features_after_decoder4 + features_after_dense3
        self.decoder3 = DenseBlock(decoder3_in, growth_rate, num_layers_per_block, dilations[2])
        features_after_decoder3 = decoder3_in + growth_rate * num_layers_per_block
        
        self.up2 = TransitionUp(features_after_decoder3, features_after_decoder3)
        decoder2_in = features_after_decoder3 + features_after_dense2
        self.decoder2 = DenseBlock(decoder2_in, growth_rate, num_layers_per_block, dilations[1])
        features_after_decoder2 = decoder2_in + growth_rate * num_layers_per_block
        
        self.up1 = TransitionUp(features_after_decoder2, features_after_decoder2)
        decoder1_in = features_after_decoder2 + features_after_dense1
        self.decoder1 = DenseBlock(decoder1_in, growth_rate, num_layers_per_block, dilations[0])
        features_after_decoder1 = decoder1_in + growth_rate * num_layers_per_block
        
        # Final output
        self.final_conv = nn.Conv2d(features_after_decoder1, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv_init(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        x = self.down1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.down2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.down3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.down4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up4(x, enc4)
        x = self.decoder4(x)
        
        x = self.up3(x, enc3)
        x = self.decoder3(x)
        
        x = self.up2(x, enc2)
        x = self.decoder2(x)
        
        x = self.up1(x, enc1)
        x = self.decoder1(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x


def get_ddu_net(in_channels=3, out_channels=1, **kwargs):
    """
    Factory function to create DDU-Net model
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        **kwargs: Additional arguments for DDUNet
    
    Returns:
        DDUNet model instance
    """
    return DDUNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
