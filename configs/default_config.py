"""
Default configuration for DDU-Net
"""

# Model configuration
MODEL_CONFIG = {
    'in_channels': 3,
    'out_channels': 1,
    'init_features': 32,
    'growth_rate': 16,
    'num_layers_per_block': 4,
    'dilations': [1, 1, 2, 4, 8]
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    'image_size': 256
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation': 15,
    'brightness': 0.2,
    'contrast': 0.2
}
