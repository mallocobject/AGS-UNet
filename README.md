# DDU-Net

Dense Dilated U-Net (DDU-Net) is a deep learning architecture for image segmentation that combines:
- **Dense Blocks**: Feature reuse through dense connections for better gradient flow
- **Dilated Convolutions**: Larger receptive fields without increasing parameters
- **U-Net Structure**: Skip connections between encoder and decoder for precise localization

## Architecture

DDU-Net uses a U-Net structure with dense blocks at each level. The dense blocks incorporate dilated convolutions with increasing dilation rates in deeper layers, allowing the network to capture both local and global context effectively.

Key features:
- Dense blocks with configurable growth rate
- Multi-scale dilated convolutions (1, 1, 2, 4, 8)
- Transition layers for downsampling and upsampling
- Skip connections for feature fusion

## Installation

```bash
# Clone the repository
git clone https://github.com/mallocobject/DDU-Net.git
cd DDU-Net

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- numpy
- pillow
- tqdm
- matplotlib
- scikit-learn

## Usage

### Training

```python
from models import DDUNet
from utils import SegmentationDataset, Trainer, get_default_transforms
import torch.optim as optim
from torch.utils.data import DataLoader

# Create model
model = DDUNet(
    in_channels=3,
    out_channels=1,
    init_features=32,
    growth_rate=16
)

# Prepare data
image_transform, mask_transform = get_default_transforms(256)
train_dataset = SegmentationDataset(
    image_dir='data/train/images',
    mask_dir='data/train/masks',
    transform=image_transform,
    target_transform=mask_transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Train
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = Trainer(model, optimizer, device='cuda')
metrics = trainer.train_epoch(train_loader)
```

### Training with Example Script

```bash
python examples/train.py \
    --data_dir /path/to/dataset \
    --output_dir checkpoints \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### Inference

```python
from models import DDUNet
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = DDUNet(in_channels=3, out_channels=1)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
image = Image.open('image.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    mask = torch.sigmoid(output).squeeze().numpy()
```

### Inference with Example Script

```bash
python examples/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input_dir /path/to/images \
    --output_dir predictions
```

## Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── masks/
│       ├── img1.png
│       ├── img2.png
│       └── ...
└── val/
    ├── images/
    │   └── ...
    └── masks/
        └── ...
```

## Model Parameters

- `in_channels`: Number of input channels (default: 3 for RGB)
- `out_channels`: Number of output channels/classes (default: 1 for binary segmentation)
- `init_features`: Number of features in the first layer (default: 32)
- `growth_rate`: Growth rate for dense blocks (default: 16)
- `num_layers_per_block`: Number of layers in each dense block (default: 4)
- `dilations`: List of dilation rates for each level (default: [1, 1, 2, 4, 8])

## Metrics

The implementation includes the following metrics:
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
- **Pixel Accuracy**: Proportion of correctly classified pixels

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{ddunet2025,
  title={DDU-Net: Dense Dilated U-Net for Image Segmentation},
  author={Your Name},
  year={2025},
  url={https://github.com/mallocobject/DDU-Net}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This implementation is inspired by:
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- DenseNet: Densely Connected Convolutional Networks
- Dilated Convolutions for semantic segmentation
