from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import layers


# DRSN-Demucs U-Net
class DDUnet(nn.Module):
    def __init__(
        self,
        chin: int = 1,
        chout: int = 1,
        hidden=32,
        depth: int = 5,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
        dilation: int = 1,
        causal: bool = False,
        norm_type: Optional[str] = "gLN",
    ):
        super().__init__()
