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
        chin: int = 1,  # 仅时域
        chout: int = 1,
        hidden=32,
        depth: int = 4,
        kernel_size: int = 3,
        stride: int = 2,
        causal: bool = False,
        growth: int = 1.5,
        max_hidden: int = 10_000,
        glu: bool = True,
    ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        activation = nn.GLU(dim=1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        # Encoder
        for idx in range(depth):
            encoder = [
                nn.Conv1d(
                    chin,
                    hidden,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.ReLU(),
                layers.DRSNBlock(hidden, hidden * ch_scale),
                activation,
            ]
            self.encoder.append(nn.Sequential(*encoder))

            decoder = [
                nn.Conv1d(hidden, hidden * ch_scale, kernel_size=1),
                activation,
                nn.ConvTranspose1d(
                    hidden,
                    chout,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]

            if idx > 0:
                decoder.append(nn.ReLU())

            self.decoder.append(nn.Sequential(*decoder))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = layers.BLSTM(chin, bid=not causal)

    def valid_length(self, length):
        """计算有效输入长度，确保卷积无样本丢失"""
        for _ in range(self.depth):
            length = (
                length + 2 * (self.kernel_size // 2) - self.kernel_size + 1
            ) // self.stride + 1
            length = max(length, 1)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        return length

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))  # 填充

        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
        x = x.permute(2, 0, 1)  # (T, B, C)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)  # (B, C, T)
        for decoder in reversed(self.decoder):
            skip = skips.pop()
            x = x + skip[..., : x.shape[-1]]  # 对齐长度
            x = decoder(x)
        x = x[..., :length]  # 裁剪到原始长度
        return x.squeeze(1)  # (B, T)


if __name__ == "__main__":
    model = DDUnet()
    print(model)
    x = torch.randn(100, 3600)
    y = model(x)
    print(y.shape)  # 应该输出 (100, 400)
