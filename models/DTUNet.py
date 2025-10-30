import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import (
    AbsPositionalEncoding,
    DRSNBlock,
    DiffTransformerLayer,
    CrossTransformerLayer,
    DiffCrossTransformerLayer,
    AttentionGate1D,
)


class ECA_module(nn.Module):
    def __init__(self, Channels, k_size=3):
        super(ECA_module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(EncBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.pool(self.conv(x)))


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DecBlock, self).__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="linear")
        self.relu = nn.LeakyReLU()

        self.ECA = ECA_module(Channels=out_channels)

    def forward(self, x):
        return self.ECA(self.relu(self.upsample(self.conv(x))))


class DTUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        channels = [2, 16, 32, 64, 128]
        channels = [ch // 2 for ch in channels]

        Kernal_Size = [13, 7, 7, 7]
        self.EncList1 = nn.ModuleList()
        self.EncList2 = nn.ModuleList()
        self.DecList1 = nn.ModuleList()
        self.DecList2 = nn.ModuleList()

        for i in range(4):
            self.EncList1.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=Kernal_Size[i],
                )
            )
            self.EncList2.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=Kernal_Size[i],
                )
            )
            self.DecList1.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],
                    out_channels=channels[-(i + 2)],
                    kernel_size=Kernal_Size[-(i + 1)],
                )
            )
            self.DecList2.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],
                    out_channels=channels[-(i + 2)],
                    kernel_size=Kernal_Size[-(i + 1)],
                )
            )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        encfeature1 = []
        encfeature2 = []
        for i in range(3):
            x1 = self.EncList1[i](x1)
            encfeature1.append(x1)
            x2 = self.EncList2[i](x2)
            encfeature2.append(x2)

        x1 = self.EncList1[3](x1)
        x2 = self.EncList2[3](x2)

        for i in range(3):
            x1 = self.DecList1[i](x1)
            x1 += encfeature1[-(i + 1)]
            x2 = self.DecList2[i](x2)
            x2 += encfeature2[-(i + 1)]
        x1 = self.DecList1[3](x1)
        x2 = self.DecList2[3](x2)
        x = torch.cat((x1, x2), dim=1)
        return x


if __name__ == "__main__":
    x = torch.rand(16, 2, 256)
    model = DTUNet()
    print(model)
    y = model(x)
    print(y.shape)
