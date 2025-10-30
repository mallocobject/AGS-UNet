import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import (
    AttentionGate1D,
    Shrinkage,
)


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(EncBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.relu = nn.LeakyReLU()
        # self.bn = nn.BatchNorm1d(out_channels)
        self.shrinkage = Shrinkage(out_channels, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.shrinkage(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


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
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class AGSUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        channels = [2, 16, 32, 64, 128]
        Kernal_Size = [11, 5, 5, 5]
        self.EncList = nn.ModuleList()
        self.DecList = nn.ModuleList()
        self.AttentionGates = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        for i in range(4):
            self.EncList.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=Kernal_Size[i],
                )
            )
            self.DecList.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],
                    out_channels=channels[-(i + 2)],
                    kernel_size=Kernal_Size[-(i + 1)],
                )
            )

            if i < 3:
                self.AttentionGates.append(
                    AttentionGate1D(
                        F_g=channels[i + 1],
                        F_l=channels[i + 1],
                        F_int=channels[i + 1] // 2,
                    )
                )
                self.fusion_convs.append(
                    nn.Sequential(
                        nn.Conv1d(
                            channels[i + 1] * 2,
                            channels[i + 1],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                        nn.BatchNorm1d(channels[i + 1]),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, x):
        encfeature = []
        for i in range(3):
            x = self.EncList[i](x)
            encfeature.append(x)

        x = self.EncList[3](x)

        for i in range(3):
            x = self.DecList[i](x)
            attn_feature = self.AttentionGates[2 - i](g=x, x=encfeature[-(i + 1)])
            x = torch.cat((attn_feature, x), dim=1)
            x = self.fusion_convs[2 - i](x)
        return self.DecList[3](x)


if __name__ == "__main__":
    x = torch.rand(16, 2, 256)
    model = AGSUNet()
    print(model)
    y = model(x)
    print(y.shape)
