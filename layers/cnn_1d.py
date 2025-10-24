import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, pool_size=2, pool_stride=2
    ):
        super(CNNBlock1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,  # 等价于 TF 的 SAME
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
