import torch
import torch.nn as nn


class CNNTransBlock1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, output_padding=0
    ):
        super(CNNTransBlock1D, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,  # 等价于 TF 的 SAME
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
