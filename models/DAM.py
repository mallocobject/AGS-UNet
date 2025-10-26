import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class APReLU(nn.Module):
    def __init__(self, channels):
        super(APReLU, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        fcnList = [
            nn.Linear(2 * channels, 2 * channels),
            nn.BatchNorm1d(2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        ]
        self.fcn = nn.Sequential(*fcnList)

    def forward(self, x):
        zerox = torch.zeros_like(x)
        posx = torch.max(x, zerox)
        negx = torch.min(x, zerox)

        concatx = torch.concat(
            [self.gap(posx).squeeze(-1), self.gap(negx).squeeze(-1)], dim=1
        )
        concatx = self.fcn(concatx)
        return posx + concatx.unsqueeze(2) * negx


class EncoderCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=2,
        using_APReLU=True,
    ):
        super(EncoderCell, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        if using_APReLU:
            self.activate = APReLU(out_channels)
        else:
            self.activate = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.activate(out)
        out = self.bn(out)
        return out


class DeNoiseEnc(nn.Module):
    def __init__(self, input_channels=1, using_APReLU=True):
        super().__init__()
        self.conv_kernel = [17, 17, 3, 3]
        self.paddingsize = [8, 8, 1, 1]
        self.out_channels = [2, 4, 8, 16]
        self.EncoderList = nn.ModuleList()

        current_channels = input_channels
        for i in range(4):
            self.EncoderList.add_module(
                "cell{}".format(i),
                EncoderCell(
                    in_channels=current_channels,
                    out_channels=self.out_channels[i],
                    kernel_size=self.conv_kernel[i],
                    padding=self.paddingsize[i],
                    using_APReLU=using_APReLU,
                ),
            )
            current_channels = self.out_channels[i]

    def forward(self, x):
        out = []
        for cell in self.EncoderList:
            x = cell(x)
            out.append(x)
        return out


class DAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        fcnList = [
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        ]
        self.fcn1 = nn.Sequential(*fcnList)
        self.fcn2 = nn.Sequential(*fcnList)

        # 空间注意力
        self.cap = nn.AdaptiveAvgPool1d(1)
        self.cmp = nn.AdaptiveMaxPool1d(1)
        self.convsa = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_transpose = x.transpose(1, 2)
        # Channel Attention
        gapx = self.gap(x).squeeze(2)
        gmpx = self.gmp(x).squeeze(2)
        gapx = self.fcn1(gapx)
        gmpx = self.fcn2(gmpx)
        Cattn = self.sigmoid(gapx + gmpx).unsqueeze(-1)

        # Spatial Attn
        capx = self.cap(x_transpose).transpose(1, 2)
        cmpx = self.cmp(x_transpose).transpose(1, 2)
        catcp = torch.cat((capx, cmpx), dim=1)
        Sattn = self.sigmoid(self.convsa(catcp).squeeze(1)).unsqueeze(-2)
        x = Cattn * x
        x = Sattn * x
        return x


class DecoderCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=2,
        using_APReLU=True,
        last=False,
    ):
        super().__init__()
        self.last = last

        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        if using_APReLU:
            self.activate = APReLU(out_channels)
        else:
            self.activate = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(out_channels)

        if not last:
            self.dam = DAM(out_channels)

    def forward(self, x):
        outx = self.deconv(x)
        outx = self.activate(outx)
        outx = self.bn(outx)
        if not self.last:
            outx = self.dam(outx)
        return outx


def alignment_add(tensor1, tensor2, alignment_opt="trunc"):
    """add with auto-alignment"""
    assert (
        tensor1.shape[0:-1] == tensor2.shape[0:-1]
    ), "the shape of the first tensor should be the same as the second tensor"
    short_tensor = tensor1 if tensor1.shape[-1] < tensor2.shape[-1] else tensor2
    long_tensor = tensor1 if tensor1.shape[-1] >= tensor2.shape[-1] else tensor2
    if alignment_opt == "trunc":
        return short_tensor + long_tensor[..., : short_tensor.shape[-1]]
    elif alignment_opt == "padding":
        return long_tensor + F.pad(
            short_tensor, (0, long_tensor.shape[-1] - short_tensor.shape[-1])
        )


class DeNoiseDec(nn.Module):
    def __init__(self, input_channels=16):
        super(DeNoiseDec, self).__init__()
        self.conv_kernel = [4, 4, 18, 18]
        self.paddingsize = [1, 1, 8, 8]
        self.out_channels = [8, 4, 2, 1]

        DecoderList = []
        current_channels = input_channels
        for i in range(4):
            if i != 3:
                DecoderList.append(
                    DecoderCell(
                        in_channels=current_channels,
                        out_channels=self.out_channels[i],
                        kernel_size=self.conv_kernel[i],
                        padding=self.paddingsize[i],
                        using_APReLU=True,
                    )
                )
            else:
                DecoderList.append(
                    DecoderCell(
                        in_channels=current_channels,
                        out_channels=self.out_channels[i],
                        kernel_size=self.conv_kernel[i],
                        padding=self.paddingsize[i],
                        using_APReLU=True,
                        last=True,
                    )
                )
            current_channels = self.out_channels[i]

        self.DecoderList = nn.ModuleList(DecoderList)

    def forward(self, xlist):
        y3 = self.DecoderList[0](xlist[-1])
        y2 = self.DecoderList[1](alignment_add(y3, xlist[-2]))
        y1 = self.DecoderList[2](alignment_add(y2, xlist[-3]))
        y0 = self.DecoderList[3](alignment_add(y1, xlist[-4]))
        return y0


class Seq2Seq2(nn.Module):
    def __init__(self, input_channels=1):
        super(Seq2Seq2, self).__init__()
        self.enc = DeNoiseEnc(input_channels=input_channels)
        self.dec = DeNoiseDec(input_channels=16)  # 编码器输出16通道

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度 (B, L) -> (B, 1, L)

        output = self.dec(self.enc(x))

        # 确保输出是单通道
        return output.squeeze(1)  # (B, 1, L) -> (B, L)


if __name__ == "__main__":
    a = torch.randn(4, 256)
    model = Seq2Seq2(input_channels=1)
    b = model(a)
    print(b.shape)  # 应该输出 torch.Size([4, 256])
