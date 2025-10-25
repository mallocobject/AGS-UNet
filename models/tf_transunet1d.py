from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import layers


class TFTransUNet1D(nn.Module):
    def __init__(
        self,
        chin: int = 1,
        chout: int = 1,
        hidden: int = 64,
        depth: int = 3,
        nhead: int = 4,
        kernel_size: int = 3,
        stride: int = 2,
        max_len: int = 10000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth

        # 改进的初始卷积块 - 使用 BatchNorm
        self.initial_conv = nn.Sequential(
            nn.Conv1d(chin, hidden, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # 编码器 - 使用残差块
        self.encoder = nn.ModuleList()
        current_channels = hidden

        for i in range(depth):
            encoder_block = nn.Sequential(
                nn.Conv1d(
                    current_channels,
                    current_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(current_channels * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(
                    current_channels * 2,
                    current_channels * 2,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(current_channels * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.encoder.append(encoder_block)
            current_channels *= 2

        # Transformer 瓶颈层
        self.transformer = layers.TransBlock1D(
            in_dim=current_channels,
            embed_dim=current_channels,
            num_heads=nhead,
            num_layers=4,  # 增加层数
            seq_length=max_len // (stride**depth),
        )

        # 解码器 - 改进的上采样
        self.decoder = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i in range(depth):
            # 上采样层
            upsample = nn.Sequential(
                nn.ConvTranspose1d(
                    current_channels,
                    current_channels // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                ),
                nn.BatchNorm1d(current_channels // 2),
                nn.ReLU(inplace=True),
            )
            self.upsample_layers.append(upsample)

            # 解码块
            decoder_block = nn.Sequential(
                nn.Conv1d(
                    current_channels,  # 上采样后 + 跳跃连接
                    current_channels // 2,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(current_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(
                    current_channels // 2,
                    current_channels // 2,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(current_channels // 2),
                nn.ReLU(inplace=True),
            )
            self.decoder.append(decoder_block)
            current_channels //= 2

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(current_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(current_channels, chout, kernel_size=1),
            nn.Tanh(),  # 限制输出范围
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入处理
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        # 初始卷积
        x = self.initial_conv(x)

        # 编码路径 - 保存特征用于跳跃连接
        encoder_features = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_features.append(x)

        # Transformer 瓶颈
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (B, C, T)

        # 解码路径 - 使用跳跃连接
        for i in range(self.depth):
            # 上采样
            x = self.upsample_layers[i](x)

            # 跳跃连接
            if i < len(encoder_features):
                skip_feature = encoder_features[-(i + 1)]
                # 调整尺寸匹配
                if x.shape[-1] != skip_feature.shape[-1]:
                    x = F.interpolate(
                        x,
                        size=skip_feature.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )
                x = torch.cat([x, skip_feature], dim=1)

            # 解码块
            x = self.decoder[i](x)

        # 最终输出
        x = self.final_conv(x)
        return x.squeeze(1)


if __name__ == "__main__":
    model = TFTransUNet1D()
    print(model)
    x = torch.randn(100, 3600)
    y = model(x)
    print(y.shape)
