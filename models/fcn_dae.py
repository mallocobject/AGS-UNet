import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import CNNBlock1D, CNNTransBlock1D


class FCNDAE1D(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            CNNBlock1D(1, 40, kernel_size=16, stride=2),
            CNNBlock1D(40, 20, kernel_size=16, stride=2),
            CNNBlock1D(20, 20, kernel_size=16, stride=2),
            CNNBlock1D(20, 20, kernel_size=16, stride=2),
            CNNBlock1D(20, 40, kernel_size=16, stride=2),
            CNNBlock1D(40, 1, kernel_size=16, stride=1),
        )

        # 解码器
        self.decoder = nn.Sequential(
            CNNTransBlock1D(1, 1, kernel_size=16, stride=1),
            CNNTransBlock1D(1, 40, kernel_size=16, stride=2, output_padding=1),
            CNNTransBlock1D(40, 20, kernel_size=16, stride=2, output_padding=1),
            CNNTransBlock1D(20, 20, kernel_size=16, stride=2, output_padding=1),
            CNNTransBlock1D(20, 20, kernel_size=16, stride=2, output_padding=1),
            CNNTransBlock1D(20, 40, kernel_size=16, stride=2, output_padding=1),
            nn.ConvTranspose1d(40, 1, kernel_size=16, stride=1, padding=8),
        )

    def forward(self, x):
        length = x.size(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度，变为 [B, 1, L]
        # x: [B, 1, L]
        z = self.encoder(x)
        out = self.decoder(z)
        return out.squeeze(1)[:, :length]  # 去掉通道维度，变回 [B, L]


# ========== 测试 ==========
if __name__ == "__main__":
    model = FCNDAE1D()
    x = torch.randn(4, 3600)  # batch=4, 1通道, 长度3600
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
