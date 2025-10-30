import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate1D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: 门控信号的通道数 (来自解码器)
            F_l: 跳跃连接信号的通道数 (来自编码器)
            F_int: 中间层的通道数
        """
        super().__init__()

        # 1D卷积替代2D卷积
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: 来自解码器的门控信号 (batch_size, F_g, length)
            x: 来自编码器的跳跃连接特征 (batch_size, F_l, length)
        """
        # 1. 对门控信号和跳跃连接特征进行线性变换
        g1 = self.W_g(g)  # (batch_size, F_int, length)
        x1 = self.W_x(x)  # (batch_size, F_int, length)

        # 2. 相加并通过ReLU激活
        psi = self.relu(g1 + x1)  # (batch_size, F_int, length)

        # 3. 通过sigmoid得到注意力权重 (0-1之间)
        psi = self.psi(psi)  # (batch_size, 1, length)

        # 4. 应用注意力权重到原始跳跃连接特征
        return x * psi  # (batch_size, F_l, length)
