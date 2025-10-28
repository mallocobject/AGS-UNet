import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import RMSNorm
from layers import AbsPositionalEncoding, DRSNBlock, DiffTransformerLayer


class LinearAtten(nn.Module):
    def __init__(self, in_dims, heads=4):
        super(LinearAtten, self).__init__()
        dim_head = in_dims // heads
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dims = dim_head * heads
        self.to_qkv = nn.Conv1d(in_dims, hidden_dims * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dims, in_dims, kernel_size=1), nn.GroupNorm(1, in_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b l c -> b c l")
        b, c, l = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) l -> b h c l", h=h), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h c j, b h d j -> b h c d", k, v)
        out = torch.einsum("b h c d, b h d i -> b h c i", context, q)
        out = rearrange(out, "b h c l -> b (h c) l")
        out = self.to_out(out)
        return rearrange(out, "b c l -> b l c")


# ========= Transformer 模块 =========
class LeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAtten(in_dims=dim, heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )

        self.rel_pos = AbsPositionalEncoding(dim, max_len=1000)

    def forward(self, x):
        x = rearrange(x, "b c l -> b l c")
        x = self.rel_pos(x)
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in)
        x = x + attn_out

        ff_in = self.norm2(x)
        ff_out = self.ff(ff_in)
        x = x + ff_out
        return rearrange(x, "b l c -> b c l")


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B L C
        B, L, C = x.shape

        # padding
        pad_input = L % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, 1))

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 2, 1)  # B C L
        return x


class PatchSeparate(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim // 2, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B L C
        B, L, C = x.shape
        x = rearrange(x, "b l (c1 c2) -> b (c1 l) c2", c1=2)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 2, 1)  # B C L
        return x


class UNetEncoder(nn.Module):
    """
    UNet编码器部分,提取多尺度特征
    """

    def __init__(self, input_channels, base_channels=8, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_channels = input_channels
        out_channels = base_channels

        for i in range(num_layers):
            # 每个编码层包含两个卷积
            layer = nn.Sequential(
                DRSNBlock(in_channels, out_channels),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.layers.append(layer)

            # 下采样（除了最后一层）
            if i < num_layers - 1:
                self.downsample.append(PatchMerging(dim=out_channels))
                # 更新下一层的输入通道数
                out_channels *= 2
                in_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, sequence_length)
        Returns:
            features: 各层的特征列表，从浅到深
            bottleneck: 最底层的特征
        """
        features = []
        current = x

        for i, layer in enumerate(self.layers):
            current = layer(current)
            features.append(current)

            if i < len(self.downsample):
                current = self.downsample[i](current)

        return features, current


class UNetDecoder(nn.Module):
    """
    UNet解码器部分,结合编码器特征进行上采样
    """

    def __init__(self, output_channels, base_channels=8, num_layers=4):
        super().__init__()
        self.upsample = nn.ModuleList()
        self.layers = nn.ModuleList()

        # 计算各层的通道数（从深到浅）
        channels = [base_channels * (2**i) for i in range(num_layers, -1, -1)]

        for i in range(num_layers):
            # 上采样 + 跳跃连接
            if i > 0:
                self.upsample.append(PatchSeparate(dim=channels[i]))

            layer = nn.Sequential(
                nn.Conv1d(channels[i + 1], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels[i + 1], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            )
            self.layers.append(layer)

        # 最后一层输出
        self.final_conv = nn.Conv1d(channels[-1], output_channels, kernel_size=1)

    def forward(self, features, bottleneck):
        """
        Args:
            features: 编码器各层特征（从浅到深）
            bottleneck: 最底层特征
        Returns:
            重建的输出
        """
        current = bottleneck

        # 反转特征列表，从深到浅
        skip_features = list(reversed(features))

        for i, layer in enumerate(self.layers):
            # 上采样
            if i > 0:
                current = self.upsample[i - 1](current)
            # 跳跃连接
            current = current + skip_features[i]
            current = layer(current)

        return self.final_conv(current)


class BottleneckDiffTransformer(nn.Module):
    """
    在UNet底层应用DiffTransformer
    """

    def __init__(
        self, bottleneck_channels, d_model, num_heads, num_layers, lambda_init=0.5
    ):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.d_model = d_model

        # 将卷积特征投影到Transformer维度
        self.input_proj = nn.Linear(bottleneck_channels, d_model)

        # DiffTransformer层
        self.transformer_layers = nn.ModuleList(
            [
                DiffTransformerLayer(
                    d_model, num_heads, 0.8 - 0.6 * math.exp(-0.3 * (l - 1))
                )
                for l in range(num_layers)
            ]
        )

        # 投影回卷积特征维度
        self.output_proj = nn.Linear(d_model, bottleneck_channels)

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, bottleneck_channels, sequence_length)
        Returns:
            transformed: (batch_size, bottleneck_channels, sequence_length)
        """
        batch_size, channels, seq_len = x.shape

        # 转换为序列格式: (batch_size, seq_len, channels)
        x_seq = x.permute(0, 2, 1)

        # 投影到Transformer维度
        x_proj = self.input_proj(x_seq)  # (batch_size, seq_len, d_model)

        # 应用Transformer层
        current = x_proj
        for layer in self.transformer_layers:
            current = layer(current)

        current = self.norm(current)

        # 投影回原始维度
        output_seq = self.output_proj(
            current
        )  # (batch_size, seq_len, bottleneck_channels)

        # 转换回卷积格式
        output = output_seq.permute(
            0, 2, 1
        )  # (batch_size, bottleneck_channels, seq_len)

        return output


class UNetWithDiffTransformer(nn.Module):
    """
    结合UNet和底层DiffTransformer的完整模型
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        base_channels=8,
        num_unet_layers=4,
        d_model=256,
        num_heads=4,
        num_transformer_layers=2,
    ):
        super().__init__()

        # UNet编码器
        self.encoder = UNetEncoder(input_channels, base_channels, num_unet_layers)

        # 计算底层通道数
        bottleneck_channels = base_channels * (2 ** (num_unet_layers - 1))

        # 底层DiffTransformer
        self.bottleneck_transformer = BottleneckDiffTransformer(
            bottleneck_channels=bottleneck_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            lambda_init=0.5,
        )

        # UNet解码器
        self.decoder = UNetDecoder(output_channels, base_channels, num_unet_layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_channels, sequence_length)
        Returns:
            output: (batch_size, output_channels, sequence_length)
        """
        # 编码器提取多尺度特征
        features, bottleneck = self.encoder(x)

        # 在底层应用Transformer
        transformed_bottleneck = self.bottleneck_transformer(bottleneck)

        # 解码器重建
        output = self.decoder(features, transformed_bottleneck)

        return output


class DTUNet(nn.Module):
    """
    专门用于时间序列去噪的模型
    """

    def __init__(
        self,
        input_channels=2,
        output_channels=2,
        base_channels=8,
        num_unet_layers=4,
        d_model=64,
        num_heads=4,
        num_transformer_layers=2,
    ):
        super().__init__()

        self.unet_transformer = UNetWithDiffTransformer(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            num_unet_layers=num_unet_layers,
            d_model=d_model,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
        )

    def forward(self, noisy_signal):
        """
        Args:
            noisy_signal: 带噪声的信号 (batch_size, channels, sequence_length)
        Returns:
            denoised_signal: 去噪后的信号 (batch_size, channels, sequence_length)
        """
        return self.unet_transformer(noisy_signal)


if __name__ == "__main__":
    model = DTUNet()
    print(model)

    x = torch.randn(32, 2, 256)

    y = model(x)
    print(y.shape)
