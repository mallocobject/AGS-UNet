import torch
import torch.nn as nn
from .pos_encoding import AbsPositionalEncoding


class TransBlock1D(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads, num_layers, seq_length):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Linear(in_dim, embed_dim)

        # 位置编码
        self.pos_encoding = AbsPositionalEncoding(embed_dim, seq_length)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # 通常设置为 embed_dim 的 4 倍
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # 输入形状: [batch, seq, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出投影
        self.output_proj = nn.Linear(embed_dim, in_dim)

        # 可选：层归一化
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, in_dim]
        Returns:
            output: [batch_size, seq_len, in_dim]
        """
        # 保存输入用于残差连接
        residual = x

        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, embed_dim]

        # 添加位置编码
        x = self.pos_encoding(x)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)  # [batch_size, seq_len, embed_dim]

        # 输出投影
        x = self.output_proj(x)  # [batch_size, seq_len, in_dim]

        # 残差连接和归一化
        x = x + residual
        x = self.norm(x)

        return x
