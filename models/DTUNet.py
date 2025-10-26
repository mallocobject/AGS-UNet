import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import RMSNorm
from layers import AbsPositionalEncoding, DRSNBlock


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
        # self.attn = nn.attention(embed_dim=dim, num_heads=num_heads, batch_first=True)
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

    # ========= Transformer 模块 =========


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x):
        x = rearrange(x, "b c l -> b l c")
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out

        ff_in = self.norm2(x)
        ff_out = self.ff(ff_in)
        x = x + ff_out
        return rearrange(x, "b l c -> b c l")


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class MultiheadDuffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.size()

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.contiguous().view(bsz, seq_len, self.num_heads * 2, self.head_dim)
        k = k.contiguous().view(bsz, seq_len, self.num_kv_heads * 2, self.head_dim)
        v = v.contiguous().view(bsz, seq_len, self.num_kv_heads, 2 * self.head_dim)

        # cos, sin = rel_pos
        # q = apply_rotary_emb(q, cos, sin, interleaved=True)
        # k = apply_rotary_emb(k, cos, sin, interleaved=True)

        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=attn_weights.device), 1
        )

        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1).type_as(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_ = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.contiguous().view(
            bsz, self.num_heads, 2, seq_len, seq_len
        )
        attn_weights = attn_weights[:, :, 0] - lambda_ * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.num_heads * 2 * self.head_dim)
        )

        attn = self.out_proj(attn)
        attn = self.dropout(attn)
        return attn


class DiffTransformerBlock(nn.Module):
    def __init__(self, dim, depth=0, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadDuffAttn(
            embed_dim=dim, depth=depth, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )

        self.rel_pos = AbsPositionalEncoding(dim, max_len=1000)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
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

        return x


class PatchSeparate(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim // 2, dim // 2, bias=False)  # 输入是 dim//2
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        B, L, C = x.shape
        x = rearrange(x, "b l (c1 c2) -> b (c1 l) c2", c1=2)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class EncBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, padding=1, stride=1, use_relu=True):
        super(EncBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x


class DecBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, padding=1, use_relu=True):
        super(DecBlock, self).__init__()
        self.use_relu = use_relu

        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(in_channels)
        if use_relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            x = self.relu(self.bn(x))
        else:
            x = self.bn(x)
        return x


class DTUNet(nn.Module):
    def __init__(self, in_channels=1) -> None:
        super().__init__()

        channels = [2 ** (i + 3) for i in range(5)]
        heads = [2 ** (i + 1) for i in range(5)]

        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels[0]),
        )

        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i in range(4):
            self.encoder.append(
                # i
                nn.Sequential(
                    EncBlock(
                        in_channels=channels[i],
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        use_relu=True,
                    ),
                    DRSNBlock(
                        chin=channels[i],
                        chout=channels[i],
                        stride=1,
                    ),
                ),
            )

            self.downsamplers.append(PatchMerging(dim=channels[i]))

        for i in range(4):
            use_relu = True if i < 3 else False
            self.decoder.append(
                # 3 - i
                nn.Sequential(
                    DecBlock(
                        in_channels=channels[3 - i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        use_relu=use_relu,
                    ),
                    DecBlock(
                        in_channels=channels[3 - i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        use_relu=use_relu,
                    ),
                )
            )
            self.upsamplers.append(PatchSeparate(dim=channels[4 - i]))

        self.bottleneck = DiffTransformerBlock(
            dim=channels[-1], num_heads=heads[-1], ff_mult=4
        )

        self.final_conv = nn.Conv1d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        x = self.in_conv(x)
        skips = []
        for enc, down in zip(self.encoder, self.downsamplers):
            x = enc(x)
            skips.append(x)
            x = rearrange(x, "b c l -> b l c")
            x = down(x)
            x = rearrange(x, "b l c -> b c l")

        x = self.bottleneck(x)

        for dec, up in zip(self.decoder, self.upsamplers):
            x = rearrange(x, "b c l -> b l c")
            x = up(x)
            x = rearrange(x, "b l c -> b c l")
            skip = skips.pop()
            x = x + skip
            x = dec(x)

        x = self.final_conv(x)
        return x.squeeze(1)


if __name__ == "__main__":
    model = DTUNet()
    x = torch.randn(2, 256)
    y = model(x)
    print(y.shape)
