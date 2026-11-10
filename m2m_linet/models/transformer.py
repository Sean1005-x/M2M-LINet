"""Lightweight transformer from the official M2M-LINet implementation."""

import torch
from torch import nn
from torch.nn import functional as F

from .components import SimAM


class MASA(nn.Module):
    """Agent attention between image tokens and a pooled 8 by 8 agent grid."""

    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 4,
            channels * 4,
            kernel_size=3,
            padding=1,
            groups=channels * 4,
            bias=False,
        )
        self.pool_gate_h = nn.AdaptiveAvgPool2d((8, 8))
        self.simam = SimAM()
        self.softmax = nn.Softmax(dim=-1)

    def _split_heads(self, x):
        b, c, _, _ = x.shape
        return x.reshape(b, self.num_heads, c // self.num_heads, -1).transpose(-2, -1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v, a = self.qkv_conv(self.qkv(x)).chunk(4, dim=1)
        q, k, v = (self._split_heads(value) for value in (q, k, v))
        agent = self._split_heads(self.pool_gate_h(a))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        q_attn = self.softmax((q @ agent.transpose(-2, -1)) * self.temperature)
        k_attn = self.softmax((agent @ k.transpose(-2, -1)) * self.temperature)
        out = q_attn @ (k_attn @ v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return self.simam(out)


class GFFN(nn.Module):
    """Gated feed-forward network with depthwise spatial mixing."""

    def __init__(self, channels, expansion_factor=2):
        super().__init__()
        hidden_features = expansion_factor * channels
        self.project_in = nn.Conv2d(channels, hidden_features, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_features, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """Channel LayerNorm followed by residual MASA and GFFN branches."""

    def __init__(self, channels, num_heads=4, expansion_factor=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.MASA = MASA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.GFFN = GFFN(channels, expansion_factor)

    @staticmethod
    def _normalize(x, norm):
        b, c, h, w = x.shape
        tokens = norm(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
        return tokens.transpose(-2, -1).contiguous().reshape(b, c, h, w)

    def forward(self, x):
        x = x + self.MASA(self._normalize(x, self.norm1))
        return x + self.GFFN(self._normalize(x, self.norm2))
