"""PVT-v2 B0 feature backbone without a timm runtime dependency.

The four feature scales, layer names and attention calculations follow upstream.
Unused classifier APIs and the B1--B5 architectures are intentionally omitted.
"""

import math
from functools import partial

import torch
from torch import nn


def _init_weights(module):
    """Use the initialization rules of the upstream PVT-v2 implementation."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DropPath(nn.Module):
    """Stochastic depth, dropping a residual branch independently per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class DWConv(nn.Module):
    """Depthwise spatial mixing for flattened image tokens."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, height, width):
        b, _, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, height, width)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    """Linear expansion, depthwise convolution and linear projection."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x, height, width):
        x = self.drop(self.act(self.dwconv(self.fc1(x), height, width)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    """Spatial reduction attention used by the B0 backbone."""

    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(b, c, height, width)
            reduced = self.sr(reduced).reshape(b, c, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
        else:
            reduced = x
        kv = self.kv(reduced).reshape(b, -1, 2, self.num_heads, c // self.num_heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """PVT-v2 transformer encoder block."""

    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, drop_path):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = Attention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class OverlapPatchEmbed(nn.Module):
    """Convolutional overlapping patch embedding with channel LayerNorm."""

    def __init__(self, in_channels, dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class PVTv2B0(nn.Module):
    """Return feature maps with 32, 64, 160, 256 channels at strides 4--32."""

    def __init__(self):
        super().__init__()
        dimensions = (32, 64, 160, 256)
        heads = (1, 2, 5, 8)
        mlp_ratios = (8, 8, 4, 4)
        sr_ratios = (8, 4, 2, 1)
        drop_rates = torch.linspace(0, 0.1, 8).tolist()
        in_channels = 3
        for stage, dim in enumerate(dimensions):
            index = stage + 1
            setattr(
                self,
                f"patch_embed{index}",
                OverlapPatchEmbed(
                    in_channels,
                    dim,
                    patch_size=7 if stage == 0 else 3,
                    stride=4 if stage == 0 else 2,
                ),
            )
            setattr(
                self,
                f"block{index}",
                nn.ModuleList(
                    [
                        Block(
                            dim,
                            heads[stage],
                            mlp_ratios[stage],
                            sr_ratios[stage],
                            drop_rates[2 * stage + i],
                        )
                        for i in range(2)
                    ]
                ),
            )
            setattr(self, f"norm{index}", nn.LayerNorm(dim, eps=1e-6))
            in_channels = dim
        self.apply(_init_weights)

    def forward(self, x):
        outputs = []
        b = x.shape[0]
        for index in range(1, 5):
            x, height, width = getattr(self, f"patch_embed{index}")(x)
            for block in getattr(self, f"block{index}"):
                x = block(x, height, width)
            x = getattr(self, f"norm{index}")(x)
            x = x.reshape(b, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs
