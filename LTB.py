import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
from simam import simam_module


class MASA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MASA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1, groups=channels * 4, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pool_gate_h = nn.AdaptiveAvgPool2d((8, 8))
        self.simam = simam_module()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v, a = self.qkv_conv(self.qkv(x)).chunk(4, dim=1)
        agent = self.pool_gate_h(a)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        agent = rearrange(agent, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        q_attn = self.softmax((q @ agent.transpose(-2, -1)) * self.temperature)
        k_attn = self.softmax((agent @ k.transpose(-2, -1)) * self.temperature)
        agent_v = k_attn @ v
        out_attn = q_attn @ agent_v
        out = rearrange(out_attn, 'b head (h w) c  -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.simam(out)


class GFFN(nn.Module):
    def __init__(self, channels, expansion_factor=2):
        super(GFFN, self).__init__()
        hidden_features = expansion_factor * channels
        self.project_in = nn.Conv2d(channels, hidden_features, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=False)
        self.project_out = nn.Conv2d(hidden_features, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # gelu 相当于 relu+dropout
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.MASA = MASA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.GFFN = GFFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.MASA(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.GFFN(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        return x
