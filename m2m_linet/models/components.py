"""Shared convolution, attention and boundary enhancement building blocks.

Layer attribute names follow the upstream implementation for checkpoint loading.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SimAM(nn.Module):
    """Parameter-free attention over each channel's spatial distribution."""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-2] * x.shape[-1] - 1
        if n < 1:
            raise ValueError("SimAM requires at least two spatial positions.")
        squared = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        energy = squared / (4 * (squared.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda))
        return x * torch.sigmoid(energy + 0.5)


class ConvBNReLU(nn.Sequential):
    """Convolution, batch normalization and ReLU6 as used by MobileNet blocks."""

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                (kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Depthwise separable convolution with an optional identity shortcut."""

    def __init__(self, in_channel, out_channel, stride=1, expand_ratio=1):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
                nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_shortcut else out


class Double13(nn.Module):
    """Parallel horizontal and vertical asymmetric convolutions."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_1x3(x) + self.conv_3x1(x)


class ADConv(nn.Module):
    """Prepare the RGB input at one quarter of its original resolution."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 7, stride=4, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            Double13(out_channel, out_channel),
            SimAM(),
        )

    def forward(self, x):
        return self.conv(x)


class Scale(nn.Module):
    """Learnable scalar multiplying a residual branch."""

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([float(init_value)]))

    def forward(self, x):
        return x * self.scale


class RE(nn.Module):
    """Residual enhancement before high/low frequency decomposition."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = SimAM()
        self.conv2 = SimAM()
        self.relu = nn.PReLU(channels)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x):
        return self.weight1(x) + self.weight2(self.conv2(self.relu(self.conv1(x))))


class LCBBlock(nn.Module):
    """Process high and low spatial frequencies with lightweight convolutions."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.decoder_high = InvertedResidual(in_channel, out_channel)
        self.decoder_low = InvertedResidual(in_channel, out_channel)
        self.alise = InvertedResidual(out_channel * 2, out_channel)
        self.decoder_out = InvertedResidual(in_channel, out_channel)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = SimAM()

    def forward(self, x):
        low_input = self.down(x)
        high = x - F.interpolate(low_input, size=x.shape[-2:], mode="bilinear", align_corners=True)
        high = self.decoder_high(high)
        low = self.decoder_low(low_input)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=True)
        return self.alise(self.att(torch.cat([low, high], dim=1))) + self.decoder_out(x)


class LCB(nn.Module):
    """Lightweight convolution block used at each encoder scale."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.encoder = RE(in_channel)
        self.lcb_block = LCBBlock(in_channel, out_channel)

    def forward(self, x):
        return self.lcb_block(self.encoder(x))


class FeatureDifference(nn.Module):
    """Attend to the two dates before computing absolute feature differences."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = InvertedResidual(in_channel, out_channel)
        self.att = SimAM()

    def forward(self, x, y):
        return self.conv(torch.abs(self.att(x) - self.att(y)))


class FeatureAggregation(nn.Module):
    """Learn a joint representation of the two dates."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_out = InvertedResidual(2 * in_channel, out_channel)

    def forward(self, x, y):
        return self.conv_out(torch.cat([x, y], dim=1))


class BoundaryEnhancement(nn.Module):
    """Apply the coarse boundary attention with a residual connection."""

    def __init__(self, channel):
        super().__init__()
        self.conv = InvertedResidual(channel, channel)

    def forward(self, features, attention):
        if features.shape[-2:] != attention.shape[-2:]:
            attention = F.interpolate(
                attention,
                features.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        return self.conv(features * attention + features)


class BAEM(nn.Module):
    """Boundary-aware enhancement of difference and aggregation features."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.diff = FeatureDifference(dim_in, dim_out)
        self.aggr = FeatureAggregation(dim_in, dim_out)
        self.edge = BoundaryEnhancement(dim_out)

    def forward(self, x1, x2, edge):
        return self.edge(self.diff(x1, x2) + self.aggr(x1, x2), edge)
