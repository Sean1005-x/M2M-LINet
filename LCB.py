import torch
import torch.nn.functional as F
from simam import simam_module
from InvertedResidual import *


class ADConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ADConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, stride=4, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Double13(out_ch, out_ch),
            simam_module()
        )

    def forward(self, input):
        return self.conv(input)


class Double13(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double13, self).__init__()
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x1 = self.conv_1x3(input)
        x2 = self.conv_3x1(input)
        return x1 + x2


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class RE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RE, self).__init__()
        self.conv1 = simam_module()
        self.conv2 = simam_module()
        self.relu = nn.PReLU(in_channel)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x):
        return self.weight1(x) + self.weight2(self.conv2(self.relu(self.conv1(x))))


class LCB_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LCB_Block, self).__init__()
        self.decoder_high = InvertedResidual(in_channel, out_channel, 1, 1)
        self.decoder_low = InvertedResidual(in_channel, out_channel, 1, 1)
        self.alise = InvertedResidual(out_channel * 2, out_channel, 1, 1)
        self.decoder_out = InvertedResidual(in_channel, out_channel, 1, 1)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = simam_module()

    def forward(self, x):
        x1 = self.down(x)
        high = x - F.interpolate(x1, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high = self.decoder_high(high)
        low = self.decoder_low(x1)
        low_up = F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        out = self.alise(self.att(torch.cat([low_up, high], dim=1))) + self.decoder_out(x)
        return out


class LCB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LCB, self).__init__()
        self.encoder = RE(in_channel, in_channel)
        self.lcb_block = LCB_Block(in_channel, out_channel)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.lcb_block(x1)
        return x2
