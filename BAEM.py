import torch
import torch.nn.functional as F
import torch.nn as nn
from simam import simam_module
from InvertedResidual import InvertedResidual


class FeatureDifference(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureDifference, self).__init__()
        self.conv = InvertedResidual(in_channel, out_channel, 1, 1)
        self.att = simam_module()

    def forward(self, x, y):
        x_att = self.att(x)
        y_att = self.att(y)
        return self.conv(torch.abs(x_att - y_att))


class FeatureAggregation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureAggregation, self).__init__()
        self.conv_out = InvertedResidual(2 * in_channel, out_channel, 1, 1)
        self.att = simam_module()

    def forward(self, x, y):
        return self.conv_out(torch.cat((x, y), dim=1))


class BoundaryEnhancement(nn.Module):
    def __init__(self, channel):
        super(BoundaryEnhancement, self).__init__()
        self.conv = InvertedResidual(channel, channel, 1, 1)

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=True)
        x = c * att + c
        return self.conv(x)


class BAEM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(BAEM, self).__init__()
        self.diff = FeatureDifference(dim_in, dim_out)
        self.aggr = FeatureAggregation(dim_in, dim_out)
        self.edge = BoundaryEnhancement(dim_out)

    def forward(self, x1, x2, edge):
        x_diff = self.diff(x1, x2)
        x_aggr = self.aggr(x1, x2)
        y = x_diff + x_aggr
        return self.edge(y, edge)
