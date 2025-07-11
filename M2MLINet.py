"""
M2M-LINet
3.351G 6.554M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from pvtv2 import pvt_v2_b0
from LTB import TransformerBlock
from BAEM import BAEM
from LCB import LCB, ADConv
from simam import simam_module


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        # Transposed convolution for upsampling
        self.body = nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.body(x)


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        # Convolution for downsampling
        self.body = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.body(x)


class FeatureFusion(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_down = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_left = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_final = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn_final = norm_layer(out_channel)

    def forward(self, left, down):
        # Process down and left features
        down_mask = self.conv_down(down)
        left_mask = self.conv_left(left)

        # Upsample down_mask if necessary
        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        # Element-wise multiplication and ReLU activation
        fused = F.relu(down_mask * left_mask, inplace=True)
        return F.relu(self.bn_final(self.conv_final(fused)), inplace=True)


class M2MLINet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_heads=[4, 4, 4, 4], channels=[32, 64, 128, 256], num_refinement=1,
                 expansion_factor=2):
        super(M2MLINet, self).__init__()
        # Backbone network
        self.backbone = pvt_v2_b0()
        path = r'./pvt_v2_b0.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # Asymmetric double convolution
        self.ad_conv = ADConv(3, channels[0])

        # Lightweight convolution blocks
        lcb_encoder_channels = [64, 128, 288, 512]
        self.lcb_encoder = nn.ModuleList(
            [LCB(lcb_encoder_channels[i], channels[i]) for i in range(len(channels))])

        # Boundary Aware Enhancement Module (BAEM)
        self.refinement1 = BAEM(channels[3], channels[3])
        self.refinement2 = BAEM(channels[2], channels[2])
        self.refinement3 = BAEM(channels[1], channels[1])
        self.refinement4 = BAEM(channels[0], channels[0])

        # Feature fusion layers
        self.trans_layer1 = nn.Conv2d(6, 3, 1)
        self.trans_layer2 = nn.Conv2d(160, 32, 1)
        self.fusion1 = FeatureFusion(64, 32, 32)
        self.fusion2 = FeatureFusion(32, 32, 3)

        # Encoder layers
        self.ltb_encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                           zip(num_blocks, num_heads, channels)])

        # Downsampling layers
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        # Upsampling layers
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))])

        # Channel reduction layers
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(1, len(channels)))])

        # Decoder layers
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[0], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        # Output layers
        self.output_edge = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.simam = simam_module(256)
        self.prediction = nn.Conv2d(32, 1, kernel_size=1, bias=False)
        self.ups_out = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, bias=False)

    def forward(self, x1, x2):
        """
        Coarse Location Stage
        """

        # Extract features from CoarseNet
        x1_1, x1_2, x1_3, x1_4 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4 = self.backbone(x2)

        # Coarse localization
        out1_0 = self.fusion1(x1_2, self.trans_layer2(x1_3))
        out1_1 = self.fusion2(x1_1, out1_0)
        out2_0 = self.fusion1(x2_2, self.trans_layer2(x2_3))
        out2_1 = self.fusion2(x2_1, out2_0)

        coarse_diff = torch.abs(torch.sub(out1_1, out2_1))
        coarse_up = F.interpolate(coarse_diff, x1.size()[2:], mode='bilinear', align_corners=True)
        coarse_combined = x2 * coarse_up + x2
        coarse_final = self.trans_layer1(torch.cat([coarse_combined, x2], dim=1))

        # Asymmetric double convolution
        c0_1 = self.ad_conv(x1)
        c0_2 = self.ad_conv(coarse_final)

        """
        Fine Detail Focusing Stage
        """
        # Lightweight convolution
        c1_1 = self.lcb_encoder[0](torch.cat([c0_1, x1_1], dim=1))
        c2_1 = self.lcb_encoder[0](torch.cat([c0_2, x2_1], dim=1))

        # Edge attention
        edge_att = torch.sigmoid(self.output_edge(coarse_diff))

        # Encoder stage 1
        t_enc1_1 = self.ltb_encoders[0](c1_1)
        t_enc2_1 = self.ltb_encoders[0](c2_1)
        c1 = self.refinement4(t_enc1_1, t_enc2_1, edge_att)

        # Encoder stage 2
        t_down1_2 = self.downs[0](t_enc1_1 + c1)
        t_down2_2 = self.downs[0](t_enc2_1 + c1)
        c1_2 = self.lcb_encoder[1](torch.cat([t_down1_2, x1_2], dim=1))
        c2_2 = self.lcb_encoder[1](torch.cat([t_down2_2, x2_2], dim=1))
        t_enc1_2 = self.ltb_encoders[1](c1_2)
        t_enc2_2 = self.ltb_encoders[1](c2_2)
        c2 = self.refinement3(t_enc1_2, t_enc2_2, edge_att)

        # Encoder stage 3
        t_down1_3 = self.downs[1](t_enc1_2 + c2)
        t_down2_3 = self.downs[1](t_enc2_2 + c2)
        c1_3 = self.lcb_encoder[2](torch.cat([t_down1_3, x1_3], dim=1))
        c2_3 = self.lcb_encoder[2](torch.cat([t_down2_3, x2_3], dim=1))
        t_enc1_3 = self.ltb_encoders[2](c1_3)
        t_enc2_3 = self.ltb_encoders[2](c2_3)
        c3 = self.refinement2(t_enc1_3, t_enc2_3, edge_att)

        # Fusion module
        merge = self.reduces[0](self.simam(torch.cat([t_enc1_3, t_enc2_3], dim=1)))
        merge_down = self.downs[2](merge + c3)
        diff4 = torch.abs(torch.sub(x1_4, x2_4))
        c_merge = self.lcb_encoder[3](torch.cat([merge_down, diff4], dim=1))
        t_merge = self.ltb_encoders[3](c_merge)

        """
        Decoding Prediction Stage
        """
        # Decoder stage
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](t_merge), c3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), c2], dim=1)))
        out_dec1 = self.decoders[2](self.reduces[2](torch.cat([self.ups[2](out_dec2), c1], dim=1)))

        # Prediction
        out_dec = self.decoders[2](self.ups_out(out_dec1))
        final = F.interpolate(out_dec, x1.size()[2:], mode='bilinear', align_corners=True)
        out_x = self.prediction(final)
        e_out = self.output_edge(F.interpolate(coarse_diff, x1.size()[2:], mode='bilinear', align_corners=True))
        change_out = nn.Sigmoid()(out_x)
        edge_out = nn.Sigmoid()(e_out)

        return change_out, edge_out


if __name__ == '__main__':
    model = M2MLINet()
    print(model)
    x1 = torch.randn((1, 3, 256, 256))
    x2 = torch.randn((1, 3, 256, 256))
    x, _ = model(x1, x2)
    flops1, params1 = profile(model, (x1, x2))
    flops1, params1 = clever_format([flops1, params1], "%.3f")
    print(flops1, params1)
