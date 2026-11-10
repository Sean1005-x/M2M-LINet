"""M2M-LINet with the official interleaved convolution/transformer structure."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .components import ADConv, BAEM, LCB, SimAM
from .pvtv2 import PVTv2B0
from .transformer import TransformerBlock


class UpSample(nn.Module):
    """Double the spatial resolution and halve the channel count."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.ConvTranspose2d(channels, channels // 2, 2, stride=2, bias=False)

    def forward(self, x):
        return self.body(x)


class DownSample(nn.Module):
    """Halve the spatial resolution and double the channel count."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Conv2d(channels, channels * 2, 2, stride=2)

    def forward(self, x):
        return self.body(x)


class FeatureFusion(nn.Module):
    """Fuse two backbone scales for coarse localization."""

    def __init__(self, in_channel_left, in_channel_down, out_channel):
        super().__init__()
        self.conv_down = nn.Conv2d(in_channel_down, out_channel, 3, padding=1)
        self.conv_left = nn.Conv2d(in_channel_left, out_channel, 3, padding=1)
        self.conv_final = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn_final = nn.BatchNorm2d(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_down(down)
        left_mask = self.conv_left(left)
        if down_mask.shape[-2:] != left.shape[-2:]:
            down_mask = F.interpolate(
                down_mask, size=left.shape[-2:], mode="bilinear", align_corners=False
            )
        fused = F.relu(down_mask * left_mask, inplace=True)
        return F.relu(self.bn_final(self.conv_final(fused)), inplace=True)


class M2MLINet(nn.Module):
    """Detect change in paired RGB images and return change/edge logits.

    Inputs must share shape ``(N, 3, H, W)``; H and W must be multiples of
    32 and at least 64. Both outputs have shape ``(N, 1, H, W)``. Apply
    ``sigmoid`` for probabilities, or use logits directly with a stable loss.
    No weights are loaded or downloaded unless ``backbone_weights`` is provided.
    """

    def __init__(self, backbone_weights: Optional[Path] = None):
        super().__init__()
        channels = (32, 64, 128, 256)
        self.backbone = PVTv2B0()
        self.ad_conv = ADConv(3, channels[0])
        self.lcb_encoder = nn.ModuleList(
            [
                LCB(in_channel, out_channel)
                for in_channel, out_channel in zip((64, 128, 288, 512), channels)
            ]
        )
        self.refinement2 = BAEM(128, 128)
        self.refinement3 = BAEM(64, 64)
        self.refinement4 = BAEM(32, 32)
        self.trans_layer1 = nn.Conv2d(6, 3, 1)
        self.trans_layer2 = nn.Conv2d(160, 32, 1)
        self.fusion1 = FeatureFusion(64, 32, 32)
        self.fusion2 = FeatureFusion(32, 32, 3)
        self.ltb_encoders = nn.ModuleList(
            [
                nn.Sequential(TransformerBlock(channel), TransformerBlock(channel))
                for channel in channels
            ]
        )
        self.downs = nn.ModuleList([DownSample(channel) for channel in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(channel) for channel in reversed(channels[1:])])
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], 1, bias=False)
                for i in reversed(range(1, len(channels)))
            ]
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(TransformerBlock(channel), TransformerBlock(channel))
                for channel in reversed(channels[:-1])
            ]
        )
        self.output_edge = nn.Conv2d(3, 1, 1, bias=False)
        self.simam = SimAM()
        self.prediction = nn.Conv2d(32, 1, 1, bias=False)
        self.ups_out = nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False)
        if backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

    def load_backbone_weights(self, path: Path) -> None:
        """Load a complete PVT-v2 B0 state; only classifier head keys may differ.

        This method expects a raw backbone state dictionary or a dictionary
        containing ``state_dict``. Missing, unexpected or misshaped feature
        weights raise an error before any parameter is changed.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Backbone checkpoint must be a dictionary.")
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
            raise ValueError("Backbone state_dict must have string keys.")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        state = {
            key: value for key, value in state.items() if key not in {"head.weight", "head.bias"}
        }
        expected = self.backbone.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected) & set(state)
            if not isinstance(state[key], torch.Tensor) or state[key].shape != expected[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"Incompatible PVT-v2 B0 weights: missing={missing}; "
                f"unexpected={unexpected}; shape_mismatch={mismatched}"
            )
        self.backbone.load_state_dict(state, strict=True)

    @staticmethod
    def _validate_inputs(x1, x2):
        if x1.ndim != 4 or x2.ndim != 4 or x1.shape != x2.shape:
            raise ValueError("Paired inputs must share the same NCHW shape.")
        if x1.shape[0] < 1 or x1.shape[1] != 3:
            raise ValueError("Inputs must contain a nonempty batch of RGB images.")
        if any(size < 64 or size % 32 != 0 for size in x1.shape[-2:]):
            raise ValueError("Input height and width must be multiples of 32 and at least 64.")
        if not x1.is_floating_point() or not x2.is_floating_point():
            raise ValueError("Model inputs must be floating point tensors.")
        if x1.device != x2.device or x1.dtype != x2.dtype:
            raise ValueError("Paired inputs must have the same device and dtype.")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(x1, x2)
        x1_1, x1_2, x1_3, x1_4 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4 = self.backbone(x2)

        out1 = self.fusion2(x1_1, self.fusion1(x1_2, self.trans_layer2(x1_3)))
        out2 = self.fusion2(x2_1, self.fusion1(x2_2, self.trans_layer2(x2_3)))
        coarse_diff = torch.abs(out1 - out2)
        coarse_up = F.interpolate(coarse_diff, x1.shape[-2:], mode="bilinear", align_corners=True)
        coarse_final = self.trans_layer1(torch.cat([x2 * coarse_up + x2, x2], dim=1))
        c0_1, c0_2 = self.ad_conv(x1), self.ad_conv(coarse_final)
        edge_att = torch.sigmoid(self.output_edge(coarse_diff))

        t1 = self.ltb_encoders[0](self.lcb_encoder[0](torch.cat([c0_1, x1_1], dim=1)))
        t2 = self.ltb_encoders[0](self.lcb_encoder[0](torch.cat([c0_2, x2_1], dim=1)))
        c1 = self.refinement4(t1, t2, edge_att)
        t1 = self.ltb_encoders[1](
            self.lcb_encoder[1](torch.cat([self.downs[0](t1 + c1), x1_2], dim=1))
        )
        t2 = self.ltb_encoders[1](
            self.lcb_encoder[1](torch.cat([self.downs[0](t2 + c1), x2_2], dim=1))
        )
        c2 = self.refinement3(t1, t2, edge_att)
        t1 = self.ltb_encoders[2](
            self.lcb_encoder[2](torch.cat([self.downs[1](t1 + c2), x1_3], dim=1))
        )
        t2 = self.ltb_encoders[2](
            self.lcb_encoder[2](torch.cat([self.downs[1](t2 + c2), x2_3], dim=1))
        )
        c3 = self.refinement2(t1, t2, edge_att)

        merge = self.reduces[0](self.simam(torch.cat([t1, t2], dim=1)))
        merge_down = self.downs[2](merge + c3)
        c_merge = self.lcb_encoder[3](torch.cat([merge_down, torch.abs(x1_4 - x2_4)], dim=1))
        t_merge = self.ltb_encoders[3](c_merge)

        out = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](t_merge), c3], dim=1)))
        out = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out), c2], dim=1)))
        out = self.decoders[2](self.reduces[2](torch.cat([self.ups[2](out), c1], dim=1)))
        out = self.decoders[2](self.ups_out(out))
        final = F.interpolate(out, x1.shape[-2:], mode="bilinear", align_corners=True)
        change_logits = self.prediction(final)
        edge_logits = self.output_edge(coarse_up)
        return change_logits, edge_logits
