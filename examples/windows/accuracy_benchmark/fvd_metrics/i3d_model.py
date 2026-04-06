"""
Inception-v1 Inflated 3D ConvNet (I3D) — feature extractor for FVD.

Supports two weight formats:
  1. piergiaj/pytorch-i3d  ``rgb_imagenet.pt``  (PascalCase keys, ``bn``)
  2. TorchScript archive   ``i3d_pretrained_400.pt`` (lowercase keys, ``batch3d``)

Original paper: Carreira & Zisserman, "Quo Vadis, Action Recognition?", CVPR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit3D(nn.Module):
    """Conv3d + BatchNorm + ReLU with fixed padding."""

    def __init__(self, in_channels, out_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, use_bias=False,
                 use_bn=True, activation=True):
        super().__init__()
        self._use_bn = use_bn
        self._activation = activation
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_shape,
            stride=stride, padding=padding, bias=use_bias,
        )
        if use_bn:
            self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.conv3d(x)
        if self._use_bn:
            x = self.bn(x)
        if self._activation:
            x = F.relu(x, inplace=True)
        return x


class InceptionModule(nn.Module):
    """3D Inception block with 4 parallel branches."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b0 = Unit3D(in_channels, out_channels[0], (1, 1, 1))
        self.b1a = Unit3D(in_channels, out_channels[1], (1, 1, 1))
        self.b1b = Unit3D(out_channels[1], out_channels[2], (3, 3, 3), padding=1)
        self.b2a = Unit3D(in_channels, out_channels[3], (1, 1, 1))
        self.b2b = Unit3D(out_channels[3], out_channels[4], (3, 3, 3), padding=1)
        self.b3a = nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1), padding=1)
        self.b3b = Unit3D(in_channels, out_channels[5], (1, 1, 1))

    def forward(self, x):
        return torch.cat([
            self.b0(x),
            self.b1b(self.b1a(x)),
            self.b2b(self.b2a(x)),
            self.b3b(self.b3a(x)),
        ], dim=1)


class InceptionI3d(nn.Module):
    """I3D RGB model — returns 1024-dim pooled features (no classification head).

    Expects input (B, 3, T, H, W) with pixel values in [-1, 1].
    """

    def __init__(self):
        super().__init__()
        self.Conv3d_1a_7x7 = Unit3D(3, 64, (7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.MaxPool3d_2a_3x3 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv3d_2b_1x1 = Unit3D(64, 64, (1, 1, 1))
        self.Conv3d_2c_3x3 = Unit3D(64, 192, (3, 3, 3), padding=1)
        self.MaxPool3d_3a_3x3 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.MaxPool3d_4a_3x3 = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.MaxPool3d_5a_2x2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        """Returns (B, 1024) pooled features."""
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.avg_pool(x)
        return x.flatten(1)


def _map_torchscript_keys(ts_state: dict) -> dict:
    """Convert TorchScript-format keys to piergiaj/pytorch-i3d naming."""
    LAYER_MAP = {
        "conv3d_1a_7x7": "Conv3d_1a_7x7",
        "conv3d_2b_1x1": "Conv3d_2b_1x1",
        "conv3d_2c_3x3": "Conv3d_2c_3x3",
    }
    for i in "bcdef":
        LAYER_MAP[f"mixed_3{'bc'.index(i) + 98 == ord(i) and i or ''}"] = None
    LAYER_MAP.update({
        f"mixed_{s}": f"Mixed_{s}"
        for s in ["3b", "3c", "4b", "4c", "4d", "4e", "4f", "5b", "5c"]
    })

    BRANCH_MAP = {
        "branch_0": "b0",
        "branch_1.0": "b1a", "branch_1.1": "b1b",
        "branch_2.0": "b2a", "branch_2.1": "b2b",
        "branch_3.1": "b3b",
    }

    new_state = {}
    for key, val in ts_state.items():
        if key.startswith("conv3d_0c_1x1"):
            continue

        new_key = key
        new_key = new_key.replace(".batch3d.", ".bn.")

        for old_prefix, new_prefix in LAYER_MAP.items():
            if new_key.startswith(old_prefix + "."):
                new_key = new_prefix + new_key[len(old_prefix):]
                break

        for old_branch, new_branch in BRANCH_MAP.items():
            old_dot = "." + old_branch + "."
            if old_dot in new_key:
                new_key = new_key.replace(old_dot, "." + new_branch + ".")
                break

        new_state[new_key] = val
    return new_state


def load_i3d(weights_path: str, device: torch.device) -> InceptionI3d:
    """Load I3D weights from either rgb_imagenet.pt or TorchScript archive."""
    model = InceptionI3d()

    try:
        jit_model = torch.jit.load(weights_path, map_location="cpu")
        raw_state = jit_model.state_dict()
        state = _map_torchscript_keys(raw_state)
    except Exception:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)

    missing, unexpected = model.load_state_dict(state, strict=False)
    non_head = [k for k in missing if "logits" not in k]
    if non_head:
        import warnings
        warnings.warn(f"Missing keys in I3D checkpoint: {non_head}")

    model.eval().to(device)
    return model
