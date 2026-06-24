# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for NVFP4 Conv3d weight export (flattened-K logical schema).

These cover the conv export path added to ``unified_export_hf`` for diffusers
VAE convolutions (e.g. Wan 2.2 ``WanCausalConv3d``): routing, the exported
tensor schema, byte-exact packing against ``NVFP4QTensor.quantize``, a dequant
round-trip, quantizer-state hiding, and the logical (un-swizzled) scale layout.
All tests run on CPU.
"""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.diffusers_utils import (
    hide_quantizers_from_state_dict,
    pad_nvfp4_weights,
    swizzle_nvfp4_scales,
)
from modelopt.torch.export.layer_utils import is_quantconv3d
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_conv_weight,
    _postprocess_safetensors,
    _process_quantized_modules,
)
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import quantizer_attr_names

BLOCK_SIZE = 16

# Dynamic NVFP4 on every weight/input quantizer, mirroring the diffusers preset
# (modelopt_recipes/configs/numerics/nvfp4.yaml: e2m1 data, e4m3 block scale,
# block size 16, dynamic). Activation is dynamic -> no static input amax.
CONV_NVFP4_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: BLOCK_SIZE, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: BLOCK_SIZE, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
    ],
    "algorithm": "max",
}


class _Conv3dModel(nn.Module):
    def __init__(self, in_channels=16, out_channels=32, kernel_size=(3, 3, 3), groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, groups=groups, bias=bias).to(
            torch.float32
        )

    def forward(self, x):
        return self.conv(x)


def _quantize_conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), groups=1, bias=True):
    """Build and dynamic-NVFP4-quantize a tiny Conv3d model on CPU."""
    model = _Conv3dModel(in_channels, out_channels, kernel_size, groups, bias)

    def forward_loop(m):
        m(torch.randn(1, in_channels, 4, 8, 8))

    mtq.quantize(model, CONV_NVFP4_CFG, forward_loop)
    return model


def _flatten_pad(weight: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    out_channels = weight.shape[0]
    w_flat = weight.reshape(out_channels, -1)
    k_flat = w_flat.shape[-1]
    k_pad = ((k_flat + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if k_pad != k_flat:
        w_flat = torch.nn.functional.pad(w_flat, (0, k_pad - k_flat))
    return w_flat, k_flat, k_pad


# ---------------------------------------------------------------------------
# Routing predicate
# ---------------------------------------------------------------------------


def test_is_quantconv3d_matches_only_conv3d():
    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d = nn.Conv3d(16, 32, 3)
            self.conv2d = nn.Conv2d(16, 32, 3)
            self.convt3d = nn.ConvTranspose3d(16, 32, 3)
            self.linear = nn.Linear(16, 32)

        def forward(self, x):
            return self.conv3d(x)

    model = _Mixed()
    mtq.quantize(model, CONV_NVFP4_CFG, lambda m: m(torch.randn(1, 16, 4, 8, 8)))

    assert is_quantconv3d(model.conv3d)
    assert not is_quantconv3d(model.conv2d)
    assert not is_quantconv3d(model.convt3d)
    assert not is_quantconv3d(model.linear)


# ---------------------------------------------------------------------------
# Exported tensor schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("in_channels", "kernel_size", "expected_k_pad"),
    [
        (16, (3, 3, 3), 432),  # K_flat = 16*27 = 432 (already a multiple of 16)
        (8, (3, 3, 3), 224),  # K_flat = 8*27 = 216 -> padded up to 224
        (4, (1, 1, 1), 16),  # K_flat = 4 -> padded up to 16
    ],
)
def test_export_conv3d_schema(in_channels, kernel_size, expected_k_pad):
    out_channels = 32
    model = _quantize_conv3d(in_channels, out_channels, kernel_size)
    qa = quantizer_attr_names("weight")

    _export_quantized_conv_weight(model.conv, torch.float32)

    weight = model.conv.weight
    weight_scale = getattr(model.conv, qa.weight_scale)
    weight_scale_2 = getattr(model.conv, qa.weight_scale_2)

    assert weight.dtype == torch.uint8
    assert weight.shape == (out_channels, expected_k_pad // 2)
    assert weight_scale.dtype == torch.float8_e4m3fn
    assert weight_scale.shape == (out_channels, expected_k_pad // BLOCK_SIZE)
    assert weight_scale_2.dtype == torch.float32
    assert weight_scale_2.numel() == 1
    # The activation amax is calibrated (MaxCalibrator), so a static input scale
    # is exported as a float32 scalar -- same convention as NVFP4 Linear.
    assert hasattr(model.conv, qa.input_scale)
    input_scale = getattr(model.conv, qa.input_scale)
    assert input_scale.dtype == torch.float32
    assert input_scale.numel() == 1


def test_static_nvfp4_conv3d_is_rejected():
    """Static NVFP4 Conv3d export is out of scope (dynamic only) and must fail loudly."""
    model = _quantize_conv3d(16, 32, (3, 3, 3))
    # A global amax makes the weight quantizer look static to the exporter.
    model.conv.weight_quantizer._global_amax = torch.tensor(1.0)
    with pytest.raises(NotImplementedError, match="Static NVFP4 Conv3d"):
        _export_quantized_conv_weight(model.conv, torch.float32)


def test_export_conv3d_byte_exact_and_roundtrip():
    model = _quantize_conv3d(16, 32, (3, 3, 3))
    orig_weight = model.conv.weight.detach().clone().to(torch.float32)
    qa = quantizer_attr_names("weight")

    _export_quantized_conv_weight(model.conv, torch.float32)
    packed = model.conv.weight
    weight_scale = getattr(model.conv, qa.weight_scale)
    weight_scale_2 = getattr(model.conv, qa.weight_scale_2)

    # Byte-exact: the packed bytes must equal a direct quantize() of the same
    # flattened, padded weight (this also pins the low/high nibble order).
    w_flat, k_flat, k_pad = _flatten_pad(orig_weight)
    reference = NVFP4QTensor.quantize(w_flat, BLOCK_SIZE)[0]._quantized_data
    assert torch.equal(packed, reference)

    # Dequant round-trip reconstructs the original weight within FP4 tolerance.
    qtensor = NVFP4QTensor(w_flat.shape, torch.float32, packed)
    dequant = qtensor.dequantize(
        scale=weight_scale,
        double_scale=weight_scale_2,
        block_sizes={-1: BLOCK_SIZE},
        dtype=torch.float32,
    )
    assert torch.isfinite(dequant).all()
    # Reshape back to the conv filter shape on the unpadded reduction dim.
    dequant_unpad = dequant[:, :k_flat].reshape(orig_weight.shape)
    assert dequant_unpad.shape == orig_weight.shape
    cos = torch.cosine_similarity(
        dequant[:, :k_flat].reshape(-1),
        orig_weight.reshape(orig_weight.shape[0], -1).reshape(-1),
        dim=0,
    )
    assert cos > 0.9, f"round-trip cosine similarity too low: {cos.item()}"


def test_wrong_korder_breaks_byte_exact():
    """A transposed/permuted flatten must NOT reproduce the exported bytes."""
    model = _quantize_conv3d(16, 32, (3, 3, 3))
    orig_weight = model.conv.weight.detach().clone().to(torch.float32)

    _export_quantized_conv_weight(model.conv, torch.float32)
    packed = model.conv.weight

    # Permute kernel dims before flattening -> different K order -> different bytes.
    permuted = orig_weight.permute(0, 1, 4, 3, 2).reshape(orig_weight.shape[0], -1).contiguous()
    wrong = NVFP4QTensor.quantize(permuted, BLOCK_SIZE)[0]._quantized_data
    assert not torch.equal(packed, wrong)


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------


def test_conv3d_routed_via_process_quantized_modules():
    model = _quantize_conv3d(16, 32, (3, 3, 3))
    qa = quantizer_attr_names("weight")

    _process_quantized_modules(model, torch.float32)

    assert model.conv.weight.dtype == torch.uint8
    assert hasattr(model.conv, qa.weight_scale)
    assert hasattr(model.conv, qa.weight_scale_2)
    assert getattr(model.conv, qa.weight_scale).dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# No quantizer-state leakage
# ---------------------------------------------------------------------------


def test_conv3d_quantizers_hidden_for_save():
    model = _quantize_conv3d(16, 32, (3, 3, 3))
    _process_quantized_modules(model, torch.float32)

    # Quantizer submodules remain attached after export ...
    assert hasattr(model.conv, "weight_quantizer")

    # ... but are removed inside the save context, leaving no quantizer keys.
    with hide_quantizers_from_state_dict(model):
        assert not hasattr(model.conv, "weight_quantizer")
        leaked = [k for k in model.state_dict() if "_quantizer" in k]
        assert leaked == [], f"quantizer keys leaked into state_dict: {leaked}"

    # ... and are restored on exit.
    assert hasattr(model.conv, "weight_quantizer")


# ---------------------------------------------------------------------------
# Logical (un-swizzled) layout; pad/swizzle remain shape-compatible
# ---------------------------------------------------------------------------


def test_exported_conv_scale_is_logical_layout():
    model = _quantize_conv3d(16, 32, (3, 3, 3))
    qa = quantizer_attr_names("weight")
    _export_quantized_conv_weight(model.conv, torch.float32)

    # Default export keeps the logical [O, K_pad/16] scale, not a swizzled blob.
    weight_scale = getattr(model.conv, qa.weight_scale)
    assert weight_scale.shape == (32, 432 // BLOCK_SIZE)


def _mk_nvfp4_layer(prefix: str, out_ch: int, k_pad: int) -> dict[str, torch.Tensor]:
    """Synthesize a minimal NVFP4 layer (uint8 weight, fp8 scale, scalar scale_2)."""
    return {
        f"{prefix}.weight": torch.randint(0, 256, (out_ch, k_pad // 2), dtype=torch.uint8),
        f"{prefix}.weight_scale": torch.ones(out_ch, k_pad // BLOCK_SIZE).to(torch.float8_e4m3fn),
        f"{prefix}.weight_scale_2": torch.tensor(0.5, dtype=torch.float32),
    }


def test_pad_swizzle_exclude_conv_layers():
    """Layers in exclude_layers pass through pad/swizzle byte-for-byte unchanged.

    A Linear NVFP4 layer (not excluded) is transformed; the Conv3d layer
    (excluded, with a non-16-multiple out dim that would otherwise be padded)
    stays in its logical flattened-K layout.
    """
    sd = {**_mk_nvfp4_layer("transformer.proj", 64, 256), **_mk_nvfp4_layer("vae.conv1", 120, 240)}
    conv = {"vae.conv1"}
    conv_w = sd["vae.conv1.weight"].clone()
    conv_s = sd["vae.conv1.weight_scale"].clone()

    pad_nvfp4_weights(sd, "row_col", exclude_layers=conv)
    swizzle_nvfp4_scales(sd, exclude_layers=conv)

    # Excluded conv is unchanged.
    assert torch.equal(sd["vae.conv1.weight"], conv_w)
    assert torch.equal(sd["vae.conv1.weight_scale"], conv_s)
    assert sd["vae.conv1.weight_scale"].shape == (120, 240 // BLOCK_SIZE)
    # The non-excluded Linear IS transformed (swizzle pads rows to a multiple of 128).
    assert sd["transformer.proj.weight_scale"].shape[0] == 128


def test_postprocess_safetensors_excludes_conv(tmp_path):
    """Conv stays logical on disk when pad/swizzle are enabled for other layers."""
    from safetensors.torch import load_file, save_file

    sd = {**_mk_nvfp4_layer("transformer.proj", 64, 256), **_mk_nvfp4_layer("vae.conv1", 120, 240)}
    conv_w = sd["vae.conv1.weight"].clone()
    conv_s = sd["vae.conv1.weight_scale"].clone()

    comp_dir = tmp_path / "component"
    comp_dir.mkdir()
    save_file(dict(sd), str(comp_dir / "model.safetensors"))

    _postprocess_safetensors(
        comp_dir,
        nvfp4_exclude_layers={"vae.conv1"},
        enable_swizzle_layout=True,
        padding_strategy="row_col",
    )

    out = load_file(str(comp_dir / "model.safetensors"))
    assert torch.equal(out["vae.conv1.weight"], conv_w)
    assert torch.equal(out["vae.conv1.weight_scale"], conv_s)
    # Linear is swizzled (rows padded to 128).
    assert out["transformer.proj.weight_scale"].shape[0] == 128
