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

"""Mixed-precision (NVFP4 + FP8) behavior of mse_calibrate. Requires CUDA because
the NVFP4 forward path uses a fused Triton kernel."""

import pytest
import torch

from modelopt.torch.quantization.calib import MseCalibrator
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer


class _TwoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16, bias=False)
        self.linear2 = torch.nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def _mixed_nvfp4_fp8_config():
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {  # Layer 1 — NVFP4 static (eligible for NVFP4StaticQuantizer promotion).
                "quantizer_name": "*linear1.weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    "axis": None,
                },
            },
            {  # Layer 2 — FP8 per-tensor.
                "quantizer_name": "*linear2.weight_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
            },
        ],
        "algorithm": "max",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 path requires CUDA")
def test_mixed_nvfp4_fp8_sweep_true_skips_fp8():
    """fp8_scale_sweep=True: NVFP4 layer is promoted to NVFP4StaticQuantizer; the
    FP8 layer is left as a plain TensorQuantizer (no backend factory registered →
    no MseCalibrator replacement, max-calibrated amax preserved)."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import mse_calibrate

    device = torch.device("cuda")
    model = _TwoLayer().to(device)
    inputs = torch.randn(1, 16, device=device)
    mtq.quantize(model, _mixed_nvfp4_fp8_config(), forward_loop=lambda m: m(inputs))
    mse_calibrate(model, lambda m: m(inputs), fp8_scale_sweep=True)

    assert isinstance(model.linear1.weight_quantizer, NVFP4StaticQuantizer)
    # FP8 layer: exact type TensorQuantizer, no MseCalibrator replacement.
    assert type(model.linear2.weight_quantizer) is TensorQuantizer
    assert not isinstance(model.linear2.weight_quantizer._calibrator, MseCalibrator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 path requires CUDA")
def test_mixed_nvfp4_fp8_sweep_false_uses_mse_for_both():
    """fp8_scale_sweep=False: both NVFP4 and FP8 layers get an MseCalibrator. NVFP4
    layer is still promoted to NVFP4StaticQuantizer (promotion is independent of
    the sweep flag)."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import mse_calibrate

    device = torch.device("cuda")
    model = _TwoLayer().to(device)
    inputs = torch.randn(1, 16, device=device)
    mtq.quantize(model, _mixed_nvfp4_fp8_config(), forward_loop=lambda m: m(inputs))
    mse_calibrate(model, lambda m: m(inputs), fp8_scale_sweep=False)

    assert isinstance(model.linear1.weight_quantizer, NVFP4StaticQuantizer)
    assert isinstance(model.linear1.weight_quantizer._calibrator, MseCalibrator)
    assert type(model.linear2.weight_quantizer) is TensorQuantizer
    assert isinstance(model.linear2.weight_quantizer._calibrator, MseCalibrator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 path requires CUDA")
def test_output_layer_nvfp4_promotion_and_forward():
    """An attribute named `output_layer` with W4A16 NVFP4 config is promoted to
    NVFP4StaticQuantizer and its forward dispatches cleanly through the static
    blockwise FP4 kernel (regression for the lm_head crash that motivated the
    NVFP4 promotion in mse_calibrate)."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import mse_calibrate

    class _WithOutputLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(16, 16, bias=False)
            self.output_layer = torch.nn.Linear(16, 32, bias=False)

        def forward(self, x):
            return self.output_layer(self.decoder(x))

    device = torch.device("cuda")
    model = _WithOutputLayer().to(device)
    inputs = torch.randn(1, 16, device=device)
    nvfp4_cfg = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
        "axis": None,
    }
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*decoder.weight_quantizer", "cfg": nvfp4_cfg},
            {"quantizer_name": "*output_layer.weight_quantizer", "cfg": nvfp4_cfg},
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, forward_loop=lambda m: m(inputs))
    mse_calibrate(model, lambda m: m(inputs), fp8_scale_sweep=True)

    # Both layers (decoder + output_layer) must be promoted.
    assert isinstance(model.decoder.weight_quantizer, NVFP4StaticQuantizer)
    assert isinstance(model.output_layer.weight_quantizer, NVFP4StaticQuantizer)
    # Forward must dispatch through static_blockwise_fp4_fake_quant without
    # falling into the FP8-only scaled_e4m3 path. Pre-promotion this raised
    # NotImplementedError("Only support E=4 & M=3 for now.").
    with torch.no_grad():
        out = model(inputs)
    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 path requires CUDA")
def test_output_layer_nvfp4_export_keys():
    """A W4A16-quantized output_layer exports with CT-style weight + scale keys."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export.quant_utils import get_weight_scaling_factor
    from modelopt.torch.quantization.model_calib import mse_calibrate

    class _OutputOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = torch.nn.Linear(16, 32, bias=False)

        def forward(self, x):
            return self.output_layer(x)

    device = torch.device("cuda")
    model = _OutputOnly().to(device)
    inputs = torch.randn(1, 16, device=device)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*output_layer.weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    "axis": None,
                },
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, forward_loop=lambda m: m(inputs))
    mse_calibrate(model, lambda m: m(inputs), fp8_scale_sweep=True)

    # Export-time scale extraction must succeed for the promoted output_layer
    # without raising RuntimeError on the per-block amax shape mismatch that
    # broke the original (un-promoted) HF export.
    scale = get_weight_scaling_factor(model.output_layer)
    assert scale is not None
    # Block dim should match weight.shape[-1] / 16.
    expected_blocks = model.output_layer.weight.shape[-1] // 16
    assert scale.shape[-1] == expected_blocks
