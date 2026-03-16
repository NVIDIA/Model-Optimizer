# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""End-to-end integration tests for NVFP4_4O6_W4A4_CFG."""

import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import NVFP4_4O6_W4A4_CFG

# CPU-compatible test config: INT8 weights + 4o6 input quantizer.
# Dynamic NVFP4 block quantization for weights requires CUDA; using INT8 weights
# lets the integration tests run on CPU while still exercising the 4o6 activation path.
_4O6_INPUT_QUANTIZER = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
    "backend": "nvfp4_4o6",
    "backend_extra_args": {"scale_rule": "mse"},
}

_CPU_4O6_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": None, "enable": True},
        "*input_quantizer": _4O6_INPUT_QUANTIZER,
        "default": {"enable": False},
    },
    "algorithm": "max",
}


def _make_linear(in_f: int = 64, out_f: int = 64) -> nn.Linear:
    torch.manual_seed(42)
    return nn.Linear(in_f, out_f)


class TestQuantizeLinearWith4o6Config:
    def test_quantize_linear_with_4o6_config(self):
        """Quantizing with NVFP4_4O6_W4A4_CFG sets input_quantizer.backend = 'nvfp4_4o6'."""
        model = _make_linear()
        # Only run calibration (no inference) to avoid CUDA requirement in weight quantizer
        mtq.quantize(model, NVFP4_4O6_W4A4_CFG, lambda m: None)
        assert model.input_quantizer.backend == "nvfp4_4o6"

    def test_weight_quantizer_unaffected(self):
        """Weight quantizer should not use a custom backend."""
        model = _make_linear()
        mtq.quantize(model, NVFP4_4O6_W4A4_CFG, lambda m: None)
        assert model.weight_quantizer.backend is None


class TestForwardPassRuns:
    def test_forward_pass_runs(self):
        """Forward pass on a CPU-friendly quantized model works correctly."""
        model = _make_linear()
        x = torch.randn(4, 64)
        mtq.quantize(model, _CPU_4O6_CFG, lambda m: m(x))
        out = model(x)
        assert out.shape == (4, 64)
        assert not torch.any(torch.isnan(out))


class Test4o6LowerActError:
    def test_4o6_lower_act_error(self):
        """4o6 input quantizer gives lower or equal MSE than static-6 encoding on outlier inputs.

        This test verifies the quality guarantee of the 4o6 algorithm by comparing the
        input_quantizer output against the static-6 reference path from the 4o6 kernel directly,
        avoiding the CUDA requirement of modelopt's dynamic block quantization.
        """
        torch.manual_seed(0)
        x = torch.randn(4, 64)
        x[0, :8] *= 100  # inject outliers

        from modelopt.torch.quantization.calib.fouroversix import (
            _fake_quantize_to_e2m1,
            _quantize_to_nvfp4,
            nvfp4_4o6_fake_quant,
        )

        x_amax = x.abs().max().float()

        # 4o6 output
        out_4o6 = nvfp4_4o6_fake_quant(x, x_amax, scale_rule="mse")

        # Static-6 baseline (no adaptive scale selection)
        x_blocks = x.reshape(-1, 16).float()
        x_scaled_6, scales_6 = _quantize_to_nvfp4(x_blocks, x_amax)
        x_fq_6 = _fake_quantize_to_e2m1(x_scaled_6)
        denom = 6 * 256
        out_static6 = (
            x_fq_6 * scales_6.unsqueeze(1).to(torch.float32) * x_amax / denom
        ).reshape_as(x)

        mse_4o6 = ((out_4o6 - x) ** 2).mean().item()
        mse_static6 = ((out_static6 - x) ** 2).mean().item()

        assert mse_4o6 <= mse_static6, (
            f"4o6 act MSE ({mse_4o6:.6f}) should be <= static-6 MSE ({mse_static6:.6f})"
        )


class TestScaleRuleMae:
    def test_scale_rule_mae(self):
        """MAE scale rule runs without error and produces valid output."""
        cfg = {
            "quant_cfg": {
                "*weight_quantizer": {"num_bits": 8, "axis": None, "enable": True},
                "*input_quantizer": {
                    **_4O6_INPUT_QUANTIZER,
                    "backend_extra_args": {"scale_rule": "mae"},
                },
                "default": {"enable": False},
            },
            "algorithm": "max",
        }
        model = _make_linear()
        x = torch.randn(4, 64)
        mtq.quantize(model, cfg, lambda m: m(x))
        out = model(x)
        assert out.shape == (4, 64)
        assert not torch.any(torch.isnan(out))


class TestModeloptStateRoundtrip:
    def test_modelopt_state_roundtrip(self):
        """modelopt_state save/restore preserves backend and backend_extra_args."""
        model = _make_linear()
        x = torch.randn(4, 64)
        mtq.quantize(model, _CPU_4O6_CFG, lambda m: m(x))

        assert model.input_quantizer.backend == "nvfp4_4o6"
        assert model.input_quantizer.backend_extra_args.get("scale_rule") == "mse"

        state = mto.modelopt_state(model)

        model2 = _make_linear()
        mto.restore_from_modelopt_state(model2, state)

        assert model2.input_quantizer.backend == "nvfp4_4o6"
        assert model2.input_quantizer.backend_extra_args.get("scale_rule") == "mse"

        out = model2(x)
        assert out.shape == (4, 64)
