# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU unit tests for the LAQ algorithm using FP4 (NVFP4) quantization."""

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq

NVFP4_LAQ_POST_MSE_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "enable": False,
        },
    },
    "algorithm": {
        "method": "laq",
        "learnable_amax": ["post"],
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_LAQ_PRE_POST_MSE_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "enable": False,
        },
    },
    "algorithm": {
        "method": "laq",
        "learnable_amax": ["pre", "post"],
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_LAQ_TIED_MSE_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "enable": False,
        },
    },
    "algorithm": {
        "method": "laq",
        "learnable_amax": ["pre", "post"],
        "tied_amax": True,
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}


class SimpleModel(nn.Module):
    """Minimal model for LAQ testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.linear(x)


def _make_forward_loop(model, device):
    x = torch.randn(2, 64, device=device)

    def forward_loop(m):
        m(x)

    return forward_loop


@pytest.mark.parametrize(
    "config",
    [NVFP4_LAQ_POST_MSE_CFG, NVFP4_LAQ_PRE_POST_MSE_CFG, NVFP4_LAQ_TIED_MSE_CFG],
    ids=["post_only", "pre_and_post", "tied"],
)
def test_laq_quantize_e2e(config):
    """End-to-end: quantize a small model with LAQ + NVFP4 on GPU."""
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    forward_loop = _make_forward_loop(model, device)

    model = mtq.quantize(model, config, forward_loop=forward_loop)

    # Verify the model still produces output of the correct shape
    x = torch.randn(2, 64, device=device)
    out = model(x)
    assert out.shape == (2, 64)


def test_laq_fp4_fake_quantize_differentiable():
    """Test that _fake_quantize in FP4 LAQ mode is differentiable."""
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
        StaticBlockScaleQuantizer,
        TensorQuantizer,
    )

    device = torch.device("cuda")
    tq = TensorQuantizer()
    tq._num_bits = (2, 1)
    tq._unsigned = False
    tq._narrow_range = True
    tq._disabled = False
    tq._block_sizes = {-1: 16, "type": "static", "scale_bits": (4, 3)}
    tq._pass_through_bwd = True
    tq.register_buffer("_amax", torch.ones(4, device=device))
    tq.to(device)
    sbsq = StaticBlockScaleQuantizer.from_tensor_quantizer(
        tq, global_amax=torch.tensor(1.0, device=device)
    )

    amax = torch.ones(4, device=device) * 3.0
    per_tensor_scale = torch.tensor(1.0 / 6.0, device=device)
    sbsq.enable_laq(
        amax,
        per_tensor_scale=per_tensor_scale,
        quantize_scales=True,
        learnable_amax=["post"],
    )

    x = torch.randn(4, 16, device=device)
    out = sbsq._fake_quantize(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert sbsq._amax_post.grad is not None


def test_laq_fp4_cast_ste():
    """Test fp4_cast_ste on GPU."""
    from modelopt.torch.quantization.tensor_quant import fp4_cast_ste

    device = torch.device("cuda")
    x = torch.tensor([[-3.0, 1.5, 0.0, 6.0, -6.0, 0.5, -0.5, 2.0]], device=device)
    x.requires_grad_(True)
    # fp4_cast_ste expects [NUM_BLOCKS, BLOCK_SIZE] -- pad to block size 16
    x_padded = torch.zeros(1, 16, device=device, requires_grad=True)
    with torch.no_grad():
        x_padded[:, : x.shape[1]] = x.detach()
    x_padded = x_padded.clone().detach().requires_grad_(True)
    y = fp4_cast_ste(x_padded)
    assert y.shape == x_padded.shape
    y.sum().backward()
    assert x_padded.grad is not None
