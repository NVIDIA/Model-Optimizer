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

"""GPU unit tests for the LSQ algorithm using FP4 (NVFP4) quantization."""

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    StaticBlockScaleQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.tensor_quant import fp4_cast_ste

NVFP4_LSQ_POST_MSE_CFG = {
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
        "method": "lsq",
        "learnable_amax": ["post"],
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_LSQ_PRE_POST_MSE_CFG = {
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
        "method": "lsq",
        "learnable_amax": ["pre", "post"],
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_LSQ_TIED_MSE_CFG = {
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
        "method": "lsq",
        "learnable_amax": ["pre", "post"],
        "tied_amax": True,
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_LSQ_SKIP_PRE_SCALE_MSE_CFG = {
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
        "method": "lsq",
        "learnable_amax": ["post"],
        "quantize_pre_scale": False,
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}


class SimpleModel(nn.Module):
    """Minimal model for LSQ testing."""

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
    [
        NVFP4_LSQ_POST_MSE_CFG,
        NVFP4_LSQ_PRE_POST_MSE_CFG,
        NVFP4_LSQ_TIED_MSE_CFG,
        NVFP4_LSQ_SKIP_PRE_SCALE_MSE_CFG,
    ],
    ids=["post_only", "pre_and_post", "tied", "skip_pre_scale"],
)
def test_lsq_quantize_e2e(config):
    """End-to-end: quantize a small model with LSQ + NVFP4 on GPU."""
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    forward_loop = _make_forward_loop(model, device)

    model = mtq.quantize(model, config, forward_loop=forward_loop)
    assert model.linear.weight_quantizer._quantize_pre_scale is config["algorithm"].get(
        "quantize_pre_scale", True
    )

    # Verify the model still produces output of the correct shape
    x = torch.randn(2, 64, device=device)
    out = model(x)
    assert out.shape == (2, 64)


@pytest.mark.parametrize("tied_amax", [True, False], ids=["tied", "default_dual"])
def test_lsq_fp4_fake_quantize_differentiable(tied_amax):
    """Representable small scales retain static outputs and analytical LSQ gradients."""
    device = torch.device("cuda")
    scales = torch.tensor([2.0**-5, 1.0, 448.0], device=device)
    codes = torch.tensor(
        [0.625, -0.625, 1, -1, 1.5, -1.5, 2, -2, 3, -3, 4, -4, 6, -6, 0, 0], device=device
    )
    x = scales[:, None] * codes
    rounded = codes.clone()
    rounded[:2] = torch.tensor([0.5, -0.5], device=device)
    expected = scales[:, None] * rounded

    tq = TensorQuantizer()
    tq._num_bits = (2, 1)
    tq._unsigned = False
    tq._narrow_range = True
    tq._disabled = False
    tq._block_sizes = {-1: 16, "type": "static", "scale_bits": (4, 3)}
    tq._pass_through_bwd = True
    tq.register_buffer("_amax", 6.0 * scales)
    tq.to(device)
    sbsq = StaticBlockScaleQuantizer.from_tensor_quantizer(
        tq, global_amax=torch.tensor(6.0 * 448.0, device=device)
    )

    static_output = sbsq(x)
    torch.testing.assert_close(static_output, expected, rtol=0, atol=0)
    if tied_amax:
        sbsq.enable_lsq(learnable_amax=["pre", "post"], tied_amax=True)
    else:
        sbsq.enable_lsq()

    out = sbsq(x)
    torch.testing.assert_close(out, static_output, rtol=0, atol=0)
    (out[0, 0] + out[1, 0]).backward()
    # At x/scale=0.625, Q(x/scale)=0.5 and d(scale)/d(amax)=1/6.
    # Tied: (0.5 - 0.625)/6; default dual: 0.5/6 with fixed pre scale.
    gradient = -1.0 / 48.0 if tied_amax else 1.0 / 12.0
    torch.testing.assert_close(
        sbsq.amax_post.grad,
        torch.tensor([gradient, gradient, 0.0], device=device),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("tied_amax", [False, True])
def test_lsq_fp4_small_global_amax(tied_amax):
    """A block representable in NVFP4 survives LSQ fake quant even when global_amax < 6 * 2**-24."""
    device = torch.device("cuda")
    x = torch.full((1, 16), 2.0**-22, device=device)
    tq = TensorQuantizer()
    tq._num_bits = (2, 1)
    tq._unsigned = False
    tq._narrow_range = True
    tq._disabled = False
    tq._block_sizes = {-1: 16, "type": "static", "scale_bits": (4, 3)}
    tq.register_buffer("_amax", x.abs().amax(-1))
    sbsq = StaticBlockScaleQuantizer.from_tensor_quantizer(tq, global_amax=x.abs().amax())
    sbsq.enable_lsq(quantize_scales=True, tied_amax=tied_amax)

    assert torch.equal(sbsq._fake_quantize(x), x)


def test_lsq_fp4_cast_ste():
    """Test fp4_cast_ste on GPU."""
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
