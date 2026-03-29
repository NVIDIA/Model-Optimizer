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

"""High-level tests for quantization."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear
from _test_utils.torch.quantization.quantize_common import (
    FP4_SVDQUANT_CFG,
    INT4_AWQ_CLIP_CFG,
    INT4_AWQ_FULL_CFG,
    quantize_model_and_forward,
    save_restore_test,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.extensions import get_cuda_ext_mx
from modelopt.torch.quantization.nn import NVFP4StaticAdaRoundQuantizer, NVFP4StaticQuantizer

NVFP4_WEIGHT_ACT_MSE_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
    },
    "algorithm": {
        "method": "mse",
        "step_size": 0.25,
        "start_multiplier": 0.25,
        "stop_multiplier": 2.0,
    },
}

NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG = {
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
        "method": "mse",
        "fp8_scale_sweep": True,
    },
}

NVFP4_WEIGHT_SCALE_LEARN_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
    },
    "algorithm": {
        "method": "scale_after_dequant",
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_WEIGHT_LSQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
    },
    "algorithm": {
        "method": "lsq",
        "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
    },
}

NVFP4_ADAROUND_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
    },
    "algorithm": {
        "method": "adaround",
        "scale_after_dequant_args": {"scale_algorithm": {"method": "mse", "fp8_scale_sweep": True}},
    },
}


@pytest.mark.parametrize("model_cls", [SimpleLinear, SimpleConv, SimpleConvLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        INT4_AWQ_CLIP_CFG,
        INT4_AWQ_FULL_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        FP4_SVDQUANT_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
        mtq.NVFP4_AWQ_CLIP_CFG,
        mtq.NVFP4_AWQ_FULL_CFG,
        mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
        mtq.MXFP8_DEFAULT_CFG,
        mtq.MXFP6_DEFAULT_CFG,
        mtq.MXFP4_DEFAULT_CFG,
        mtq.MXINT8_DEFAULT_CFG,
        mtq.NVFP4_KV_ROTATE_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        NVFP4_WEIGHT_ACT_MSE_CFG,
        NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG,
    ],
)
def test_quantize(model_cls, config):
    """Test quantize function can run without problems."""
    if config in [
        mtq.NVFP4_DEFAULT_CFG,
        FP4_SVDQUANT_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
        mtq.NVFP4_AWQ_CLIP_CFG,
        mtq.NVFP4_AWQ_FULL_CFG,
        mtq.MXFP8_DEFAULT_CFG,
        mtq.MXFP6_DEFAULT_CFG,
        mtq.MXFP4_DEFAULT_CFG,
        mtq.MXINT8_DEFAULT_CFG,
        mtq.NVFP4_KV_ROTATE_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        NVFP4_WEIGHT_ACT_MSE_CFG,
        NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG,
        mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
    ]:
        if get_cuda_ext_mx() is None:
            pytest.skip("cuda_ext_mx is not available")
        if model_cls in [SimpleConv, SimpleConvLinear]:
            pytest.skip("Conv weight quantization will fail as the kernel_size < FP4 blocksize")

    if config == mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG:
        # reduce block sizes for simple testing models
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 8, -2: 8}
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]
    quantize_model_and_forward(model, config, calib_data)


@pytest.mark.parametrize(
    ("model_cls", "quant_config"),
    [
        (SimpleLinear, mtq.INT8_SMOOTHQUANT_CFG),
        (SimpleLinear, mtq.W4A8_AWQ_BETA_CFG),
        (SimpleConvLinear, mtq.INT8_DEFAULT_CFG),
        (SimpleLinear, NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG),
        (SimpleLinear, NVFP4_WEIGHT_ACT_MSE_CFG),
        (SimpleLinear, NVFP4_ADAROUND_CFG),
    ],
)
def test_save_restore(model_cls, quant_config):
    if quant_config in (
        NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG,
        NVFP4_WEIGHT_ACT_MSE_CFG,
        NVFP4_ADAROUND_CFG,
    ):
        if get_cuda_ext_mx() is None:
            pytest.skip("cuda_ext_mx is not available")
    test_cpu_restore = quant_config == mtq.INT8_SMOOTHQUANT_CFG
    save_restore_test(model_cls, "cuda", quant_config, test_cpu_restore=test_cpu_restore)


def test_scale_after_dequant_grad():
    """Test scale_after_dequant: outputs match FP8 sweep, and per_block_scale gets gradients."""
    if get_cuda_ext_mx() is None:
        pytest.skip("cuda_ext_mx is not available")

    import copy

    model = nn.Sequential(nn.Linear(32, 64, bias=False), nn.Linear(64, 16, bias=False)).cuda()
    model_ref = copy.deepcopy(model)

    calib_data = [torch.randn(2, 32, device="cuda") for _ in range(4)]

    def forward_loop(model):
        for x in calib_data:
            model(x)

    # Reference: MSE + FP8 sweep only
    mtq.quantize(model_ref, NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG, forward_loop)

    # scale_after_dequant (internally runs MSE + FP8 sweep, then converts)
    mtq.quantize(model, NVFP4_WEIGHT_SCALE_LEARN_CFG, forward_loop)

    # Outputs should match before any training
    x = torch.randn(2, 32, device="cuda")
    with torch.no_grad():
        out_ref = model_ref(x)
        out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5), (
        f"Output mismatch: max diff = {(out - out_ref).abs().max().item()}"
    )

    # Verify quantizers are in scale_after_dequant mode
    for module in model.modules():
        if isinstance(module, NVFP4StaticQuantizer) and module._scale_after_dequant:
            assert isinstance(module._per_block_scale, nn.Parameter)
            assert module._per_block_scale.requires_grad
            assert not module._per_tensor_scale.requires_grad

    # Forward + backward: verify per_block_scale gets gradients
    out = model(x)
    out.sum().backward()

    found = False
    for module in model.modules():
        if isinstance(module, NVFP4StaticQuantizer) and module._scale_after_dequant:
            assert module._per_block_scale.grad is not None
            found = True
    assert found


def test_lsq_grad():
    """Test LSQ: quantizers in LSQ mode, per_block_scale gets gradients, output is finite."""
    if get_cuda_ext_mx() is None:
        pytest.skip("cuda_ext_mx is not available")

    import copy

    model = nn.Sequential(nn.Linear(32, 64, bias=False), nn.Linear(64, 16, bias=False)).cuda()
    model_ref = copy.deepcopy(model)

    calib_data = [torch.randn(2, 32, device="cuda") for _ in range(4)]

    def forward_loop(model):
        for x in calib_data:
            model(x)

    # Reference: MSE + FP8 sweep only
    mtq.quantize(model_ref, NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG, forward_loop)

    # LSQ (internally runs MSE + FP8 sweep, then converts to LSQ mode)
    mtq.quantize(model, NVFP4_WEIGHT_LSQ_CFG, forward_loop)

    # Verify output is finite
    x = torch.randn(2, 32, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Non-finite output detected"

    # Verify quantizers are in LSQ mode
    for module in model.modules():
        if isinstance(module, NVFP4StaticQuantizer) and module._scale_after_dequant:
            assert module._lsq, "Quantizer should be in LSQ mode"
            assert isinstance(module._per_block_scale, nn.Parameter)
            assert module._per_block_scale.requires_grad
            assert not module._per_tensor_scale.requires_grad

    # Forward + backward: verify per_block_scale gets gradients
    out = model(x)
    out.sum().backward()

    found = False
    for module in model.modules():
        if isinstance(module, NVFP4StaticQuantizer) and module._lsq:
            assert module._per_block_scale.grad is not None
            found = True
    assert found


def test_adaround_grad():
    """Test adaround: quantizers upgraded, round_logits gets gradients, dist_loss works."""
    if get_cuda_ext_mx() is None:
        pytest.skip("cuda_ext_mx is not available")

    import copy

    model = nn.Sequential(nn.Linear(32, 64, bias=False), nn.Linear(64, 16, bias=False)).cuda()
    model_ref = copy.deepcopy(model)

    calib_data = [torch.randn(2, 32, device="cuda") for _ in range(4)]

    def forward_loop(model):
        for x in calib_data:
            model(x)

    # Reference: scale_after_dequant only (no adaround)
    mtq.quantize(model_ref, NVFP4_WEIGHT_SCALE_LEARN_CFG, forward_loop)

    # Single call: adaround with embedded scale_after_dequant_args
    mtq.quantize(model, NVFP4_ADAROUND_CFG, forward_loop)

    # -- 1. Verify quantizers are upgraded to NVFP4StaticAdaRoundQuantizer --
    adaround_quantizers = [
        m
        for m in model.modules()
        if isinstance(m, NVFP4StaticAdaRoundQuantizer) and m._adaround_enabled
    ]
    assert len(adaround_quantizers) > 0, "No NVFP4StaticAdaRoundQuantizer found in model"

    # -- 2. Verify round_logits is a learnable parameter --
    for q in adaround_quantizers:
        assert hasattr(q, "round_logits")
        assert isinstance(q.round_logits, nn.Parameter)
        assert q.round_logits.requires_grad

    # -- 3. AdaRound Eval mode: outputs should be close to RTN
    model.eval()

    x = torch.randn(2, 32, device="cuda")
    out_ref = model_ref(x)
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-2), (
        f"Output mismatch: max diff = {(out - out_ref).abs().max().item()}"
    )

    # -- 4. Forward + backward: verify round_logits gets gradients --
    model.train()
    out = model(x)
    out.sum().backward()

    for q in adaround_quantizers:
        assert q.round_logits.grad is not None, "round_logits did not receive gradients"
        assert q.round_logits.grad.abs().sum() > 0, "round_logits gradients are all zero"

    # -- 5. dist_loss is differentiable and returns a scalar --
    model.zero_grad()
    total_dist_loss = sum(q.dist_loss() for q in adaround_quantizers)
    assert total_dist_loss.ndim == 0, "dist_loss should return a scalar"
    assert total_dist_loss.item() >= 0, "dist_loss should be non-negative"
    total_dist_loss.backward()
    for q in adaround_quantizers:
        assert q.round_logits.grad is not None, (
            "dist_loss did not produce gradients on round_logits"
        )

    # -- 6. AdaRound Eval mode: round_prob should be 0 or 1 --
    model.eval()
    for q in adaround_quantizers:
        round_prob = q.get_round_prob()
        hard = ((round_prob == 0) | (round_prob == 1)).all()
        # With stretched sigmoid, values near 0/1 are expected but not exact for all entries.
        # At minimum, no value should be far from {0, 1}.
        assert hard, "Eval round_prob is not 0 or 1"
