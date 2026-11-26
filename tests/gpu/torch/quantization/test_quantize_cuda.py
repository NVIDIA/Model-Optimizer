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

import copy

import pytest
import torch
from _test_utils.torch.quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear
from _test_utils.torch.quantization.quantize_common import (
    FP4_SVDQUANT_CFG,
    INT4_AWQ_CLIP_CFG,
    INT4_AWQ_FULL_CFG,
    quantize_model_and_forward,
    save_restore_test,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.extensions import get_cuda_ext_mx

NVFP4_WEIGHT_ACT_MSE_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
    ],
    "algorithm": {
        "method": "mse",
        "step_size": 0.25,
        "start_multiplier": 0.25,
        "stop_multiplier": 2.0,
    },
}

NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG = {
    "quant_cfg": [
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
        {"quantizer_name": "*input_quantizer", "enable": False},
    ],
    "algorithm": {
        "method": "mse",
        "fp8_scale_sweep": True,
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
        config = copy.deepcopy(config)
        for entry in config["quant_cfg"]:
            if entry.get("quantizer_name") == "*weight_quantizer":
                entry.setdefault("cfg", {})["block_sizes"] = {-1: 8, -2: 8}
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]
    quantize_model_and_forward(model, config, calib_data)


@pytest.mark.skipif(
    not hasattr(tensor_quant.triton_kernel, "fp4_fake_quant_block"),
    reason="fp4_fake_quant_block requires compute >= 8.9",
)
@pytest.mark.parametrize(
    "config", [mtq.NVFP4_DEFAULT_CFG, mtq.W4A8_NVFP4_FP8_CFG, mtq.NVFP4_MLP_WEIGHT_ONLY_CFG]
)
def test_quantize_nvfp4_block_size(config):
    """NVFP4 presets fake-quantize weights in the block size they configure, end to end."""
    # Local import: transformers is optional and only this test in the module needs it.
    from _test_utils.torch.transformers_models import get_tiny_llama

    model = get_tiny_llama().cuda()
    down_proj = model.model.layers[0].mlp.down_proj
    with torch.no_grad():
        # Global amax 42: 0.75 stays exact in a block of its own but rounds to 0 in a block shared with 42.
        down_proj.weight.copy_(torch.tensor([0.75] * 16 + [42.0] * 16).expand_as(down_proj.weight))
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device="cuda")
    mtq.quantize(model, config, lambda m: m(input_ids))

    weight = down_proj.weight.detach().clone()
    if down_proj.weight_quantizer.block_sizes[-1] == 16:
        expected = weight
    else:
        expected = torch.where(weight == 0.75, 0.0, weight)
    with torch.no_grad():
        assert torch.equal(down_proj.weight_quantizer(weight), expected)
        logits = model(input_ids).logits
        down_proj.weight.copy_(expected)
        down_proj.weight_quantizer.disable()
        assert torch.equal(model(input_ids).logits, logits)


@pytest.mark.parametrize(
    ("model_cls", "quant_config"),
    [
        (SimpleLinear, mtq.INT8_SMOOTHQUANT_CFG),
        (SimpleLinear, mtq.W4A8_AWQ_BETA_CFG),
        (SimpleConvLinear, mtq.INT8_DEFAULT_CFG),
        (SimpleLinear, NVFP4_WEIGHT_MSE_FP8_SWEEP_CFG),
        (SimpleLinear, NVFP4_WEIGHT_ACT_MSE_CFG),
    ],
)
def test_save_restore(model_cls, quant_config):
    test_cpu_restore = quant_config == mtq.INT8_SMOOTHQUANT_CFG
    save_restore_test(model_cls, "cuda", quant_config, test_cpu_restore=test_cpu_restore)
