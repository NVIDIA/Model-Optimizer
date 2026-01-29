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

import pytest
import torch
from _test_utils.torch.export.utils import (
    SmallLinearModelwithCustomWeight,
    ToyModel,
    only_input_quantizer_fp8_config,
    only_output_quantizer_fp8_config,
    only_weight_quantizer_fp8_config,
    partial_fp8_config,
    partial_fp8_kv_cache_config,
    partial_int4_awq_config,
    partial_int8_kv_cache_config,
    partial_nvfp4_awq_config,
    partial_nvfp4_config,
    partial_w4a8_config,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_config import (
    KV_CACHE_FP8,
    KV_CACHE_INT8,
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_W4A8_AWQ,
)
from modelopt.torch.export.quant_utils import (
    adjust_attn_amax_values,
    all_items_same,
    get_kv_cache_dtype,
    get_quant_config,
    get_quantization_format,
    get_scaling_factor,
    get_scaling_factor_from_weight,
    get_weight_block_size,
    postprocess_state_dict,
    process_layer_quant_config,
)
from modelopt.torch.quantization.config import (
    FP8_DEFAULT_CFG,
    INT4_AWQ_CFG,
    INT8_SMOOTHQUANT_CFG,
    INT8_WEIGHT_ONLY_CFG,
    NVFP4_AWQ_LITE_CFG,
    NVFP4_DEFAULT_CFG,
    W4A8_AWQ_BETA_CFG,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (partial_fp8_config, QUANTIZATION_FP8),
        (partial_w4a8_config, QUANTIZATION_W4A8_AWQ),
        (partial_nvfp4_config, QUANTIZATION_NVFP4),
        (partial_nvfp4_awq_config, QUANTIZATION_NVFP4_AWQ),
        (partial_int4_awq_config, QUANTIZATION_INT4_AWQ),
    ],
)
def test_get_quantization_format(config, expected):
    model = ToyModel().to("cuda")
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10, device="cuda")))
    assert get_quantization_format(model) == expected


@pytest.mark.parametrize(
    ("layer_config_dict", "expected_processed_dict"),
    [
        (
            {
                "layer1.quantization": "nvfp4",  # All qformats
                "layer1.awq_block_size": 16,
                "layer3.quantization": "int4_awq",
                "layer3.awq_block_size": 8,
                "layer4.quantization": "w4a8_awq",
                "layer4.awq_block_size": 64,
                "layer5.quantization": "int8_sq",
                "layer6.quantization": "fp8",
                "layer7.quantization": "xyz",
                "layer8.quantization": None,
            },
            {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "quantized_layers": {
                    "layer1": {"quant_algo": "NVFP4", "group_size": 16},
                    "layer3": {
                        "quant_algo": "W4A16_AWQ",
                        "group_size": 8,
                        "has_zero_point": False,
                        "pre_quant_scale": True,
                    },
                    "layer4": {
                        "quant_algo": "W4A8_AWQ",
                        "group_size": 64,
                        "has_zero_point": False,
                        "pre_quant_scale": True,
                    },
                    "layer5": {"quant_algo": "W8A8_SQ_PER_CHANNEL"},
                    "layer6": {"quant_algo": "FP8"},
                    "layer7": {"quant_algo": "xyz"},
                },
            },
        ),
        (
            {
                "layer1.quantization": "nvfp4",  # Auto quant with one qformat case
                "layer1.awq_block_size": 16,
                "layer2.quantization": "nvfp4",
                "layer2.awq_block_size": 16,
                "layer8.quantization": None,
            },
            {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": ["layer8"],
            },
        ),
    ],
)
def test_process_layer_quant_config(layer_config_dict, expected_processed_dict):
    per_layer_config = process_layer_quant_config(layer_config_dict)
    assert per_layer_config == expected_processed_dict


@pytest.mark.parametrize(
    ("item_list", "expected"),
    [
        (["a", "a", "a"], True),
        (["b", "a", "a"], False),
        (["a"], True),
        ([True, True, True], True),
        ([True, False, True], False),
        ([False, False, False], True),
        ([False], True),
        ([True], True),
    ],
)
def test_all_items_same(item_list, expected):
    generated = all_items_same(item_list)
    assert generated == expected


@pytest.mark.parametrize(
    ("weight", "group_size", "expected"),
    [
        (
            torch.tensor([[0.0, 0.35, 0.28, 7.0], [0.49, 0.84, -0.77, 0.07]]),
            2,
            torch.tensor([[0.05, 1.0], [0.12, 0.11]]),
        ),  # group_size != 0 and divides weight.shape[1]
        (
            torch.tensor([[0.127, 0.0, 1.27, -12.7], [0.0, 127.0, 0.254, 2.54]]),
            0,
            torch.tensor([0.1, 1.0]),
        ),  # group_size = 0
        (
            torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, -0.127, 0.254, 2.54]]),
            0,
            torch.tensor([1.0, 0.02]),
        ),  # zero replaced with 1.0
        (
            torch.tensor([[0.0, 0.84, -0.77, 0.07], [0.0, 0.0, 0.0, 0.0]]),
            2,
            torch.tensor([[0.12, 0.11], [1.0, 1.0]]),
        ),  # zero replaced with 1.0
    ],
)
def test_get_scaling_factor_from_weight(weight, group_size, expected):
    scaling_factor = get_scaling_factor_from_weight(weight, group_size)
    # Check if shapes match
    if group_size != 0:
        assert list(scaling_factor.shape) == [weight.shape[0], weight.shape[1] // group_size]
    else:
        assert list(scaling_factor.shape) == [weight.shape[0]]

    assert torch.allclose(scaling_factor, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("state_dict", "quantization", "maxbound", "expected_state_dict"),
    [
        (  # Test replacements and KV cache scaling
            {
                "layer1.k_bmm_quantizer._amax": torch.tensor([0.128]),
                "layer1.v_bmm_quantizer._amax": torch.tensor([256.0]),
                "layer1.input_quantizer._pre_quant_scale": torch.tensor([0.128]),
            },
            KV_CACHE_FP8,
            128.0,
            {
                "layer1.k_proj.k_scale": torch.tensor([1.0]),
                "layer1.v_proj.v_scale": torch.tensor([2.0]),
                "layer1.pre_quant_scale": torch.tensor([0.128]),
            },
        ),
        (  # Test skipping output_quantizer _amax keys other than k_scale and v_scale
            {
                "layer1.q_bmm_quantizer._amax": torch.tensor([0.128]),
                "layer1.k_bmm_quantizer._amax": torch.tensor([0.128]),
                "layer1.v_bmm_quantizer._amax": torch.tensor([256]),
            },
            KV_CACHE_FP8,
            128.0,
            {
                "layer1.k_proj.k_scale": torch.tensor([0.001]),
                "layer1.v_proj.v_scale": torch.tensor([2.0]),
            },
        ),
        (  # Test squeezing tensor with leading dimension 1
            {
                "layer1.k_proj.weight": torch.ones(1, 1),
            },
            KV_CACHE_FP8,
            128.0,
            {
                "layer1.k_proj.weight": torch.ones(1),
            },
        ),
        (  # Test case with no KV cache scaling + AWQ quant
            {
                "layer1.input_quantizer._pre_quant_scale": torch.tensor([0.128]),
            },
            QUANTIZATION_NONE,
            128.0,
            {
                "layer1.pre_quant_scale": torch.tensor([0.128]),
            },
        ),
    ],
)
def test_postprocess_state_dict(state_dict, quantization, maxbound, expected_state_dict):
    processed_state_dict = postprocess_state_dict(state_dict, maxbound, quantization)
    assert processed_state_dict == expected_state_dict


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (partial_fp8_kv_cache_config, KV_CACHE_FP8),
        (partial_int8_kv_cache_config, KV_CACHE_INT8),
    ],
)
def test_get_kv_cache_dtype(config, expected):
    model = ToyModel().to("cuda")
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10, device="cuda")))

    # Create list of modules in model
    modules = []
    for name, module in model.named_modules():
        modules.append(module)

    kv_cache_dtype = get_kv_cache_dtype(modules)
    assert kv_cache_dtype == expected


# Tensor Quantizer extraction for export tests
@pytest.mark.parametrize(
    ("q_weight", "k_weight", "v_weight", "o_weight", "expected_qkv_amax", "expected_o_amax"),
    [
        (
            torch.tensor([[0.1, 0.3], [0.22, 0.45]]),
            torch.tensor([[0.44, 0.32], [0.11, 0.95]]),
            torch.tensor([[0.9, 0.03], [0.92, 0.8]]),
            torch.tensor([[0.01, 0.97], [0.29, 0.77]]),
            0.95,
            0.97,
        ),
        (
            torch.tensor([[0.1, 0.3], [0.22, 0.45]]),
            torch.tensor([[0.44, 0.32], [0.11, -0.95]]),
            torch.tensor([[0.9, 0.03], [0.92, 0.8]]),
            torch.tensor([[0.01, -0.97], [0.29, 0.77]]),
            0.95,
            0.97,
        ),
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        FP8_DEFAULT_CFG,
        NVFP4_DEFAULT_CFG,
    ],
)
def test_adjust_attn_amax_values(
    q_weight, k_weight, v_weight, o_weight, expected_qkv_amax, expected_o_amax, config
):
    # Initialize model and quantize to insert quantizers
    model = SmallLinearModelwithCustomWeight([q_weight, k_weight, v_weight, o_weight]).to("cuda")
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, q_weight.shape[1], device="cuda")))
    adjust_attn_amax_values(model)
    # Weight quantizer amax must remain unchanged for non qkv layers
    assert (
        model.q_proj.weight_quantizer.amax
        == model.k_proj.weight_quantizer.amax
        == model.v_proj.weight_quantizer.amax
        == expected_qkv_amax
    )
    # Weight quantizer amax must be updated for q,k,v layers
    assert model.o_proj.weight_quantizer.amax == expected_o_amax


@pytest.mark.parametrize(
    ("config", "expected_block_size"),
    [
        (FP8_DEFAULT_CFG, 0),
        (INT8_WEIGHT_ONLY_CFG, 0),
        (INT8_SMOOTHQUANT_CFG, 0),
        (NVFP4_DEFAULT_CFG, 16),
        (NVFP4_AWQ_LITE_CFG, 16),
        (W4A8_AWQ_BETA_CFG, 128),
        (INT4_AWQ_CFG, 128),
        (partial_nvfp4_config, 16),
    ],
)
def test_get_weight_block_size(config, expected_block_size):
    model = ToyModel().to("cuda")
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10, device="cuda")))

    for _, module in model.named_modules():
        block_size = get_weight_block_size(module)
        if hasattr(module, "weight_quantizer"):
            if (
                isinstance(module.weight_quantizer, SequentialQuantizer)
                or module.weight_quantizer.is_enabled
            ):
                assert block_size == expected_block_size
            else:
                assert block_size == 0

        else:
            assert block_size == 0


@pytest.mark.parametrize(
    ("config", "maxbound", "expected_amax"),
    [
        (only_weight_quantizer_fp8_config, 448, [0.45, 0.95, 0.92, 0.97]),
        (only_input_quantizer_fp8_config, 448, [1.0, 0.67, 0.68, 0.9]),
        (only_output_quantizer_fp8_config, 448, [0.67, 0.68, 0.9, 0.88]),
    ],
)
@pytest.mark.parametrize(
    ("q_weight", "k_weight", "v_weight", "o_weight"),
    [
        (
            torch.tensor([[0.1, 0.3], [0.22, 0.45]]),
            torch.tensor([[0.44, 0.32], [0.11, 0.95]]),
            torch.tensor([[0.9, 0.03], [0.92, 0.8]]),
            torch.tensor([[0.01, 0.97], [0.29, 0.77]]),
        ),
    ],
)
def test_get_scaling_factor(
    q_weight, k_weight, v_weight, o_weight, config, expected_amax, maxbound
):
    # Initialize model and quantize to insert quantizers
    model = SmallLinearModelwithCustomWeight([q_weight, k_weight, v_weight, o_weight]).to("cuda")
    mtq.quantize(model, config, lambda x: x(torch.ones(1, 2, q_weight.shape[1], device="cuda")))
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            scale = get_scaling_factor(module)
            print(f"DEBUG LOG: Scale: {scale}, Expected: {expected_amax[0] / maxbound}")
            assert torch.allclose(
                scale,
                torch.tensor((expected_amax[0] / maxbound), dtype=scale.dtype),
                rtol=1e-3,
                atol=1e-3,
            )
            expected_amax.pop(0)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            partial_fp8_config,
            {
                "exclude_modules": ["linears.0", "linears.2"],
                "kv_cache_quant_algo": None,
                "quant_algo": "FP8",
            },
        ),
        (
            partial_w4a8_config,
            {
                "exclude_modules": ["linears.0", "linears.1"],
                "group_size": 128,
                "has_zero_point": False,
                "kv_cache_quant_algo": None,
                "pre_quant_scale": True,
                "quant_algo": "W4A8_AWQ",
            },
        ),
        (
            partial_nvfp4_config,
            {
                "exclude_modules": ["linears.0", "linears.2"],
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "quant_algo": "NVFP4",
            },
        ),
        (
            partial_nvfp4_awq_config,
            {
                "exclude_modules": ["linears.0", "linears.1"],
                "group_size": 16,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "kv_cache_quant_algo": None,
                "quant_algo": "NVFP4_AWQ",
            },
        ),
        (
            partial_int4_awq_config,
            {
                "exclude_modules": ["linears.0", "linears.1"],
                "group_size": 128,
                "has_zero_point": False,
                "kv_cache_quant_algo": None,
                "pre_quant_scale": True,
                "quant_algo": "W4A16_AWQ",
            },
        ),
        (
            partial_fp8_kv_cache_config,
            {
                "exclude_modules": ["linears.0", "linears.2"],
                "quant_algo": "FP8",
                "kv_cache_quant_algo": "FP8",
            },
        ),
        (
            partial_int8_kv_cache_config,
            {
                "exclude_modules": ["linears.0", "linears.2"],
                "quant_algo": "FP8",
                "kv_cache_quant_algo": "INT8",
            },
        ),
    ],
)
def test_get_quant_config(config, expected):
    model = ToyModel().to("cuda")
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10, device="cuda")))
    quant_config = get_quant_config(model)
    assert quant_config["quantization"] == expected
