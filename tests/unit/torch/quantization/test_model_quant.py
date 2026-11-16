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

"""Initialization of floating QAD weights through configured quantizers."""

import copy

import pytest
import torch
from torch import nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.nn.modules.tensor_quantizer as quantizer_module
from modelopt.torch.quantization.utils import enable_weight_cache, weight_cache_scope

SEQUENTIAL_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [
                {"num_bits": 8, "axis": 0},
                {"num_bits": 4, "axis": 0},
            ],
            "enable": True,
        },
    ],
    "algorithm": "max",
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "config",
    [mtq.INT8_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.FP8_DEFAULT_CFG, SEQUENTIAL_CFG],
)
def test_initialize_qdq_weights_preserves_state_and_frozen_weights(config, dtype):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(256, 16), nn.Linear(16, 8)).to(dtype=dtype)
    inputs = torch.randn(2, 256, dtype=dtype)
    mtq.quantize(model, copy.deepcopy(config), lambda m: m(inputs))
    model[1].requires_grad_(False)
    expected = model[0].weight_quantizer(model[0].weight).detach().clone()
    parameters = dict(model.named_parameters())
    frozen = {name: p.detach().clone() for name, p in parameters.items() if name != "0.weight"}
    buffers = {name: b.clone() for name, b in model.named_buffers()}
    state = copy.deepcopy(mto.modelopt_state(model))

    assert mtq.initialize_qdq_weights(model, trainable_only=True) is model

    torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
    assert model[0].weight.dtype == dtype
    assert all(p is parameters[name] for name, p in model.named_parameters())
    for name, value in frozen.items():
        torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
    assert mto.modelopt_state(model) == state
    assert model[0].weight_quantizer.is_enabled


def test_initialize_qdq_weights_undoes_pre_scale():
    model = nn.Linear(16, 8, bias=False)
    mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG, lambda m: m(torch.randn(2, 16)))
    quantizer = model.weight_quantizer
    quantizer.pre_quant_scale = torch.linspace(0.5, 2.0, 16)
    expected = quantizer(model.weight).detach() / quantizer.pre_quant_scale
    scale = quantizer.pre_quant_scale.clone()
    mtq.initialize_qdq_weights(model)
    torch.testing.assert_close(model.weight, expected)
    torch.testing.assert_close(quantizer.pre_quant_scale, scale, rtol=0, atol=0)
    assert quantizer.is_enabled


@pytest.mark.parametrize(
    "alias",
    [
        "quantized",
        "transpose",
        "frozen",
        "disabled",
        "unquantized",
        "frozen_bias",
        "parameter_alias",
    ],
)
def test_initialize_qdq_weights_refuses_ambiguous_alias_before_changes(alias):
    model = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
    if alias not in {"frozen_bias", "parameter_alias"}:
        model[1].weight = nn.Parameter(
            model[0].weight.T if alias == "transpose" else model[0].weight
        )
    mtq.quantize(
        model, mtq.INT8_WEIGHT_ONLY_CFG, lambda modules: [m(torch.randn(2, 16)) for m in modules]
    )
    if alias == "frozen_bias":
        model[1].bias = nn.Parameter(model[0].weight[0], requires_grad=False)
    elif alias == "parameter_alias":
        model[0].extra_parameter = model[0].weight
    elif alias == "frozen":
        model[1].weight.requires_grad_(False)
    elif alias == "disabled":
        model[1].weight_quantizer.disable()
    elif alias == "unquantized":
        weight = model[1].weight
        model[1] = nn.Linear(16, 16)
        model[1].weight = weight
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError):
        mtq.initialize_qdq_weights(model, trainable_only=True)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [torch.zeros(16), torch.ones(8), torch.ones(2, 16, 16)])
def test_initialize_qdq_weights_validates_all_scales_before_changes(scale):
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG, lambda m: m(torch.randn(2, 16)))
    model[1].weight_quantizer.pre_quant_scale = scale
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError):
        mtq.initialize_qdq_weights(model)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotate_back", [False, True])
def test_initialize_qdq_weights_returns_to_latent_domain(monkeypatch, rotate_back, dtype):
    # CPU reference for the CUDA-only normalized Hadamard implementation.
    def hadamard(inputs, rotate_fp32=False, block_size=None):
        pairs = inputs.reshape(*inputs.shape[:-1], -1, 2)
        a, b = pairs.unbind(-1)
        return torch.stack((a + b, a - b), -1).reshape(inputs.shape) * (2**-0.5)

    monkeypatch.setattr(quantizer_module, "normalized_hadamard_transform", hadamard)
    torch.manual_seed(42)
    model = nn.Linear(16, 8, bias=False, dtype=dtype)
    inputs = torch.randn(2, 16, dtype=dtype)
    mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG, lambda m: m(inputs))
    q = model.weight_quantizer
    q.pre_quant_scale = torch.linspace(0.5, 2.0, 16, dtype=dtype)
    q.set_from_attribute_config(
        {"rotate": {"enable": True, "mode": "rotate_back" if rotate_back else "rotate"}}
    )
    transformed = q(model.weight).detach()
    expected = (transformed if rotate_back else hadamard(transformed)) / q.pre_quant_scale
    output_before = model(inputs).detach()
    mtq.initialize_qdq_weights(model)
    torch.testing.assert_close(model.weight, expected)
    relative_error = (
        model(inputs).float() - output_before.float()
    ).norm() / output_before.float().norm()
    assert relative_error < (0.03 if dtype == torch.bfloat16 else 1e-5)
    assert q.rotate_is_enabled and q.rotate_back_is_enabled == rotate_back


def test_initialize_qdq_weights_refuses_adapters_before_changes():
    model = nn.Linear(16, 8)
    mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG, lambda m: m(torch.randn(2, 16)))
    mtq.enable_quant_lora(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="adapter-free"):
        mtq.initialize_qdq_weights(model)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("initialize_children", [False, True])
def test_initialize_mixed_weights_with_cache(initialize_children):
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "0.weight_quantizer",
                "cfg": {"num_bits": 8, "axis": 0},
                "enable": True,
            },
            {
                "quantizer_name": "1.weight_quantizer",
                "cfg": {"num_bits": 4, "axis": 0},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    inputs = torch.randn(2, 16)
    mtq.quantize(model, config, lambda m: m(inputs))
    expected = [layer.weight_quantizer(layer.weight).detach().clone() for layer in model]
    enable_weight_cache(model)
    with weight_cache_scope(model) as stats:
        baseline = model(inputs).detach()
        misses = stats.misses
        for target in model if initialize_children else [model]:
            mtq.initialize_qdq_weights(target)
        assert stats.retained_bytes == 0 and stats.misses == misses
        torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
    for layer, weight in zip(model, expected):
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
