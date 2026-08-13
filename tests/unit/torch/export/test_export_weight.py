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


import copy

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config, partial_w4a8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quantized_weight import (
    build_hf_quantization_config,
    capture_quantized_weight_export_state,
    export_quantized_weight,
)
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _process_quantized_modules,
)
from modelopt.torch.quantization.nn import GroupedQuantizer
from modelopt.torch.quantization.utils import quantizer_attr_names


@pytest.mark.parametrize(
    "weight_name",
    ["weight", "weight_2", "some_other_w"],
)
def test_quantizer_attr_names(weight_name):
    quantizer_attrs = quantizer_attr_names(weight_name)
    if weight_name == "weight":
        assert quantizer_attrs.weight_scale == "weight_scale"
        assert quantizer_attrs.input_scale == "input_scale"
        assert quantizer_attrs.weight_scale_2 == "weight_scale_2"
        assert quantizer_attrs.weight_quantizer == "weight_quantizer"
        assert quantizer_attrs.input_quantizer == "input_quantizer"
        assert quantizer_attrs.output_quantizer == "output_quantizer"
        assert quantizer_attrs.output_scale == "output_scale"
    else:
        assert quantizer_attrs.weight_scale == f"{weight_name}_weight_scale"
        assert quantizer_attrs.input_scale == f"{weight_name}_input_scale"
        assert quantizer_attrs.weight_scale_2 == f"{weight_name}_weight_scale_2"
        assert quantizer_attrs.weight_quantizer == f"{weight_name}_weight_quantizer"
        assert quantizer_attrs.input_quantizer == f"{weight_name}_input_quantizer"
        assert quantizer_attrs.output_quantizer == f"{weight_name}_output_quantizer"
        assert quantizer_attrs.output_scale == f"{weight_name}_output_scale"


def test_export_per_tensor_quantized_weight():
    model = ToyModel(dims=[32, 256, 32, 128])

    mtq.quantize(model, partial_fp8_config, lambda x: x(torch.randn(1, 4, 32)))

    orig_dtype = model.linears[0].weight.dtype
    quantizer_attrs = quantizer_attr_names("weight")
    _export_quantized_weight(model.linears[0], torch.float32, "weight")
    assert model.linears[0].weight.dtype == orig_dtype
    assert hasattr(model.linears[0], quantizer_attrs.weight_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.weight_quantizer).is_enabled
    assert not hasattr(model.linears[0], quantizer_attrs.weight_scale)
    assert not hasattr(model.linears[0], quantizer_attrs.weight_scale_2)
    assert not hasattr(model.linears[0], quantizer_attrs.input_scale)
    assert hasattr(model.linears[0], quantizer_attrs.input_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.input_quantizer).is_enabled
    assert hasattr(model.linears[0], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[0], quantizer_attrs.output_scale)

    _export_quantized_weight(model.linears[1], torch.float32, "weight")
    assert model.linears[1].weight.dtype == torch.float8_e4m3fn
    assert hasattr(model.linears[1], quantizer_attrs.weight_quantizer)
    assert hasattr(model.linears[1], quantizer_attrs.weight_scale)
    assert not hasattr(model.linears[1], quantizer_attrs.weight_scale_2)
    assert hasattr(model.linears[1], quantizer_attrs.input_quantizer)
    assert hasattr(model.linears[1], quantizer_attrs.input_scale)
    assert hasattr(model.linears[1], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[1], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[1], quantizer_attrs.output_scale)


def test_export_per_block_quantized_weight():
    model = ToyModel(dims=[32, 256, 256, 32])

    mtq.quantize(model, partial_w4a8_config, lambda x: x(torch.randn(1, 4, 32)))

    quantizer_attrs = quantizer_attr_names("weight")
    _export_quantized_weight(model.linears[2], torch.float32, "weight")
    assert model.linears[2].weight.dtype == torch.uint8
    assert hasattr(model.linears[2], quantizer_attrs.weight_quantizer)
    assert hasattr(model.linears[2], quantizer_attrs.weight_scale)
    assert hasattr(model.linears[2], quantizer_attrs.weight_scale_2)
    assert hasattr(model.linears[2], quantizer_attrs.input_scale)
    assert hasattr(model.linears[2], quantizer_attrs.input_quantizer)

    assert hasattr(model.linears[2], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[2], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[2], quantizer_attrs.output_scale)


@pytest.mark.parametrize(
    ("quant_cfg", "expected_algo", "has_input_scale"),
    [
        (mtq.NVFP4_DEFAULT_CFG, "NVFP4", True),
        (mtq.W4A16_NVFP4_CFG, "W4A16_NVFP4", False),
    ],
)
def test_pure_nvfp4_export_matches_module_export(
    quant_cfg,
    expected_algo,
    has_input_scale,
):
    module = nn.Linear(16, 2, bias=False)
    mtq.quantize(module, copy.deepcopy(quant_cfg), lambda m: m(torch.ones(1, 16)))
    with torch.no_grad():
        module.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(2, 16) / 8 - 2)
    module.weight_quantizer.amax = torch.tensor(2.0)
    if has_input_scale:
        module.input_quantizer.amax = torch.tensor(3.0)
    source_weight = module.weight.detach().clone()
    source_data_ptr = module.weight.data_ptr()
    source_parameters = tuple(module.parameters())
    source_buffers = {
        name: (buffer, buffer.detach().clone()) for name, buffer in module.named_buffers()
    }

    state = capture_quantized_weight_export_state(module)
    pure = export_quantized_weight(module.weight, state, dtype=torch.float32)
    repeated = export_quantized_weight(module.weight, state, dtype=torch.float32)

    expected_weight = torch.tensor(
        [
            [255, 239, 238, 222, 221, 204, 171, 154],
            [16, 34, 67, 84, 101, 102, 118, 119],
        ],
        dtype=torch.uint8,
    )
    expected_weight_scale = torch.tensor([[448.0], [416.0]], dtype=torch.float8_e4m3fn)
    assert torch.equal(pure.weight, expected_weight)
    assert torch.equal(pure.weight_scale, expected_weight_scale)
    assert pure.weight_scale_2 == pytest.approx(0.0007440476329065859)
    assert torch.equal(repeated.weight, pure.weight)
    assert torch.equal(repeated.weight_scale, pure.weight_scale)
    assert torch.equal(repeated.weight_scale_2, pure.weight_scale_2)
    if has_input_scale:
        assert pure.input_scale == pytest.approx(0.0011160714784637094)

    assert module.weight.data_ptr() == source_data_ptr
    assert torch.equal(module.weight, source_weight)
    assert all(
        parameter is source_parameter
        for parameter, source_parameter in zip(module.parameters(), source_parameters, strict=True)
    )
    assert set(dict(module.named_buffers())) == set(source_buffers)
    for name, buffer in module.named_buffers():
        source_buffer, source_value = source_buffers[name]
        assert buffer is source_buffer
        assert torch.equal(buffer, source_value)
    assert not hasattr(module, "weight_scale")
    assert not hasattr(module, "weight_scale_2")
    assert not hasattr(module, "input_scale")

    _export_quantized_weight(module, torch.float32)

    assert torch.equal(module.weight, pure.weight)
    assert torch.equal(module.weight_scale, pure.weight_scale)
    assert torch.equal(module.weight_scale_2, pure.weight_scale_2)
    assert hasattr(module, "input_scale") is has_input_scale
    if has_input_scale:
        assert torch.equal(module.input_scale, pure.input_scale)

    config = build_hf_quantization_config({"model.layers.0.mlp.up_proj": state})
    assert config["quant_algo"] == expected_algo
    assert config["config_groups"]["group_0"]["weights"]["group_size"] == state.block_size


def test_pure_nvfp4_capture_preserves_independent_expert_state():
    modules = [nn.Linear(16, 2, bias=False) for _ in range(2)]
    for module, weight_amax, input_amax in zip(
        modules,
        (2.0, 4.0),
        (3.0, 5.0),
        strict=True,
    ):
        mtq.quantize(
            module,
            copy.deepcopy(mtq.NVFP4_DEFAULT_CFG),
            lambda m: m(torch.ones(1, 16)),
        )
        module.weight_quantizer.amax = torch.tensor(weight_amax)
        module.input_quantizer.amax = torch.tensor(input_amax)

    grouped_module = nn.Module()
    grouped_module.weight0 = modules[0].weight
    grouped_module.weight1 = modules[1].weight
    grouped_module.weight_quantizer = GroupedQuantizer(
        modules[0].weight_quantizer,
        modules[1].weight_quantizer,
    )
    states = [
        capture_quantized_weight_export_state(
            grouped_module,
            f"weight{expert_idx}",
            weight_quantizer=grouped_module.weight_quantizer[expert_idx],
            input_quantizer=modules[expert_idx].input_quantizer,
        )
        for expert_idx in range(2)
    ]

    assert torch.equal(states[0].weight_amax, torch.tensor(2.0))
    assert torch.equal(states[1].weight_amax, torch.tensor(4.0))
    assert torch.equal(states[0].input_amax, torch.tensor(3.0))
    assert torch.equal(states[1].input_amax, torch.tensor(5.0))


def test_hf_quantization_config_supports_mixed_nvfp4():
    states = {}
    for name, quant_cfg in (
        ("model.layers.0.mlp.up_proj", mtq.NVFP4_DEFAULT_CFG),
        ("model.layers.1.mlp.up_proj", mtq.W4A16_NVFP4_CFG),
    ):
        module = nn.Linear(16, 2, bias=False)
        mtq.quantize(
            module,
            copy.deepcopy(quant_cfg),
            lambda m: m(torch.ones(1, 16)),
        )
        module.weight_quantizer.amax = torch.tensor(2.0)
        if module.input_quantizer.is_enabled:
            module.input_quantizer.amax = torch.tensor(3.0)
        states[name] = capture_quantized_weight_export_state(module)
    states["model.layers.0.self_attn.q_proj"] = None

    config = build_hf_quantization_config(states)

    assert config["quant_algo"] == "MIXED_PRECISION"
    assert len(config["config_groups"]) == 2
    assert config["ignore"] == ["model.layers.0.self_attn*"]


def test_pure_nvfp4_capture_rejects_disabled_weight_quantizer():
    module = nn.Linear(16, 2, bias=False)
    mtq.quantize(
        module,
        copy.deepcopy(mtq.NVFP4_DEFAULT_CFG),
        lambda m: m(torch.ones(1, 16)),
    )
    module.weight_quantizer.disable()

    with pytest.raises(RuntimeError, match="Missing calibrated weight amax"):
        capture_quantized_weight_export_state(module)


def test_pure_nvfp4_capture_rejects_deferred_nvfp4_format():
    module = nn.Linear(16, 2, bias=False)
    mtq.quantize(
        module,
        copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG),
        lambda m: m(torch.ones(1, 16)),
    )

    with pytest.raises(NotImplementedError, match="w4a8_nvfp4_fp8"):
        capture_quantized_weight_export_state(module)


class QuantMoELinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])

    def forward(self, x):
        return self.experts[0](x)


class _SingleRoutedExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = QuantMoELinear()

    def forward(self, x):
        return self.moe(x)


def test_process_quantized_modules_fills_step3p5_moe_input_scale_for_unrouted_experts():
    model = _SingleRoutedExpertModel()
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        ],
        "algorithm": "max",
    }

    mtq.quantize(model, quant_cfg, lambda m: m(torch.randn(2, 4, 8)))

    assert model.moe.experts[0].input_quantizer.amax is not None
    assert model.moe.experts[1].input_quantizer.amax is None

    _process_quantized_modules(model, torch.float32)

    assert hasattr(model.moe.experts[0], "input_scale")
    assert hasattr(model.moe.experts[1], "input_scale")
