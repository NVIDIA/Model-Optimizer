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

import copy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.export.quantized_weight_export as weight_export
import modelopt.torch.quantization as mtq
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.quantized_weight_export import (
    capture_quantized_weight_export_state,
    get_quantized_weight_export_spec,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import GroupedQuantizer, NVFP4StaticQuantizer, TensorQuantizer


def _fp8_linear() -> nn.Linear:
    module = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8 - 1)
    return mtq.quantize(module, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m(torch.ones(2, 4)))


def _static_w4a16_linear(
    weight: torch.Tensor,
    per_block_amax: torch.Tensor,
    global_amax: torch.Tensor,
) -> nn.Linear:
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    module.weight.data.copy_(weight)
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
    )
    quantizer = NVFP4StaticQuantizer(quant_attribute_cfg=cfg)
    quantizer.amax = per_block_amax.clone()
    quantizer.global_amax = global_amax.clone()
    module.weight_quantizer = quantizer
    module.input_quantizer = TensorQuantizer()
    module.input_quantizer.disable()
    return module


class _SquareTransposedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.arange(512, dtype=torch.float32).reshape(2, 16, 16))
        self.down_proj = nn.Parameter(torch.arange(512, dtype=torch.float32).reshape(2, 16, 16))

        fp8_cfg = QuantizerAttributeConfig(num_bits=(4, 3))
        self.gate_up_proj_weight_quantizer = TensorQuantizer(fp8_cfg)
        self.gate_up_proj_weight_quantizer._amax = torch.tensor(2.0)
        self.gate_up_proj_input_quantizer = TensorQuantizer(fp8_cfg)
        self.gate_up_proj_input_quantizer._amax = torch.tensor(1.0)

        nvfp4_cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
        )
        self.down_proj_weight_quantizer = NVFP4StaticQuantizer(quant_attribute_cfg=nvfp4_cfg)
        self.down_proj_weight_quantizer.amax = torch.arange(32, dtype=torch.float32) + 1
        self.down_proj_weight_quantizer.global_amax = torch.tensor(32.0)
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer.disable()

    def iter_weights_for_calibration(self):
        for name in ("gate_up_proj", "down_proj"):
            yield getattr(self, name).transpose(-1, -2), getattr(self, f"{name}_weight_quantizer")


class _GroupedWeights(nn.Module):
    """Minimal TEGroupedLinear-style numbered-weight layout."""

    def __init__(self):
        super().__init__()
        self.weight0 = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        self.weight1 = nn.Parameter(torch.arange(16, 32, dtype=torch.float32).reshape(4, 4))
        cfg = QuantizerAttributeConfig(num_bits=(4, 3))
        self.quantizers = nn.ModuleList([TensorQuantizer(cfg), TensorQuantizer(cfg)])
        for index, quantizer in enumerate(self.quantizers, start=1):
            quantizer._amax = torch.tensor(float(index))
        self.input_quantizer = TensorQuantizer()
        self.input_quantizer.disable()

    def iter_weights_for_calibration(self):
        yield self.weight0, self.quantizers[0]
        yield self.weight1, self.quantizers[1]


def test_capture_does_not_modify_zero_amax():
    module = _fp8_linear()
    module.weight_quantizer._amax.zero_()
    module.input_quantizer._amax.zero_()
    weight_amax = module.weight_quantizer._amax.clone()
    input_amax = module.input_quantizer._amax.clone()

    capture_quantized_weight_export_state(module)

    torch.testing.assert_close(module.weight_quantizer._amax, weight_amax)
    torch.testing.assert_close(module.input_quantizer._amax, input_amax)


def test_capture_resolves_numbered_grouped_weights_by_storage():
    module = _GroupedWeights()

    state0 = capture_quantized_weight_export_state(module, "weight0")
    state1 = capture_quantized_weight_export_state(module, "weight1")

    assert state0 is not None and state1 is not None
    assert state0.quantization_format == state1.quantization_format == "fp8"
    assert state0.tensors[0].value.item() != state1.tensors[0].value.item()


@pytest.mark.parametrize("has_iterator", [True, False])
def test_capture_rejects_unresolved_numbered_weight(has_iterator):
    module = _GroupedWeights()
    module.weight_quantizer = GroupedQuantizer(*module.quantizers)
    module.iter_weights_for_calibration = (
        (lambda: iter(((module.weight0, module.quantizers[0]),))) if has_iterator else None
    )

    with pytest.raises(NotImplementedError, match="weight1"):
        capture_quantized_weight_export_state(module, "weight1")


def test_capture_rejects_full_fused_weight_with_per_expert_quantizers():
    module = _GroupedWeights()
    module.weight = nn.Parameter(torch.stack((module.weight0, module.weight1)))
    module.weight_quantizer = GroupedQuantizer(*module.quantizers)
    module.iter_weights_for_calibration = lambda: (
        (module.weight[index], quantizer) for index, quantizer in enumerate(module.quantizers)
    )

    with pytest.raises(NotImplementedError, match="matching full weight view"):
        capture_quantized_weight_export_state(module)


@pytest.mark.parametrize("parameter_name", ["bias", "layer_norm_weight"])
def test_unquantized_sibling_has_no_export_state_or_spec(parameter_name):
    module = _fp8_linear()
    module.register_parameter(parameter_name, nn.Parameter(torch.ones(4)))

    assert capture_quantized_weight_export_state(module, parameter_name) is None
    assert get_quantized_weight_export_spec(module, parameter_name) is None


def test_unquantized_weight_has_no_export_state_or_spec():
    module = nn.Linear(4, 4, bias=False)

    assert capture_quantized_weight_export_state(module) is None
    assert get_quantized_weight_export_spec(module) is None


@pytest.mark.parametrize(
    ("features", "quantizer_cfg", "error"),
    [
        pytest.param(4, {"num_bits": 8}, "int8_wo", id="int8-weight-only"),
        pytest.param(
            32,
            {
                "num_bits": (2, 1),
                "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            },
            "weight-only MXFP4",
            id="mxfp4-weight-only",
        ),
        pytest.param(
            128,
            {"num_bits": (4, 3), "block_sizes": {-2: 64, -1: 128}},
            "only with 128x128 blocks",
            id="fp8-rectangular-blocks",
        ),
    ],
)
def test_export_spec_rejects_unsupported_quantization(features, quantizer_cfg, error):
    module = nn.Linear(features, features, bias=False)
    module.weight_quantizer = TensorQuantizer(QuantizerAttributeConfig(**quantizer_cfg))
    module.input_quantizer = TensorQuantizer()
    module.input_quantizer.disable()

    with pytest.raises(NotImplementedError, match=error):
        get_quantized_weight_export_spec(module)


@pytest.mark.parametrize(
    ("quantization_format", "block_size", "input_enabled"),
    [("fp8", 0, True), ("fp8", 0, False), ("w4a16_nvfp4", 16, False)],
)
def test_export_spec_does_not_materialize_tensor_state(
    quantization_format, block_size, input_enabled, monkeypatch
):
    if quantization_format == "fp8":
        module = _fp8_linear()
    else:
        weight = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32
        module = _static_w4a16_linear(
            weight, weight.abs().amax(dim=1, keepdim=True), torch.tensor(1.0)
        )
    monkeypatch.setattr(
        weight_export,
        "_state_tensor",
        lambda *args, **kwargs: pytest.fail("export spec materialized tensor state"),
    )

    spec = get_quantized_weight_export_spec(module)

    assert spec is not None
    assert spec.quantization_format == quantization_format
    assert spec.block_size == block_size
    if not input_enabled and module.input_quantizer.is_enabled:
        module.input_quantizer.disable()
        updated_spec = get_quantized_weight_export_spec(module)
        assert updated_spec != spec
        assert updated_spec.quantization_format == spec.quantization_format
        assert updated_spec.block_size == spec.block_size
        spec = updated_spec
    assert spec.input_quantizer_enabled is input_enabled


def test_capture_uses_the_requested_weight_format_and_transposed_layout():
    module = _SquareTransposedExperts()
    gate_state = capture_quantized_weight_export_state(module, "gate_up_proj")
    down_state = capture_quantized_weight_export_state(module, "down_proj")

    assert gate_state.quantization_format == "fp8"
    assert down_state.quantization_format == "w4a16_nvfp4"
    assert gate_state.weight_shape == down_state.weight_shape == (2, 16, 16)
    assert gate_state.packing_permutation == down_state.packing_permutation == (0, 2, 1)


def test_converted_hf_config_preserves_nvfp4_group_size():
    config = convert_hf_quant_config_format(
        {
            "quantization": {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 32,
            }
        }
    )

    assert config["group_size"] == 32
    assert config["config_groups"]["group_0"]["weights"]["group_size"] == 32
