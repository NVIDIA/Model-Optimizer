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


import dataclasses

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config, partial_w4a8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import postprocess_state_dict
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _process_quantized_modules,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.nn import TensorQuantizer
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


def _iq_linear(num_bits, in_features=256):
    linear = nn.Linear(in_features, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes={-1: 256},
            backend="ggml",
        )
    )
    return linear


@pytest.mark.parametrize(
    ("num_bits", "payload_bytes"),
    [("iq1_s", 50), ("iq1_m", 56), ("iq2_xxs", 66), ("iq2_xs", 74), ("iq2_s", 82)],
)
def test_export_iq_payload_as_weight(num_bits, payload_bytes):
    linear = _iq_linear(num_bits)

    _export_quantized_weight(linear, torch.bfloat16)
    state_dict = postprocess_state_dict(linear.state_dict(), maxbound=448, quantization=None)

    assert isinstance(linear.weight, nn.Parameter)
    assert state_dict["weight"].shape == (4, 1, payload_bytes)
    assert state_dict["weight"].dtype == torch.uint8
    assert "packed_weights" not in state_dict
    assert "weight_shape" not in state_dict


def _without_search(monkeypatch, num_bits):
    """Make the format's encoder fail, so any call proves export ran the search again."""

    def search(*args, **kwargs):
        raise AssertionError(f"export re-ran the {num_bits} search")

    record = dataclasses.replace(IQ_FORMAT_REGISTRY[num_bits], quantize=search)
    monkeypatch.setitem(IQ_FORMAT_REGISTRY, num_bits, record)


@pytest.mark.parametrize("num_bits", sorted(IQ_FORMAT_REGISTRY))
def test_export_reuses_the_payload_fake_quant_packed(monkeypatch, num_bits):
    """A weight fake quant already packed is exported from those bytes, not packed again.

    Fake quant sees the weight reshaped into 256-value blocks, so the cached payload is keyed on a
    different shape than the weight export holds; a wider weight keeps that difference real.
    """
    linear = _iq_linear(num_bits, in_features=512)
    linear.weight_quantizer(linear.weight)
    cache = linear.weight_quantizer._quantizer_cache
    assert cache.input_key.shape != tuple(linear.weight.shape)
    _without_search(monkeypatch, num_bits)

    _export_quantized_weight(linear, torch.bfloat16)

    assert torch.equal(linear.weight, cache.packed_weights.reshape(linear.weight.shape))
    assert linear.weight.shape[:2] == (4, 2)


@pytest.mark.parametrize("num_bits", sorted(IQ_FORMAT_REGISTRY))
def test_export_repacks_a_weight_changed_since_fake_quant(num_bits):
    """An in-place update after the forward invalidates the cached payload."""
    linear = _iq_linear(num_bits, in_features=512)
    linear.weight_quantizer(linear.weight)
    with torch.no_grad():
        linear.weight.mul_(-1)
    expected, _ = IQ_FORMAT_REGISTRY[num_bits].quantize(linear.weight.detach().clone())

    _export_quantized_weight(linear, torch.bfloat16)

    assert torch.equal(linear.weight, expected)


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
