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
import dataclasses

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config, partial_w4a8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_format import QUANTIZATION_NONE
from modelopt.torch.export.quant_utils import get_quantization_format, postprocess_state_dict
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _process_quantized_modules,
    _upcast_ggml_weights,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight
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
            block_sizes={-1: GGML_FORMAT_REGISTRY[num_bits].block_size},
            backend="ggml",
        )
    )
    return linear


@pytest.mark.parametrize(
    ("num_bits", "block_size", "payload_bytes"),
    [
        ("iq1_s", 256, 50),
        ("iq1_m", 256, 56),
        ("iq2_xxs", 256, 66),
        ("iq2_xs", 256, 74),
        ("iq2_s", 256, 82),
        ("q8_0", 32, 34),
    ],
)
def test_export_iq_payload_as_weight(num_bits, block_size, payload_bytes):
    linear = _iq_linear(num_bits)

    _export_quantized_weight(linear, torch.bfloat16)
    state_dict = postprocess_state_dict(linear.state_dict(), maxbound=448, quantization=None)

    assert isinstance(linear.weight, nn.Parameter)
    assert state_dict["weight"].shape == (4, 256 // block_size, payload_bytes)
    assert state_dict["weight"].dtype == torch.uint8
    assert "packed_weights" not in state_dict
    assert "weight_shape" not in state_dict


def _without_search(monkeypatch, num_bits):
    """Make the format's encoder fail, so any call proves export ran the search again."""

    def search(*args, **kwargs):
        raise AssertionError(f"export re-ran the {num_bits} search")

    record = dataclasses.replace(GGML_FORMAT_REGISTRY[num_bits], quantize=search)
    monkeypatch.setitem(GGML_FORMAT_REGISTRY, num_bits, record)


@pytest.mark.parametrize("num_bits", sorted(GGML_FORMAT_REGISTRY))
def test_export_reuses_the_payload_fake_quant_packed(monkeypatch, num_bits):
    """A weight fake quant already packed is exported from those bytes, not packed again.

    Fake quant sees the weight reshaped into format-sized blocks, so the cached payload is keyed on a
    different shape than the weight export holds; a wider weight keeps that difference real.
    """
    linear = _iq_linear(num_bits, in_features=512)
    linear.weight_quantizer(linear.weight)
    cache = linear.weight_quantizer._quantizer_cache
    assert cache.input_key.shape != tuple(linear.weight.shape)
    _without_search(monkeypatch, num_bits)

    _export_quantized_weight(linear, torch.bfloat16)

    assert torch.equal(linear.weight, cache.packed_weights.reshape(linear.weight.shape))
    assert linear.weight.shape[:2] == (4, 512 // GGML_FORMAT_REGISTRY[num_bits].block_size)


@pytest.mark.parametrize("num_bits", sorted(GGML_FORMAT_REGISTRY))
def test_export_repacks_a_weight_changed_since_fake_quant(num_bits):
    """An in-place update after the forward invalidates the cached payload."""
    linear = _iq_linear(num_bits, in_features=512)
    linear.weight_quantizer(linear.weight)
    with torch.no_grad():
        linear.weight.mul_(-1)
    expected, _ = GGML_FORMAT_REGISTRY[num_bits].quantize(linear.weight.detach().clone())

    _export_quantized_weight(linear, torch.bfloat16)

    assert torch.equal(linear.weight, expected)


def test_export_drops_the_payload_gptq_pinned():
    """GPTQ pins its payload to the quantizer as a buffer; only the packed weight is exported."""
    linear = _iq_linear("q8_0")
    packed, _ = GGML_FORMAT_REGISTRY["q8_0"].quantize(linear.weight.detach())
    pin_packed_weight(linear.weight_quantizer, "q8_0", packed)

    _export_quantized_weight(linear, torch.bfloat16)
    state_dict = postprocess_state_dict(linear.state_dict(), maxbound=448, quantization=None)

    assert list(state_dict) == ["weight"]


def _ggml_model(num_bits, dtype=torch.bfloat16, input_cfg=None):
    """A one-linear model whose weight quantizer is ``num_bits``; ``input_cfg`` enables its input."""
    model = nn.Sequential(nn.Linear(512, 4, bias=False, dtype=dtype))
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": num_bits, "backend": "ggml"},
            "enable": True,
        },
    ]
    if input_cfg is not None:
        quant_cfg.append({"quantizer_name": "*input_quantizer", "cfg": input_cfg, "enable": True})
    return mtq.quantize(model, {"quant_cfg": quant_cfg, "algorithm": None})


@pytest.mark.parametrize("num_bits", sorted(GGML_FORMAT_REGISTRY))
def test_upcast_ggml_writes_what_the_packed_export_decodes_to(num_bits):
    """The upcast weight is exactly the decoding of the payload the packed export writes."""
    model = _ggml_model(num_bits)
    packed = copy.deepcopy(model[0])
    _export_quantized_weight(packed, torch.bfloat16)

    _upcast_ggml_weights(model, torch.bfloat16)

    expected = GGML_FORMAT_REGISTRY[num_bits].dequantize(
        packed.weight, torch.tensor((4, 512)), dtype=torch.bfloat16
    )
    assert torch.equal(model[0].weight, expected)
    # The layer now exports as an unquantized one, so it stays out of the quantization config.
    assert get_quantization_format(model) == QUANTIZATION_NONE


@pytest.mark.parametrize(
    ("dtype", "input_cfg", "error"),
    [
        (torch.bfloat16, {"num_bits": (4, 3)}, "enabled input quantizer"),
        (torch.bfloat16, "pre_quant_scale", "pre_quant_scale"),
        (torch.float16, None, "stored as torch.float16"),
        (torch.float32, None, "stored as torch.float32"),
    ],
    ids=["input_quantizer", "pre_quant_scale", "fp16_weight", "fp32_weight"],
)
def test_upcast_ggml_refuses_a_layer_before_changing_any(dtype, input_cfg, error):
    """Activation quantization the upcast would drop, or a weight not stored in BF16, is refused.

    The packed export refuses the first two too; disabling the quantizer must not skip that. The
    model is left as it was, so a refusal never leaves it half upcast.
    """
    model = _ggml_model("q8_0", dtype, None if input_cfg == "pre_quant_scale" else input_cfg)
    if input_cfg == "pre_quant_scale":
        model[0].input_quantizer.pre_quant_scale = torch.ones(512)
    weight = model[0].weight.detach().clone()

    with pytest.raises((NotImplementedError, ValueError), match=error):
        _upcast_ggml_weights(model, torch.bfloat16)

    assert torch.equal(model[0].weight, weight)
    assert model[0].weight_quantizer.is_enabled


@pytest.mark.parametrize("pinned", [False, True])
def test_upcast_ggml_writes_back_an_offloaded_weight(monkeypatch, pinned):
    """The decoded values reach the offloaded copy; a payload GPTQ pinned is decoded, not redone."""
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device

    model = _ggml_model("q8_0")
    linear = model[0]
    q8_0 = GGML_FORMAT_REGISTRY["q8_0"]
    payload, shape = q8_0.quantize(linear.weight.detach())
    expected = q8_0.dequantize(payload, shape, dtype=torch.bfloat16)
    assert not torch.equal(linear.weight, expected)
    if pinned:
        # How GPTQ leaves a layer: its payload pinned, and the weight equal to the decoding.
        with torch.no_grad():
            linear.weight.copy_(expected)
        pin_packed_weight(linear.weight_quantizer, "q8_0", payload)
        _without_search(monkeypatch, "q8_0")
    weights_map = {"weight": linear.weight.detach().clone()}
    add_hook_to_module(
        linear, AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map)
    )
    set_module_tensor_to_device(linear, "weight", "meta")

    _upcast_ggml_weights(model, torch.bfloat16)

    assert linear.weight.is_meta
    assert torch.equal(weights_map["weight"], expected)


def test_upcast_ggml_warns_when_there_is_nothing_to_upcast():
    """Asking for an upcast of a model with no GGML weights is most likely a mistake."""
    with pytest.warns(UserWarning, match="no GGML-quantized weights"):
        _upcast_ggml_weights(nn.Sequential(nn.Linear(4, 4)), torch.bfloat16)


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
