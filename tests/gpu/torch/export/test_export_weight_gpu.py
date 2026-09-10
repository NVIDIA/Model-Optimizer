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
import math

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_w4a8_config
from torch.nn import functional as F
from torch.nn import init

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_config import QUANTIZATION_MXFP8
from modelopt.torch.export.quant_utils import (
    get_activation_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    postprocess_state_dict,
    to_quantized_weight,
)
from modelopt.torch.export.quantized_weight_export import (
    build_hf_quantization_config,
    capture_quantized_weight_export_state,
    export_quantized_weight_tensors,
    select_quantized_weight_export_state,
)
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule, QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from modelopt.torch.quantization.utils import quantizer_attr_names


class ToyLinear(nn.Module):
    in_features: int
    out_features: int
    toyweight: torch.Tensor  # intentionally not named weight

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.toyweight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.toyweight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.toyweight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class ToyModelLinear(torch.nn.Module):
    def __init__(self, dims=[10, 10, 10, 10]):
        super().__init__()
        assert len(dims) >= 2
        if len(dims) == 2:
            self.linears = ToyLinear(dims[0], dims[1])
        else:
            linears = [ToyLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            self.linears = torch.nn.Sequential(*linears)

    def forward(self, x):
        return self.linears(x)


@QuantModuleRegistry.register({ToyLinear: "ToyLinear"})
class _ToyLinearQuant(QuantModule):
    """Base class for modules where the input is quantized."""

    toyweight_input_quantizer: TensorQuantizer
    toyweight_weight_quantizer: TensorQuantizer
    toyweight_output_quantizer: TensorQuantizer
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    def forward(self, input, *args, **kwargs):
        """Quantize the input before calling the original forward method."""
        input = self.toyweight_input_quantizer(input)
        weight = self.toyweight_weight_quantizer(self.toyweight)
        output = F.linear(input, weight)
        return self.toyweight_output_quantizer(output)

    def _setup(self):
        """Patch the module's forward method to quantize the input."""
        self._register_temp_attribute(
            "toyweight_weight_quantizer", TensorQuantizer(self.default_quant_desc_weight)
        )
        self._register_temp_attribute(
            "toyweight_input_quantizer", TensorQuantizer(self.default_quant_desc_input)
        )
        self._register_temp_attribute(
            "toyweight_output_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self.toyweight_output_quantizer.disable()


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


@pytest.mark.parametrize("quant_cfg", [mtq.NVFP4_DEFAULT_CFG, mtq.W4A16_NVFP4_CFG])
def test_functional_nvfp4_export_matches_existing_helpers_without_mutation(quant_cfg):
    in_features = 256
    torch.manual_seed(0)
    module = nn.Linear(in_features, in_features, bias=False, device="cuda", dtype=torch.bfloat16)
    calib_input = torch.randn(2, 4, in_features, device="cuda", dtype=torch.bfloat16)
    module = mtq.quantize(module, copy.deepcopy(quant_cfg), lambda model: model(calib_input))
    original_weight = module.weight.detach().clone()
    original_buffers = {name: value.detach().clone() for name, value in module.named_buffers()}

    state = capture_quantized_weight_export_state(module)
    actual = export_quantized_weight_tensors(module.weight, state, torch.float16)

    quantization_format = get_quantization_format(module)
    weight_scale = get_weight_scaling_factor(module)
    weight_scale_2 = get_weight_scaling_factor_2(module)
    expected = {
        "weight": to_quantized_weight(
            module.weight.to(torch.float16),
            weight_scale,
            quantization_format,
            weight_scale_2,
            get_weight_block_size(module),
        ),
        "weight_scale": weight_scale,
        "weight_scale_2": weight_scale_2.squeeze(),
    }
    if module.input_quantizer.is_enabled:
        expected["input_scale"] = get_activation_scaling_factor(module).squeeze()

    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    torch.testing.assert_close(module.weight, original_weight)
    assert set(dict(module.named_buffers())) == set(original_buffers)
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, original_buffers[name])


@pytest.mark.parametrize(
    "quant_cfg",
    [
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
        mtq.MXFP8_DEFAULT_CFG,
        mtq.MXFP4_DEFAULT_CFG,
        mtq.W4A8_MXFP4_FP8_CFG,
        mtq.W4A8_NVFP4_FP8_CFG,
    ],
)
def test_functional_export_matches_existing_noninteger_helpers(quant_cfg):
    in_features = 256
    torch.manual_seed(0)
    module = nn.Linear(in_features, in_features, bias=False, device="cuda", dtype=torch.bfloat16)
    calib_input = torch.randn(2, 4, in_features, device="cuda", dtype=torch.bfloat16)
    module = mtq.quantize(module, copy.deepcopy(quant_cfg), lambda model: model(calib_input))
    original_weight = module.weight.detach().clone()
    original_buffers = {name: value.detach().clone() for name, value in module.named_buffers()}

    state = capture_quantized_weight_export_state(module)
    actual = export_quantized_weight_tensors(module.weight, state, torch.float16)

    quantization_format = get_quantization_format(module)
    weight_scale = get_weight_scaling_factor(module)
    weight_scale_2 = get_weight_scaling_factor_2(module)
    expected = {
        "weight": to_quantized_weight(
            module.weight.to(torch.float16),
            weight_scale,
            quantization_format,
            weight_scale_2,
            get_weight_block_size(module),
        ),
        "weight_scale": weight_scale,
    }
    if weight_scale_2 is not None:
        expected["weight_scale_2"] = weight_scale_2.squeeze()
    if module.input_quantizer.is_enabled and module.input_quantizer.amax is not None:
        expected["input_scale"] = get_activation_scaling_factor(module).squeeze()

    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    torch.testing.assert_close(module.weight, original_weight)
    assert set(dict(module.named_buffers())) == set(original_buffers)
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, original_buffers[name])

    _export_quantized_weight(module, torch.float16)

    for name, value in actual.items():
        torch.testing.assert_close(getattr(module, name), value, rtol=0, atol=0)
    if quantization_format == QUANTIZATION_MXFP8:
        assert hasattr(module, "weight_scale")
        assert not hasattr(module.weight_quantizer, "_scale")


def test_functional_mxfp8_preserves_cached_scale_during_selection():
    features = 256
    module = nn.Linear(features, features, bias=False, device="cuda", dtype=torch.bfloat16)
    calib_input = torch.randn(2, 4, features, device="cuda", dtype=torch.bfloat16)
    module = mtq.quantize(
        module,
        copy.deepcopy(mtq.MXFP8_DEFAULT_CFG),
        lambda model: model(calib_input),
    )
    cached_scale = get_weight_scaling_factor(module).clone()
    cached_scale[0].add_(1)
    module.weight_quantizer._scale = cached_scale

    indices = torch.arange(features // 2)
    state = select_quantized_weight_export_state(
        capture_quantized_weight_export_state(module),
        0,
        indices,
    )
    exported = export_quantized_weight_tensors(
        module.weight.index_select(0, indices.to(module.weight.device)),
        state,
        torch.float16,
    )

    torch.testing.assert_close(exported["weight_scale"], cached_scale[: features // 2])


def test_weight_derived_mxfp4_state_rejects_partial_block_selection():
    features = 256
    module = nn.Linear(features, features, bias=False, device="cuda", dtype=torch.bfloat16)
    calib_input = torch.randn(2, 4, features, device="cuda", dtype=torch.bfloat16)
    module = mtq.quantize(
        module,
        copy.deepcopy(mtq.MXFP4_DEFAULT_CFG),
        lambda model: model(calib_input),
    )

    with pytest.raises(ValueError, match="complete quantization blocks"):
        select_quantized_weight_export_state(
            capture_quantized_weight_export_state(module),
            1,
            (0,),
        )


def test_mixed_noninteger_states_build_one_canonical_config():
    features = 256
    states = {}
    for name, quant_cfg in (
        ("model.layers.0.self_attn.q_proj.weight", mtq.FP8_DEFAULT_CFG),
        ("model.layers.0.self_attn.k_proj.weight", mtq.MXFP8_DEFAULT_CFG),
        ("model.layers.0.mlp.down_proj.weight", mtq.NVFP4_DEFAULT_CFG),
    ):
        module = nn.Linear(features, features, bias=False, device="cuda", dtype=torch.bfloat16)
        calib_input = torch.randn(2, 4, features, device="cuda", dtype=torch.bfloat16)
        module = mtq.quantize(
            module,
            copy.deepcopy(quant_cfg),
            lambda model: model(calib_input),
        )
        states[name] = capture_quantized_weight_export_state(module)

    config = build_hf_quantization_config(states)

    assert config["quant_algo"] == "MIXED_PRECISION"
    assert set(config["quantized_layers"]) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.down_proj",
    }


def test_export_compressed_nvfp4_weight():
    """``mtq.compress`` (used by ``hf_ptq --low_memory_mode``) leaves the weight as packed NVFP4
    nibbles, so per-block scales cannot be recomputed from it. The export must reuse the scales
    stored on the quantizer and must not leak those internal buffers into the state_dict.
    """
    in_features, block_size = 256, 16
    calib = lambda x: x(torch.randn(1, 4, in_features).cuda())  # noqa: E731

    model = ToyModel(dims=[in_features, in_features, in_features, in_features]).cuda()
    reference = mtq.quantize(copy.deepcopy(model), mtq.NVFP4_DEFAULT_CFG, calib)
    compressed = mtq.quantize(copy.deepcopy(model), mtq.NVFP4_DEFAULT_CFG, calib)
    mtq.compress(compressed)

    quantizer_attrs = quantizer_attr_names("weight")
    ref_module, compressed_module = reference.linears[2], compressed.linears[2]
    _export_quantized_weight(ref_module, torch.float16, "weight")
    _export_quantized_weight(compressed_module, torch.float16, "weight")

    ref_scale = getattr(ref_module, quantizer_attrs.weight_scale)
    compressed_scale = getattr(compressed_module, quantizer_attrs.weight_scale)

    # Per-block scale covers the logical input dim, not the packed one.
    assert compressed_scale.shape == ref_scale.shape
    assert compressed_scale.shape[-1] == in_features // block_size

    # weight_scale * weight_scale_2 is what dequantization consumes; it must match the
    # uncompressed export rather than the compression-time normalization.
    ref_2 = getattr(ref_module, quantizer_attrs.weight_scale_2)
    compressed_2 = getattr(compressed_module, quantizer_attrs.weight_scale_2)
    assert torch.allclose(
        compressed_scale.float() * compressed_2.float(),
        ref_scale.float() * ref_2.float(),
        rtol=0.05,
    )

    # Internal compression buffers must be stripped by postprocess_state_dict, which keys off
    # RealQuantLinear.list_of_scale_tensors -- a missing underscore there let _double_scale leak.
    stripped = postprocess_state_dict(compressed.state_dict(), 1.0, None)
    assert not any(key.endswith("weight_quantizer._double_scale") for key in stripped)
    assert not any(key.endswith("weight_quantizer._scale") for key in stripped)
