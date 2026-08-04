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

import math

import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_nvfp4_config, partial_w4a8_config
from torch.nn import functional as F
from torch.nn import init

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule, QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from modelopt.torch.quantization.utils import quantizer_attr_names, reduce_block_amax


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


def test_export_nvfp4_static_weight_dynamic_vs_static_match():
    """Dynamic vs static NVFP4 export: same weight and scales after export even when amaxs are
    cleared on one layer (lazy calibration via _ensure_weight_quantizer_calibrated fills them from weights).
    """
    device = "cuda"
    dims = [32, 32, 32, 32]
    block_size = 16
    calib_input = torch.randn(1, 4, 32, device=device)
    nvfp4_layer_indices = [1, 2]  # layers with NVFP4 enabled in partial_nvfp4_config

    torch.manual_seed(42)
    model_dynamic = ToyModel(dims=dims).to(device)
    mtq.quantize(model_dynamic, partial_nvfp4_config, lambda x: x(calib_input))

    torch.manual_seed(42)
    model_static = ToyModel(dims=dims).to(device)
    mtq.quantize(model_static, partial_nvfp4_config, lambda x: x(calib_input))

    # Convert NVFP4 layers to NVFP4StaticQuantizer with per-block and global amax
    for idx in nvfp4_layer_indices:
        layer = model_static.linears[idx]
        weight = layer.weight.data
        per_block_amax = reduce_block_amax(weight, block_sizes={-1: block_size})
        tq = layer.weight_quantizer
        if hasattr(tq, "_amax"):
            delattr(tq, "_amax")
        tq.register_buffer("_amax", per_block_amax.to(weight.device).clone().detach())
        NVFP4StaticQuantizer.from_tensor_quantizer(tq, global_amax=per_block_amax.max())

    # Clear amaxs on layer 1 to exercise lazy calibration during export
    for linear, is_static in [(model_dynamic.linears[1], False), (model_static.linears[1], True)]:
        wq = linear.weight_quantizer
        if hasattr(wq, "_amax"):
            delattr(wq, "_amax")
        if is_static and hasattr(wq, "_global_amax"):
            delattr(wq, "_global_amax")

    quantizer_attrs = quantizer_attr_names("weight")
    for idx in nvfp4_layer_indices:
        _export_quantized_weight(model_dynamic.linears[idx], torch.float32, "weight")
        _export_quantized_weight(model_static.linears[idx], torch.float32, "weight")

    for idx in nvfp4_layer_indices:
        dyn_linear = model_dynamic.linears[idx]
        sta_linear = model_static.linears[idx]
        assert torch.equal(dyn_linear.weight, sta_linear.weight), (
            f"Layer {idx}: exported NVFP4 weight should match (dynamic vs static)"
        )
        assert torch.allclose(
            getattr(dyn_linear, quantizer_attrs.weight_scale).float(),
            getattr(sta_linear, quantizer_attrs.weight_scale).float(),
        ), f"Layer {idx}: weight_scale should match"
        assert torch.allclose(
            getattr(dyn_linear, quantizer_attrs.weight_scale_2).float(),
            getattr(sta_linear, quantizer_attrs.weight_scale_2).float(),
        ), f"Layer {idx}: weight_scale_2 should match"
