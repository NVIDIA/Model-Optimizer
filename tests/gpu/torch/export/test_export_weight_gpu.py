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
from _test_utils.torch.export.utils import ToyModel, partial_w4a8_config
from torch.nn import functional as F
from torch.nn import init

import modelopt.torch.quantization as mtq
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
