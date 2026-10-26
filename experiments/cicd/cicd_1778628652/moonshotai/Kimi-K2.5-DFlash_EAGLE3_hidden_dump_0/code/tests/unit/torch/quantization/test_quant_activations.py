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

"""Tests of quantized activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization import set_quantizer_attributes_partial, tensor_quant
from modelopt.torch.quantization.nn import QuantModuleRegistry


class TestQuantLeakyReLU:
    def test_fake_quant_per_tensor(self):
        input_shape = (2, 2)
        negative_slope = 0.01
        leaky_relu_object = nn.LeakyReLU(negative_slope=negative_slope)
        quant_leaky_relu_object = QuantModuleRegistry.convert(leaky_relu_object)

        test_input = torch.randn(input_shape)
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = quant_leaky_relu_object(test_input)
        out2 = F.leaky_relu(quant_input, negative_slope)
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_fake_quant_per_channel(self):
        input_shape = (2, 2, 2)
        negative_slope = 0.01
        leaky_relu_object = nn.LeakyReLU(negative_slope=negative_slope)
        quant_leaky_relu_object = QuantModuleRegistry.convert(leaky_relu_object)
        set_quantizer_attributes_partial(quant_leaky_relu_object, lambda name: True, {"axis": (1)})

        test_input = torch.randn(input_shape)
        quant_input = tensor_quant.fake_tensor_quant(
            test_input, torch.abs(test_input).amax(dim=(0, 2), keepdim=True)
        )

        out1 = quant_leaky_relu_object(test_input)
        out2 = F.leaky_relu(quant_input, negative_slope)
        assert torch.allclose(out1, out2, rtol=0, atol=0)
