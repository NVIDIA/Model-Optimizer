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

"""Tests of QuantPooling module."""

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules import quant_pooling


class TestQuantMaxPool:
    @pytest.mark.parametrize(
        ("pool_cls", "f_pool", "input_shape"),
        [
            (quant_pooling.QuantMaxPool1d, F.max_pool1d, (2, 5, 5)),
            (quant_pooling.QuantMaxPool2d, F.max_pool2d, (2, 5, 5, 5)),
            (quant_pooling.QuantMaxPool3d, F.max_pool3d, (2, 5, 5, 5, 5)),
        ],
    )
    def test_input_fake_quant(self, pool_cls, f_pool, input_shape):
        quant_pooling_object = pool_cls(kernel_size=3, stride=1)

        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(
            test_input, torch.max(torch.abs(test_input)), None
        )

        out1 = f_pool(quant_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_input_variable_bits(self):
        # Repeat checking the output for variable number of bits to QuantizerAttributeConfig
        for bits in [2, 4, 6]:
            quant_desc_input = QuantizerAttributeConfig(num_bits=bits)

            quant_pooling.QuantMaxPool2d.default_quant_desc_input = quant_desc_input
            quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

            test_input = torch.randn(1, 5, 5, 5)

            quant_input = tensor_quant.fake_tensor_quant(
                test_input, torch.max(torch.abs(test_input)), None, bits
            )

            out1 = F.max_pool2d(quant_input, 3, 1, 0, 1, False, False)
            out2 = quant_pooling_object(test_input)
            assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_input_fake_quant_disable(self):
        quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_pooling_object.input_quantizer.disable()

        out1 = F.max_pool2d(test_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_input_multi_axis(self):
        quant_desc_input = QuantizerAttributeConfig(num_bits=8, axis=(0, 1))

        quant_pooling.QuantMaxPool2d.default_quant_desc_input = quant_desc_input
        quant_pooling_object = quant_pooling.QuantMaxPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(2, 7, 5, 5, dtype=torch.double)
        input_amax = torch.amax(torch.abs(test_input), dim=(2, 3), keepdim=True)
        quant_input = tensor_quant.fake_tensor_quant(test_input, input_amax, None)

        out1 = F.max_pool2d(quant_input, 3, 1, 0, 1, False, False)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)


class TestQuantAvgPoolNd:
    @pytest.mark.parametrize(
        ("pool_cls", "f_pool", "input_shape"),
        [
            (quant_pooling.QuantAvgPool1d, F.avg_pool1d, (2, 5, 5)),
            (quant_pooling.QuantAvgPool2d, F.avg_pool2d, (2, 5, 5, 5)),
            (quant_pooling.QuantAvgPool3d, F.avg_pool3d, (2, 5, 5, 5, 5)),
        ],
    )
    def test_input_fake_quant(self, pool_cls, f_pool, input_shape):
        quant_pooling_object = pool_cls(kernel_size=3, stride=1)

        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(
            test_input, torch.max(torch.abs(test_input)), None
        )

        out1 = f_pool(quant_input, 3, 1, 0, False, True)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    @pytest.mark.parametrize("bits", [2, 4, 6])
    def test_input_variable_bits(self, bits):
        # Repeat checking the output for variable number of bits to QuantizerAttributeConfig
        quant_desc_input = QuantizerAttributeConfig(num_bits=bits)

        quant_pooling.QuantAvgPool2d.default_quant_desc_input = quant_desc_input
        quant_pooling_object = quant_pooling.QuantAvgPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_input = tensor_quant.fake_tensor_quant(
            test_input, torch.max(torch.abs(test_input)), None, bits
        )

        out1 = F.avg_pool2d(quant_input, 3, 1, 0, False, True, None)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_input_fake_quant_disable(self):
        quant_pooling_object = quant_pooling.QuantAvgPool2d(kernel_size=3, stride=1)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_pooling_object.input_quantizer.disable()

        out1 = F.avg_pool2d(test_input, 3, 1, 0, False, True, None)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)


class TestQuantAdaptiveAvgPoolNd:
    @pytest.mark.parametrize(
        ("pool_cls", "f_pool", "input_shape"),
        [
            (quant_pooling.QuantAdaptiveAvgPool1d, F.adaptive_avg_pool1d, (2, 5, 5)),
            (quant_pooling.QuantAdaptiveAvgPool2d, F.adaptive_avg_pool2d, (2, 5, 5, 5)),
            (quant_pooling.QuantAdaptiveAvgPool3d, F.adaptive_avg_pool3d, (2, 5, 5, 5, 5)),
        ],
    )
    def test_input_fake_quant(self, pool_cls, f_pool, input_shape):
        quant_pooling_object = pool_cls(output_size=3)

        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(
            test_input, torch.max(torch.abs(test_input)), None
        )

        out1 = f_pool(quant_input, 3)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_input_variable_bits(self):
        # Repeat checking the output for variable number of bits to QuantizerAttributeConfig
        for bits in [2, 4, 6]:
            quant_desc_input = QuantizerAttributeConfig(num_bits=bits)

            quant_pooling.QuantAdaptiveAvgPool2d.default_quant_desc_input = quant_desc_input
            quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool2d(output_size=3)

            test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

            quant_input = tensor_quant.fake_tensor_quant(
                test_input, torch.max(torch.abs(test_input)), None, bits
            )

            out1 = F.adaptive_avg_pool2d(quant_input, 3)
            out2 = quant_pooling_object(test_input)
            assert torch.allclose(out1, out2, rtol=0, atol=0)

    def test_input_fake_quant_disable(self):
        quant_pooling_object = quant_pooling.QuantAdaptiveAvgPool2d(output_size=3)

        test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

        quant_pooling_object.input_quantizer.disable()

        out1 = F.adaptive_avg_pool2d(test_input, 3)
        out2 = quant_pooling_object(test_input)
        assert torch.allclose(out1, out2, rtol=0, atol=0)
