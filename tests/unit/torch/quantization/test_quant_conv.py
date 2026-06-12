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

"""Tests of QuantConv module."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules import quant_conv
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

NUM_IN_CHANNELS = 3
NUM_OUT_CHANNELS = 5


class TestQuantConvND:
    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_no_quant(self, conv_cls, f_conv, input_shape):
        kernel_size = 8

        quant_conv_object = conv_cls(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size, bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(input_shape)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = f_conv(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_weight_fake_quant_per_tensor(self, conv_cls, f_conv, input_shape):
        kernel_size = 8

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(),
        )
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(input_shape)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, weight_copy.abs().amax())

        out1 = f_conv(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_weight_fake_quant_per_channel(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(axis=(0)),
        )
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(input_shape)

        weight_copy = quant_conv_object.weight.clone()
        amax = weight_copy.abs().amax(dim=(1, 2), keepdim=True)
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = f_conv(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_input(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size, bias=False)
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = f_conv(quant_input, quant_conv_object.weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_tensor(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(),
        )
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, weight_copy.abs().amax())

        out1 = f_conv(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(axis=(0)),
        )
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy, weight_copy.abs().amax(dim=list(range(1, len(input_shape))), keepdim=True)
        )

        out1 = f_conv(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel_other_prec(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_desc_input = QuantizerAttributeConfig(num_bits=4)
        quant_desc_weight = QuantizerAttributeConfig(num_bits=3, axis=(0))

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight,
        )
        test_input = torch.randn(input_shape)

        test_input_quantizer = TensorQuantizer(quant_desc_input)
        weight_quantizer = TensorQuantizer(quant_desc_weight)

        quant_input = test_input_quantizer(test_input)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_quantizer(weight_copy)

        out1 = f_conv(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel_bias(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=QuantizerAttributeConfig(axis=(0)),
        )
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy, weight_copy.abs().amax(dim=list(range(1, len(input_shape))), keepdim=True)
        )

        out1 = f_conv(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "nn_conv_cls", "input_shape"),
        [
            (quant_conv.QuantConv1d, nn.Conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, nn.ConvTranspose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, nn.Conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, nn.ConvTranspose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, nn.Conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, nn.ConvTranspose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_against_unquantized(self, conv_cls, nn_conv_cls, input_shape):
        kernel_size = 3
        test_input = torch.randn(input_shape)

        quant_conv = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantizerAttributeConfig(num_bits=16),
            quant_desc_weight=QuantizerAttributeConfig(num_bits=16, axis=(0)),
        )

        conv = nn_conv_cls(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size, bias=True)
        conv.load_state_dict(quant_conv.state_dict())

        quant_conv.input_quantizer.disable()
        quant_conv.weight_quantizer.disable()

        quant_output = quant_conv(test_input)
        output = conv(test_input)

        assert torch.allclose(quant_output, output)
