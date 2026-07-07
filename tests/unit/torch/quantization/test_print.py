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

"""Test for str and repr."""

import torch
from torch import nn

from modelopt.torch.quantization import calib, tensor_quant
from modelopt.torch.quantization import nn as qnn
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer


class TestPrint:
    def test_print_descriptor(self):
        test_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
        print(test_desc)

    def test_print_tensor_quantizer(self):
        test_quantizer = TensorQuantizer()
        print(test_quantizer)

    def test_constant_amax_tensor_quantizer_repr(self):
        test_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(num_bits=(4, 3), use_constant_amax=True)
        )

        assert "amax=4.48e+02(const)" in repr(test_quantizer)

    def test_disabled_tensor_quantizer_repr_shows_enabled_state(self):
        test_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                enable=False,
                rotate={
                    "enable": True,
                    "mode": "rotate_back",
                    "rotate_fp32": True,
                    "block_size": 8,
                },
            )
        )
        test_quantizer.pre_quant_scale = torch.tensor([1.0, 2.0])

        assert test_quantizer.extra_repr() == (
            "disabled pre_quant_scale=[1.00e+00, 2.00e+00](2)"
            " rotated (rotate_back) (fp32) (block=8)"
        )

    def test_print_module(self):
        class _TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(33, 65, 3)
                self.quant_conv = qnn.Conv2d(33, 65, 3)
                self.linear = nn.Linear(33, 65)
                self.quant_linear = qnn.Linear(33, 65)

        test_module = _TestModule()
        print(test_module)

    def test_print_calibrator(self):
        print(calib.MaxCalibrator(7, 1, False))
        hist_calibrator = calib.HistogramCalibrator(8, None, True)
        hist_calibrator.collect(torch.rand(10))
        print(hist_calibrator)
