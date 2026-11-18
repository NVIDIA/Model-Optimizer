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
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer


class TestPrint:
    def test_print_descriptor(self):
        test_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
        print(test_desc)

    def test_print_tensor_quantizer(self):
        test_quantizer = TensorQuantizer()
        print(test_quantizer)

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
