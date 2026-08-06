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

"""Tests of tensor quantizer."""

import pytest
import torch
from _test_utils.torch.quantization.tensor_quantizer_common import (
    BlockQuantTester,
    SequentialQuantizerTester,
    TensorQuantizerTester,
)

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer


class TestTensorQuantizerCPU(TensorQuantizerTester):
    device = "cpu"

    def test_disabled_extra_repr(self):
        quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False)).to(self.device)
        assert quantizer.extra_repr() == "disabled"

        quantizer.pre_quant_scale = torch.tensor([0.5, 2.0]).to(self.device)
        extra_repr = quantizer.extra_repr()
        assert extra_repr.startswith("disabled")
        assert "pre_quant_scale=" in extra_repr

    def test_dynamic_amax_set_rejected(self):
        quantizer = TensorQuantizer(QuantizerAttributeConfig(type="dynamic")).to(self.device)
        with pytest.raises(AssertionError, match="Dynamic quantization"):
            quantizer.amax = 1.0


class TestBlockQuantCPU(BlockQuantTester):
    device = "cpu"


class TestSequentialQuantizerCPU(SequentialQuantizerTester):
    device = "cpu"
