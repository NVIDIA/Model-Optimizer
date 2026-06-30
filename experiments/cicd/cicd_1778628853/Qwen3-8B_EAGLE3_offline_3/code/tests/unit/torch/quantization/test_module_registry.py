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

import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleConv

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import QuantModuleRegistry
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer


class QuantReLU(nn.ReLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        self.input_quantizer = TensorQuantizer(QuantizerAttributeConfig())
        self.output_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        # NOTE: If needed, quantized weights by adding weight quantizer

    def forward(self, x):
        return self.output_quantizer(super().forward(self.input_quantizer(x)))


def test_register_custom_module():
    model = SimpleConv()

    num_old_modules = sum(type(m) is nn.ReLU for m in model.modules())

    mtq.register(original_cls=nn.ReLU, quantized_cls=QuantReLU)
    mtq.replace_quant_module(model)

    model(model.get_input())

    assert sum(isinstance(m, QuantReLU) for m in model.modules()) == num_old_modules

    mtq.unregister(nn.ReLU)

    assert QuantModuleRegistry.get(nn.ReLU) is None
