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

"""Test quantized RNN modules for cuda."""

import copy

import pytest
import torch
import torch.nn as nn

from modelopt.torch.quantization.conversion import set_quantizer_attributes_partial
from modelopt.torch.quantization.nn import QuantModuleRegistry


@pytest.mark.parametrize(
    ("original_cls", "bidirectional", "bias"),
    [
        (nn.LSTM, True, True),
        (nn.LSTM, False, False),
    ],
)
def test_no_quant_proj(original_cls, bidirectional, bias):
    rnn_object = original_cls(
        8,
        8,
        2,
        bidirectional=bidirectional,
        bias=bias,
        proj_size=4,
    ).cuda()
    rnn_object_original = copy.deepcopy(rnn_object)
    quant_rnn_object = QuantModuleRegistry.convert(rnn_object)

    set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"enable": False})

    test_input = torch.randn((3, 2, 8), device="cuda")

    out1 = quant_rnn_object(test_input)[0]
    out2 = rnn_object_original(test_input)[0]
    assert torch.allclose(out1, out2, atol=1e-5)
