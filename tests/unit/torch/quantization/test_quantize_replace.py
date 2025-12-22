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

"""Tests for modelopt.torch.quantization.quantize level APIs."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import (
    SDPAAttention,
    SimpleConv,
    SimpleConvLinear,
    SimpleLinear,
)

import modelopt.torch.quantization as mtq


def _is_quantized_linear_conv(module):
    return hasattr(module, "weight_quantizer") and hasattr(module, "input_quantizer")


@pytest.mark.parametrize("model_cls", [SimpleLinear, SimpleConv, SimpleConvLinear, SDPAAttention])
def test_quantize_replace(model_cls):
    if model_cls == SDPAAttention:
        mtq.plugins.register_attention_for_kv_quant(SDPAAttention)
    model_ref = model_cls()
    model_atq = model_cls()
    model_atq.load_state_dict(model_ref.state_dict())
    dummy_input = model_cls.get_input()

    mtq.replace_quant_module(model_atq)

    for name, module in model_atq.named_modules():
        assert not isinstance(module, nn.Conv2d) or _is_quantized_linear_conv(module)
        assert not isinstance(module, nn.Linear) or _is_quantized_linear_conv(module)

    mtq.set_quantizer_attribute(model_atq, "*", {"enable": False})

    out_ref = model_ref(dummy_input)
    out_atq = model_atq(dummy_input)

    assert torch.allclose(out_ref, out_atq)

    if model_cls == SDPAAttention:
        mtq.unregister(SDPAAttention)
