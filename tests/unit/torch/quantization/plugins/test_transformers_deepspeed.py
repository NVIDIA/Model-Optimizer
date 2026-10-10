# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import types
from collections import OrderedDict

from torch import nn

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.transformers import make_deepspeed_compatible


class ZeROOrderedDict(OrderedDict):
    """Stand-in for the ZeRO-3 parameter dict in deepspeed.runtime.zero.parameter_offload."""


def _model_with_quantizer():
    return nn.Sequential(nn.Linear(4, 4), TensorQuantizer())


def test_quantizer_params_follow_zero3_once_deepspeed_is_imported(monkeypatch):
    parameter_offload = types.ModuleType("deepspeed.runtime.zero.parameter_offload")
    parameter_offload.ZeROOrderedDict = ZeROOrderedDict
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.zero.parameter_offload", parameter_offload)
    model = _model_with_quantizer()
    model[0]._parameters = ZeROOrderedDict(model[0]._parameters)

    make_deepspeed_compatible(model)
    assert isinstance(model[1]._parameters, ZeROOrderedDict)


def test_no_op_until_deepspeed_is_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "deepspeed.runtime.zero.parameter_offload", raising=False)
    model = _model_with_quantizer()
    make_deepspeed_compatible(model)
    assert type(model[1]._parameters) is dict
