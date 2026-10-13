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

import types

import torch
import torch.nn.functional as F
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_module import QuantLinearConvBase


def test_quant_input_base_ignores_forward_pre_dm_in_mro():
    """Regression test for recursion when `_forward_pre_dm` points to a wrapper forward in the MRO.

    In complex wrapper stacks, `_forward_pre_dm` may accidentally end up referencing a `forward`
    method already present in the quant wrapper MRO (e.g. QuantLinearConvBase.forward). If
    QuantInputBase.forward calls that directly, it can recurse indefinitely:

      QuantLinearConvBase.forward -> super().forward (QuantInputBase.forward)
        -> _forward_pre_dm (QuantLinearConvBase.forward) -> ...

    The fix is to detect this case and fall back to `super().forward` instead.
    """
    lin = nn.Linear(8, 8, bias=False)
    QuantModuleRegistry.convert(lin)

    # Force the problematic state: `_forward_pre_dm` points to a wrapper forward already in MRO.
    lin._forward_pre_dm = types.MethodType(QuantLinearConvBase.forward, lin)

    x = torch.randn(2, 8)
    y = lin(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 8)


def test_quantize_calibration_calls_quantizers_with_runtime_forward_patch():
    """Regression test for on-the-fly forward patching during mtq.quantize calibration.

    Some frameworks replace `module.forward` on-the-fly with a closure just before a forward pass.
    During mtq.quantize calibration, quantizers must still run (input + weight at minimum).
    """
    lin = nn.Linear(8, 8, bias=True).to(torch.float32)

    called = {"patched_forward": 0, "input_q": 0, "weight_q": 0}

    # Monkey patch instance-level forward (closure-style, no `self` argument).
    def patched_forward(x):
        called["patched_forward"] += 1
        # Use module parameters directly; if quantization wrappers are active, weight access
        # should still be routed through the quantized path.
        w = lin.weight.to(dtype=x.dtype)
        b = lin.bias.to(dtype=x.dtype) if lin.bias is not None else None
        return F.linear(x, w, b)

    def _count_input(_m, _inp, _out):
        called["input_q"] += 1

    def _count_weight(_m, _inp, _out):
        called["weight_q"] += 1

    lin.forward = patched_forward
    x = torch.randn(2, 8, dtype=torch.float16)

    def forward_loop(model):
        # Patch forward on-the-fly (after conversion, right before calibration forward).

        # Count quantizer executions during calibration.
        model.input_quantizer.register_forward_hook(_count_input)
        model.weight_quantizer.register_forward_hook(_count_weight)

        model(x)

    mtq.quantize(lin, mtq.INT8_DEFAULT_CFG, forward_loop)
    lin(x)

    assert called["patched_forward"] == 2
    assert called["input_q"] == 2
    assert called["weight_q"] == 2
