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

"""Tests for the ``should_process`` write-mask on the calibration algorithms."""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import awq_lite, gptq, max_calibrate, mse_calibrate
from modelopt.torch.quantization.nn import TensorQuantizer

QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 32}}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]


class _Model(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.keep = nn.Linear(d, d, bias=False)
        self.skip = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.skip(self.keep(x))


def _quantized():
    torch.manual_seed(0)
    return mtq.quantize(_Model().eval(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)


def _forward_loop(model):
    torch.manual_seed(1)
    for _ in range(2):
        model(torch.randn(2, 8, 32))


def _writable_state(model):
    """Everything a calibration algorithm may write: quantizer amax *and* module weights."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            amax = getattr(module, "_amax", None)
            state[f"amax:{name}"] = None if amax is None else module.amax.clone()
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            state[f"weight:{name}"] = module.weight.detach().clone()
        pqs = getattr(module, "pre_quant_scale", None)
        if pqs is not None:
            state[f"pqs:{name}"] = pqs.detach().clone()
    return state


def _only(model, prefix):
    """Write-mask admitting one subtree, keyed on identity exactly as the real one is."""
    admitted = {
        id(m)
        for name, m in model.named_modules()
        if name == prefix or name.startswith(prefix + ".")
    }
    return lambda module: id(module) in admitted


@pytest.mark.parametrize("algo", [max_calibrate, mse_calibrate, awq_lite, gptq])
def test_the_write_mask_confines_an_algorithm_to_its_scope(algo):
    model = _quantized()
    before = _writable_state(model)
    algo(model, forward_loop=_forward_loop, should_process=_only(model, "keep"))
    after = _writable_state(model)

    changed = {n for n in set(before) | set(after) if not _equal(before.get(n), after.get(n))}
    assert changed, f"{algo.__name__} wrote nothing at all -- the test would pass vacuously"
    outside = sorted(n for n in changed if not n.split(":", 1)[1].startswith("keep"))
    assert not outside, f"{algo.__name__} wrote outside its scope: {outside}"


def test_no_mask_means_the_whole_model():
    model = _quantized()
    max_calibrate(model, forward_loop=_forward_loop)
    calibrated = {
        n for n, v in _writable_state(model).items() if n.startswith("amax:") and v is not None
    }
    assert any("keep" in n for n in calibrated)
    assert any("skip" in n for n in calibrated)


def test_the_mask_confines_the_weight_only_path_too():
    # No forward loop routes through `weight_only_quantize`, a separate loop from the one the
    # forward-loop path uses; it needs its own guard.
    model = _quantized()
    before = _writable_state(model)
    max_calibrate(model, forward_loop=None, should_process=_only(model, "keep"))
    after = _writable_state(model)

    changed = {n for n in set(before) | set(after) if not _equal(before.get(n), after.get(n))}
    assert changed, "weight-only calibration wrote nothing -- the test would pass vacuously"
    outside = sorted(n for n in changed if not n.split(":", 1)[1].startswith("keep"))
    assert not outside, f"weight-only calibration wrote outside its scope: {outside}"


def test_the_mask_never_toggles_enable_state():
    model = _quantized()
    before = {
        n: bool(m.is_enabled) for n, m in model.named_modules() if isinstance(m, TensorQuantizer)
    }
    max_calibrate(model, forward_loop=_forward_loop, should_process=_only(model, "keep"))
    after = {
        n: bool(m.is_enabled) for n, m in model.named_modules() if isinstance(m, TensorQuantizer)
    }
    assert before == after


def _equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and torch.equal(a, b)
