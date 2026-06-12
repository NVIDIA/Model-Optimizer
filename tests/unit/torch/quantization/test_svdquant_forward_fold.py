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

"""SVDQuant forward / fold coverage.

These tests protect the invariants the diffusers SVDQuant export relies on: the
LoRA factors stay on the ``weight_quantizer`` in the live model and the export
layer promotes them. They complement (and do not modify) the existing
``test_calib.py::test_svdquant_lora_weights``.
"""

from functools import partial

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq


class _SVDMLP(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _forward_loop(model, dataloader):
    for batch in dataloader:
        model(batch)


def _quantize_svdquant(dim: int = 64) -> nn.Module:
    model = _SVDMLP(dim)
    quant_config = mtq.INT8_SMOOTHQUANT_CFG.copy()
    quant_config["algorithm"] = {"method": "svdquant", "lowrank": 8}
    data = [torch.randn(2, dim) for _ in range(2)]
    mtq.quantize(model, quant_config, partial(_forward_loop, dataloader=data))
    return model


def _quantized_linears(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


def test_svdquant_lora_stays_on_weight_quantizer():
    """LoRA lives on the quantizer, not the module (the export layer promotes it)."""
    model = _quantize_svdquant()
    linears = _quantized_linears(model)
    assert linears
    for module in linears:
        wq = module.weight_quantizer
        assert wq.svdquant_lora_a is not None
        assert wq.svdquant_lora_b is not None
        # Not refactored onto the module.
        assert not hasattr(module, "svdquant_lora_a")
        assert not hasattr(module, "svdquant_lora_b")


def test_svdquant_forward_includes_nonzero_residual():
    """The forward output includes a nonzero low-rank residual term."""
    model = _quantize_svdquant()
    for module in _quantized_linears(model):
        x = torch.randn(2, module.in_features)

        residual = module._compute_lora_residual(x)
        assert residual is not None
        assert torch.count_nonzero(residual) > 0

        full = module(x)

        # Temporarily drop the LoRA buffers to get the base (no-residual) output.
        wq = module.weight_quantizer
        lora_a = wq._svdquant_lora_a
        lora_b = wq._svdquant_lora_b
        delattr(wq, "_svdquant_lora_a")
        delattr(wq, "_svdquant_lora_b")
        try:
            base = module(x)
        finally:
            wq.register_buffer("_svdquant_lora_a", lora_a)
            wq.register_buffer("_svdquant_lora_b", lora_b)

        # The residual measurably changes the forward output.
        assert not torch.allclose(full, base)


def test_svdquant_fold_weight_removes_buffers_and_changes_weight():
    """fold_weight() folds the residual into the weight and drops the buffers."""
    model = _quantize_svdquant()
    for module in _quantized_linears(model):
        wq = module.weight_quantizer
        assert hasattr(wq, "_svdquant_lora_a")
        assert hasattr(wq, "_svdquant_lora_b")

        weight_before = module.weight.detach().clone()
        module.fold_weight()

        assert not hasattr(wq, "_svdquant_lora_a")
        assert not hasattr(wq, "_svdquant_lora_b")
        # Folding (quantized weight + low-rank residual) changes the stored weight.
        assert not torch.allclose(module.weight, weight_before)
