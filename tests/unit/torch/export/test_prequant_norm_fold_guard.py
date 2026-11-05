# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regression tests for the out-of-group consumer guard in ``_fuse_shared_input_modules``.

Folding a ``pre_quant_scale`` rewrites the shared norm itself, so it is only valid while the
norm output feeds nothing but the fused group.  When another consumer exists -- e.g.
GatedDeltaNet's ``in_proj_a`` / ``in_proj_b``, which AWQ recipes exclude from quantization --
that consumer silently receives ``pre_quant_scale``-scaled activations.  Measured on
MiMo-V2.6-Distill-Qwen-9B: activation relL2 1.3-1.8 on the affected layers, PPL 11.34 ->
10.73 once the fold and its resmoothing are skipped for that norm.
"""

import copy
import warnings

import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_quantization_format
from modelopt.torch.export.unified_export_hf import (
    _fuse_shared_input_modules,
    collect_shared_input_modules,
)

DIM = 8
GROUP_PQS = (torch.linspace(0.5, 1.5, DIM), torch.linspace(1.5, 2.5, DIM))


def _zero_centered_norm(dim: int = DIM):
    """A norm named like the real Qwen3.5 zero-centered norm (``x * (1 + weight)``)."""

    class Qwen3_5RMSNorm(torch.nn.Module):  # noqa: N801 - the real Qwen3.5 class name
        def __init__(self, hidden_size: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(hidden_size))

        def forward(self, x):
            return x * (1.0 + self.weight)

    return Qwen3_5RMSNorm(dim)


class _Block(torch.nn.Module):
    """Norm feeding a fused q/k pair, plus (optionally) an excluded consumer of the same norm."""

    def __init__(self, with_out_of_group_consumer: bool):
        super().__init__()
        self.input_layernorm = _zero_centered_norm()
        self.q_proj = torch.nn.Linear(DIM, DIM, bias=False)
        self.k_proj = torch.nn.Linear(DIM, DIM, bias=False)
        self.extra_proj = (
            torch.nn.Linear(DIM, DIM, bias=False) if with_out_of_group_consumer else None
        )

    def forward(self, x):
        hidden = self.input_layernorm(x)
        out = self.q_proj(hidden) + self.k_proj(hidden)
        if self.extra_proj is not None:
            out = out + self.extra_proj(hidden)
        return out


class _Model(torch.nn.Module):
    def __init__(self, with_out_of_group_consumer: bool):
        super().__init__()
        self.block = _Block(with_out_of_group_consumer)

    def forward(self, x):
        return self.block(x)


def _quantized_model(with_out_of_group_consumer: bool):
    """Quantize the block with INT4_AWQ_CFG, leaving ``extra_proj`` unquantized."""
    model = _Model(with_out_of_group_consumer).eval()
    quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    quant_cfg["quant_cfg"].append({"quantizer_name": "*extra_proj*", "enable": False})
    mtq.quantize(model, quant_cfg, lambda m: m(torch.randn(2, DIM)))

    # Distinct per-module scales: the fold is only valid for a shared, averaged scale.
    for module, scale in zip((model.block.q_proj, model.block.k_proj), GROUP_PQS):
        module.input_quantizer._pre_quant_scale = scale.clone()
    return model


def _fuse(model):
    """Run the real fusion path and return the warnings it emitted."""
    input_to_linear, output_to_layernorm, input_to_consumers = collect_shared_input_modules(
        model, lambda: model(torch.randn(2, DIM)), collect_layernorms=True, collect_consumers=True
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fuse_shared_input_modules(
            model,
            input_to_linear,
            output_to_layernorm,
            qkv_only=False,
            fuse_layernorms=True,
            quantization_format=get_quantization_format(model),
            input_to_consumers=input_to_consumers,
        )
    return [str(w.message) for w in caught]


def test_out_of_group_consumer_blocks_the_fold():
    """A norm feeding an excluded consumer keeps its weight and per-module pre_quant_scale."""
    model = _quantized_model(with_out_of_group_consumer=True)
    norm = model.block.input_layernorm
    members = [model.block.q_proj, model.block.k_proj]
    weight_before = norm.weight.detach().clone()
    member_weights_before = [m.weight.detach().clone() for m in members]

    messages = _fuse(model)

    assert any("Not folding pre_quant_scale" in message for message in messages)
    assert torch.equal(norm.weight, weight_before)
    for module, weight, scale in zip(members, member_weights_before, GROUP_PQS):
        assert torch.equal(module.weight, weight)
        assert torch.equal(module.input_quantizer._pre_quant_scale, scale)
        assert not hasattr(module, "fused_with_prequant")


def test_fold_still_applies_without_out_of_group_consumers():
    """Control: without the extra consumer the block folds as before, using the average scale."""
    model = _quantized_model(with_out_of_group_consumer=False)
    norm = model.block.input_layernorm
    weight_before = norm.weight.detach().clone()

    messages = _fuse(model)

    assert not any("Not folding pre_quant_scale" in message for message in messages)
    scale_avg = torch.stack([scale.float() for scale in GROUP_PQS]).mean(dim=0)
    expected = (weight_before + 1.0) * scale_avg - 1.0
    assert torch.allclose(norm.weight, expected, rtol=1e-6, atol=1e-6)
    for module in (model.block.q_proj, model.block.k_proj):
        assert not hasattr(module.input_quantizer, "_pre_quant_scale")
        assert module.fused_with_prequant
