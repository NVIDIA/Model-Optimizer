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

"""Tests for ``algo_cfg`` lowering, validation and scoped execution."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ValidationError

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algo_cfg import (
    AlgoCfgValidationError,
    compile_algo_cfg,
    derive_handoff,
    stage_targets,
)
from modelopt.torch.quantization.config import AlgoCfgEntry

# Two-level weight quantizer (the W4A8 / INT4-AWQ shape): each weight quantizer becomes a
# SequentialQuantizer whose levels are *grandchildren* of the linear.
SEQUENTIAL_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {
        "quantizer_name": "*weight_quantizer",
        "cfg": [{"num_bits": 4, "block_sizes": {-1: 32}}, {"num_bits": (4, 3), "axis": None}],
    },
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
]

QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 32}}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]


class _MLP(nn.Module):
    def __init__(self, d=32, h=64):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    # q/k/v are the canonical fused sibling group: they share one weight `global_amax`.
    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _Block(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d)

    def forward(self, x):
        return x + self.mlp(x + self.self_attn(x))


class _Model(nn.Module):
    def __init__(self, d=32, n=2):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _model(seed=0):
    torch.manual_seed(seed)
    return _Model().eval()


def _forward_loop(model):
    torch.manual_seed(1)
    for _ in range(2):
        model(torch.randn(2, 8, 32))


def _compile(model, *entries, algorithm=None):
    return compile_algo_cfg({"algo_cfg": list(entries), "algorithm": algorithm}, model)


@pytest.fixture
def quantized():
    return mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)


# --- Config surface -------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        {"cfg": ["max"]},  # no selector
        {"module_name": "*", "quantizer_name": "*", "cfg": ["max"]},  # two selectors
        {"module_name": "*", "cfg": []},  # empty pipeline
        # `None` must be caught before the bare-value wrap turns it into a truthy `[None]`,
        # which would claim the scope (suppressing the fallback) while running no stage.
        {"module_name": "*", "cfg": None},
    ],
)
def test_entry_needs_exactly_one_selector_and_a_pipeline(entry):
    with pytest.raises((ValidationError, ValueError)):
        AlgoCfgEntry(**entry)


def test_entry_accepts_a_bare_algorithm():
    assert AlgoCfgEntry(module_name="*mlp*", cfg="gptq").cfg == ["gptq"]


# --- Lowering and resolution ----------------------------------------------------------


def test_pipeline_order_is_preserved(quantized):
    plan = _compile(quantized, AlgoCfgEntry(module_name="*mlp*", cfg=["mse", "gptq"]))
    assert [s.algo for s in plan] == ["mse", "gptq"]
    assert [s.order for s in plan] == [0, 1]


def test_module_scope_selects_that_subtree_only(quantized):
    plan = _compile(quantized, AlgoCfgEntry(module_name="*self_attn*", cfg=["max"]))
    modules, quantizers = stage_targets(quantized, plan[0])
    assert modules and all("self_attn" in m for m in modules)
    assert quantizers and all("self_attn" in q for q in quantizers)


def test_quantizer_scope_selects_a_role(quantized):
    plan = _compile(quantized, AlgoCfgEntry(quantizer_name="*input_quantizer", cfg=["max"]))
    _, quantizers = stage_targets(quantized, plan[0])
    assert quantizers and all(q.endswith("input_quantizer") for q in quantizers)


def test_sequential_quantizer_levels_resolve_to_their_linear():
    """A SequentialQuantizer nests levels as grandchildren; the scope must still own them."""
    model = mtq.quantize(_model(), {"quant_cfg": SEQUENTIAL_QUANT_CFG, "algorithm": None}, None)
    plan = _compile(model, AlgoCfgEntry(module_name="*mlp*", cfg=["max"]))
    _, quantizers = stage_targets(model, plan[0])
    assert any(q.endswith((".0", ".1")) for q in quantizers), quantizers
    assert all("mlp" in q for q in quantizers)


def test_fallback_algorithm_excludes_what_entries_claimed(quantized):
    plan = _compile(quantized, AlgoCfgEntry(module_name="*mlp*", cfg=["gptq"]), algorithm="max")
    fallback = [s for s in plan if s.algo == "max"]
    assert len(fallback) == 1
    _, quantizers = stage_targets(quantized, fallback[0])
    assert quantizers and not any("mlp" in q for q in quantizers)


def test_total_coverage_drops_the_redundant_fallback(quantized):
    plan = _compile(quantized, AlgoCfgEntry(module_name="*", cfg=["mse"]), algorithm="max")
    assert [s.algo for s in plan] == ["mse"]


# --- The two rules --------------------------------------------------------------------


def test_a_scope_matching_nothing_is_rejected(quantized):
    with pytest.raises(AlgoCfgValidationError, match="matches no target"):
        _compile(quantized, AlgoCfgEntry(module_name="*no_such_module*", cfg=["max"]))


def test_an_unscopable_algorithm_cannot_be_scoped(quantized):
    with pytest.raises(AlgoCfgValidationError, match="does not honour the scoping write-mask"):
        _compile(quantized, AlgoCfgEntry(module_name="*mlp*", cfg=["svdquant"]))


def test_an_unscopable_algorithm_is_fine_at_whole_model_scope(quantized):
    assert [s.algo for s in _compile(quantized, AlgoCfgEntry(module_name="*", cfg=["svdquant"]))]


def test_whole_model_scope_is_not_defeated_by_unreachable_quantizers(quantized):
    """A ``module_name`` glob cannot reach KV quantizers, which hang off the attention module.

    Measuring whole-model scope against *every* quantizer would reject ``module_name: "*"`` on
    any model with KV quantization enabled, even though it is whole-model scope.
    """
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    for block in quantized.layers:
        block.self_attn.k_bmm_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(num_bits=8, axis=None)
        )
    assert [s.algo for s in _compile(quantized, AlgoCfgEntry(module_name="*", cfg=["svdquant"]))]


def test_an_unknown_algorithm_is_rejected(quantized):
    with pytest.raises(AlgoCfgValidationError, match="unknown algorithm"):
        _compile(quantized, AlgoCfgEntry(module_name="*", cfg=["not_an_algorithm"]))


# --- The hand-off ---------------------------------------------------------------------


def test_handoff_is_granted_when_an_earlier_stage_covered_every_target(quantized):
    plan = _compile(quantized, AlgoCfgEntry(module_name="*", cfg=["mse", "gptq"]))
    assert derive_handoff(quantized, plan, 0) == {}
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_handoff_is_refused_when_the_producer_is_narrower(quantized):
    """Coverage, not overlap: a narrow producer leaves the rest of the targets with no amax."""
    plan = _compile(
        quantized,
        AlgoCfgEntry(module_name="*mlp*", cfg=["mse"]),
        AlgoCfgEntry(module_name="*", cfg=["gptq"]),
    )
    wide = next(i for i, s in enumerate(plan) if s.algo == "gptq")
    assert derive_handoff(quantized, plan, wide) == {}


def test_handoff_is_not_offered_to_an_algorithm_that_cannot_take_it(quantized):
    plan = _compile(quantized, AlgoCfgEntry(module_name="*", cfg=["mse", "awq_lite"]))
    assert derive_handoff(quantized, plan, 1) == {}


# --- Unsupported combinations warn rather than failing silently -----------------------


def test_splitting_a_fused_sibling_group_warns(quantized):
    """q/k/v share one weight global_amax, reconciled over the whole model after calibration."""
    with pytest.warns(UserWarning, match="fused weight amax group"):
        _compile(quantized, AlgoCfgEntry(module_name="*q_proj*", cfg=["max"]), algorithm="max")


def test_covering_the_whole_group_does_not_warn(quantized, recwarn):
    _compile(quantized, AlgoCfgEntry(module_name="*self_attn*", cfg=["max"]), algorithm="max")
    assert not [w for w in recwarn if "fused weight amax group" in str(w.message)]


# --- End to end -----------------------------------------------------------------------


def test_each_scope_runs_its_own_algorithm():
    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [
                {"module_name": "*self_attn*", "cfg": ["awq_lite"]},
                {"module_name": "*mlp*", "cfg": ["mse", "gptq"]},
            ],
        },
        _forward_loop,
    )
    from modelopt.torch.opt.conversion import ModeloptStateManager

    # Each stage is its own mode application, so the sequence is visible in modelopt state
    # rather than hidden inside one opaque entry.
    modes = [str(entry[0]) for entry in ModeloptStateManager(model).modes_with_states()]
    assert any("awq_lite" in m for m in modes), modes
    assert any("mse" in m for m in modes), modes
    assert any("gptq" in m for m in modes), modes


def test_chaining_mse_into_gptq_keeps_the_searched_grid():
    """Without the hand-off, gptq re-derives amax from max and the mse search is discarded."""

    def run(cfg):
        model = mtq.quantize(
            _model(),
            {"quant_cfg": QUANT_CFG, "algorithm": None, "algo_cfg": cfg},
            _forward_loop,
        )
        return {
            name: module._amax.detach().clone()
            for name, module in model.named_modules()
            if name.endswith("weight_quantizer") and getattr(module, "_amax", None) is not None
        }

    chained = run([{"module_name": "*", "cfg": ["mse", "gptq"]}])
    alone = run([{"module_name": "*", "cfg": ["gptq"]}])
    assert chained, "nothing was calibrated"
    assert any(not torch.equal(chained[k], alone[k]) for k in chained)
