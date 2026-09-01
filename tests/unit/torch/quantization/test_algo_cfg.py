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

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algo_cfg import (
    AlgoCfgValidationError,
    compile_algo_cfg,
    derive_handoff,
    plan_hash,
    resolve_targets,
    stage_predicate,
    stage_targets,
)
from modelopt.torch.quantization.config import AlgoCfgEntry

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


@pytest.fixture
def quantized():
    """Quantizers inserted, nothing calibrated: compile only needs the model structure."""
    return mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)


def _weight_amax(model):
    return {
        name: module._amax.detach().clone()
        for name, module in model.named_modules()
        if name.endswith("weight_quantizer") and getattr(module, "_amax", None) is not None
    }


# ---------------------------------------------------------------------------- config


def test_entry_requires_exactly_one_selector():
    with pytest.raises(ValueError, match="exactly one of"):
        AlgoCfgEntry(module_name="*mlp*", quantizer_name="*weight_quantizer", cfg=["max"])
    with pytest.raises(ValueError, match="exactly one of"):
        AlgoCfgEntry(cfg=["max"])


def test_entry_requires_nonempty_cfg():
    with pytest.raises(ValueError, match="at least one algorithm"):
        AlgoCfgEntry(module_name="*mlp*", cfg=[])


def test_entry_wraps_a_bare_cfg_in_a_list():
    assert AlgoCfgEntry(module_name="*mlp*", cfg="max").cfg == ["max"]


# ---------------------------------------------------------------------------- lowering


def test_algorithm_and_equivalent_algo_cfg_compile_to_the_same_plan(quantized):
    """The legacy whole-model path is the all-``"*"`` case, not a second engine."""
    legacy = compile_algo_cfg({"algorithm": "max"}, quantized)
    explicit = compile_algo_cfg(
        {"algo_cfg": [{"quantizer_name": "*", "cfg": ["max"]}], "algorithm": None}, quantized
    )
    assert plan_hash(legacy) == plan_hash(explicit)


def test_pipeline_lowers_in_order_with_kwargs(quantized):
    plan = compile_algo_cfg(
        {
            "algo_cfg": [
                {"module_name": "*mlp*", "cfg": ["max", {"method": "mse", "step_size": 0.05}]}
            ],
            "algorithm": None,
        },
        quantized,
    )
    assert [stage.algo for stage in plan] == ["max", "mse"]
    assert [stage.order for stage in plan] == [0, 1]
    assert plan[1].cfg["step_size"] == 0.05


def test_fallback_algorithm_excludes_scopes_claimed_by_entries(quantized):
    """Otherwise the model-wide default silently re-runs over every scoped pipeline."""
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}], "algorithm": "max"}, quantized
    )
    fallback = plan[-1]
    assert fallback.exclude == (("module_name", "*mlp*"),)

    _, fallback_quantizers = stage_targets(quantized, fallback)
    _, mlp_quantizers = resolve_targets(quantized, "*mlp*", "module_name")
    assert mlp_quantizers
    assert not (fallback_quantizers & mlp_quantizers)


def test_plan_hash_ignores_provenance_only_differences(quantized):
    a = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}], "algorithm": None}, quantized
    )
    b = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}], "algorithm": None}, quantized
    )
    assert plan_hash(a) == plan_hash(b)


# ---------------------------------------------------------------------------- validation


def test_unknown_algorithm_is_rejected(quantized):
    with pytest.raises(AlgoCfgValidationError, match="unknown algorithm"):
        compile_algo_cfg(
            {"algo_cfg": [{"module_name": "*", "cfg": ["awq_supreme"]}], "algorithm": None},
            quantized,
        )


def test_scope_matching_nothing_is_rejected(quantized):
    with pytest.raises(AlgoCfgValidationError, match="matches no target"):
        compile_algo_cfg(
            {"algo_cfg": [{"module_name": "*cross_attn*", "cfg": ["max"]}], "algorithm": None},
            quantized,
        )


def test_weight_only_algorithm_on_input_quantizers_is_rejected(quantized):
    with pytest.raises(AlgoCfgValidationError, match="only writes weight quantizers"):
        compile_algo_cfg(
            {
                "algo_cfg": [{"quantizer_name": "*input_quantizer", "cfg": ["mse"]}],
                "algorithm": None,
            },
            quantized,
        )


def test_fusible_siblings_must_share_one_pipeline(quantized):
    with pytest.raises(AlgoCfgValidationError, match="fusible siblings"):
        compile_algo_cfg(
            {
                "algo_cfg": [
                    {"module_name": "*gate_proj", "cfg": ["awq_lite"]},
                    {"module_name": "*up_proj", "cfg": ["max"]},
                ],
                "algorithm": None,
            },
            quantized,
        )


def test_stage_whose_output_is_overwritten_before_being_read_is_rejected(quantized):
    with pytest.raises(AlgoCfgValidationError, match="is dead"):
        compile_algo_cfg(
            {
                "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max", "mse", "max"]}],
                "algorithm": None,
            },
            quantized,
        )


def test_repeating_a_smoothing_algorithm_is_rejected(quantized):
    """``awq_lite`` folds ``1/s`` into the weight; a second pass folds again without unfolding."""
    with pytest.raises(AlgoCfgValidationError, match="pre_quant_scale"):
        compile_algo_cfg(
            {
                "algo_cfg": [{"module_name": "*mlp*", "cfg": ["awq_lite", "awq_lite"]}],
                "algorithm": None,
            },
            quantized,
        )


def test_awq_then_mse_then_awq_reports_both_problems(quantized):
    with pytest.raises(AlgoCfgValidationError) as excinfo:
        compile_algo_cfg(
            {
                "algo_cfg": [{"module_name": "*mlp*", "cfg": ["awq_lite", "mse", "awq_lite"]}],
                "algorithm": None,
            },
            quantized,
        )
    message = str(excinfo.value)
    assert "2 problem(s)" in message
    assert "is dead" in message
    assert "pre_quant_scale" in message


def test_awq_then_mse_is_accepted(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["awq_lite", "mse"]}], "algorithm": None},
        quantized,
    )
    assert [stage.algo for stage in plan] == ["awq_lite", "mse"]


def test_stages_sharing_a_module_but_writing_different_roles_do_not_conflict(quantized):
    """A ``module_name`` scope resolves to both roles; overlap is judged on what is written."""
    plan = compile_algo_cfg(
        {
            "algo_cfg": [
                {"module_name": "*mlp*", "cfg": ["max", "mse"]},
                {"quantizer_name": "*input_quantizer", "cfg": ["max"]},
            ],
            "algorithm": None,
        },
        quantized,
    )
    assert len(plan) == 3


def test_strict_false_downgrades_violations_to_warnings(quantized):
    config = {
        "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max", "mse", "max"]}],
        "algorithm": None,
    }
    with pytest.warns(UserWarning, match="is dead"):
        plan = compile_algo_cfg(config, quantized, strict=False)
    assert len(plan) == 3


# ---------------------------------------------------------------------------- handoff


def test_mse_after_a_stage_that_produced_amax_skips_its_own_max_init(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["max", "mse"]}], "algorithm": None},
        quantized,
    )
    assert derive_handoff(quantized, plan, 0) == {}
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_handoff_is_dropped_for_algorithms_without_the_matching_knob():
    """`awq_clip` also consumes a prior stage's amax but has no `skip_max_init` field."""
    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max", "awq_clip"]}],
        },
        _forward_loop,
    )
    assert _weight_amax(model)


def test_leading_mse_still_initializes_its_own_amax(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["mse"]}], "algorithm": None}, quantized
    )
    assert derive_handoff(quantized, plan, 0) == {}


# ---------------------------------------------------------------------------- scoping


def test_stage_predicate_matches_only_its_own_targets(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}], "algorithm": None}, quantized
    )
    should_process = stage_predicate(quantized, plan[0])
    assert should_process("layers.0.mlp.gate_proj")
    assert should_process("layers.0.mlp.gate_proj.weight_quantizer")
    assert not should_process("layers.0.self_attn.q_proj")
    assert not should_process("layers.0.self_attn.q_proj.weight_quantizer")


def test_scoped_stage_writes_only_its_targets():
    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}],
        },
        _forward_loop,
    )
    calibrated = _weight_amax(model)
    assert calibrated
    assert all("mlp" in name for name in calibrated)


def test_scoping_never_toggles_enable_state():
    from modelopt.torch.quantization.nn import TensorQuantizer

    def flags(model):
        return {
            name: bool(module.is_enabled)
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer)
        }

    before = flags(mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None))
    after = flags(
        mtq.quantize(
            _model(),
            {
                "quant_cfg": QUANT_CFG,
                "algorithm": None,
                "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}],
            },
            _forward_loop,
        )
    )
    assert before == after


def test_scoped_plan_records_a_single_calibration_mode():
    from modelopt.torch.opt.conversion import ModeloptStateManager

    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max", "mse"]}],
        },
        _forward_loop,
    )
    modes = [str(mode) for mode, _, _ in ModeloptStateManager(model).modes_with_states()]
    assert modes == ["quantize", "calibration_plan"]


def test_a_stage_can_follow_mse(quantized):
    """``mse`` installs a search calibrator; it must not outlive its own stage."""
    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*mlp*", "cfg": ["max", "mse", "max"]}],
            "strict": False,
        },
        _forward_loop,
    )
    assert _weight_amax(model)


def test_legacy_path_is_numerically_unchanged():
    legacy = mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": "max"}, _forward_loop)
    planned = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"quantizer_name": "*", "cfg": ["max"]}],
        },
        _forward_loop,
    )
    legacy_amax, planned_amax = _weight_amax(legacy), _weight_amax(planned)
    assert set(legacy_amax) == set(planned_amax)
    assert all(torch.equal(legacy_amax[k], planned_amax[k]) for k in legacy_amax)
