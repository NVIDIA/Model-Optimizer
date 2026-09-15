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

from typing import Literal

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
    plan_hash,
    resolve_targets,
    stage_predicate,
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

# Weight-only: nothing needs a forward, so `weight_only_quantize` inside `max_calibrate` is the
# only thing that writes weight amax. Scoping bugs that a forward pass would paper over show up.
WEIGHT_ONLY_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 32}}},
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


def _uncalibrated_weight_quantizers(model):
    from modelopt.torch.quantization.nn import TensorQuantizer

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer)
        and "weight_quantizer" in name
        and not module._disabled
        and getattr(module, "_amax", None) is None
    ]


def _run_chain(cfg, quant_cfg=None):
    """Calibrate a fresh model with one scoped pipeline; returns weight amax by name."""
    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": quant_cfg or QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*mlp*", "cfg": cfg}],
        },
        _forward_loop,
    )
    return _weight_amax(model)


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
    """`optimizes` is about what an algorithm *improves*, not everything it writes.

    `mse` does seed input amax through its internal max bootstrap, so it is not literally a
    no-op here -- but it refines nothing on the input side, which is what the user asked for.
    """
    with pytest.raises(AlgoCfgValidationError, match="only improves weight quantizers"):
        compile_algo_cfg(
            {
                "algo_cfg": [{"quantizer_name": "*input_quantizer", "cfg": ["mse"]}],
                "algorithm": None,
            },
            quantized,
        )


def test_whole_module_algorithm_cannot_take_a_partial_quantizer_scope(quantized):
    """`awq_lite` writes the input quantizer too, so a weight-only scope cannot hold it."""
    with pytest.raises(AlgoCfgValidationError, match="writes whole modules"):
        compile_algo_cfg(
            {
                "algo_cfg": [{"quantizer_name": "*weight_quantizer", "cfg": ["awq_lite"]}],
                "algorithm": None,
            },
            quantized,
        )


def test_whole_module_algorithm_accepts_a_scope_closed_over_its_modules(quantized):
    """Whole-model and `module_name` scopes both cover every quantizer a module owns."""
    for entry in (
        {"quantizer_name": "*", "cfg": ["awq_lite"]},
        {"module_name": "*mlp*", "cfg": ["awq_lite"]},
    ):
        plan = compile_algo_cfg({"algo_cfg": [entry], "algorithm": None}, quantized)
        assert [stage.algo for stage in plan] == ["awq_lite"]


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


def _writable_state(model):
    from modelopt.torch.quantization.nn import TensorQuantizer

    state = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            role = "weight_amax" if "weight_quantizer" in name else "input_amax"
            for token, attr in ((role, "_amax"), ("pre_quant_scale", "_pre_quant_scale")):
                value = getattr(module, attr, None)
                state[(token, name)] = None if value is None else value.detach().clone().float()
        if hasattr(module, "weight_quantizer") and hasattr(module, "weight"):
            state[("weight", name)] = module.weight.detach().clone().float()
    return state


@pytest.mark.parametrize("algo", ["max", "mse"])
def test_declared_produces_is_an_upper_bound_on_what_the_algorithm_writes(algo):
    """Declared capabilities are a claim about the implementation; the rules trust it blindly.

    `mse` is here because it was wrong: it seeds `input_amax` through an internal
    `max_calibrate` before refining weight amax, which the table did not declare. The full
    sweep over all eleven algorithms lives in the out-of-tree `demos/05_conformance.py`; these
    two are the CI-sized version.
    """
    from modelopt.torch.quantization.algo_cfg import capabilities_for

    before = _writable_state(
        mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)
    )
    after = _writable_state(
        mtq.quantize(
            _model(),
            {
                "quant_cfg": QUANT_CFG,
                "algorithm": None,
                "algo_cfg": [{"quantizer_name": "*", "cfg": [algo]}],
            },
            _forward_loop,
        )
    )
    written = set()
    for key in set(before) | set(after):
        old, new = before.get(key), after.get(key)
        if old is None and new is None:
            continue
        if old is None or new is None or old.shape != new.shape or not torch.equal(old, new):
            written.add(key[0])

    declared = set(capabilities_for(algo).produces)
    assert written <= declared, f"{algo} writes {sorted(written - declared)}, undeclared"


def test_a_custom_algorithm_inherits_conservative_capabilities():
    """The registry is the extension point; an algorithm registered there must still be checked.

    When capabilities lived in a side table, anything absent from it got `None` and every
    validation rule skipped it silently. Inheriting the pessimistic base default means a custom
    algorithm is over-constrained instead of exempt.
    """
    from modelopt.torch.quantization.algo_cfg import ALL_WRITABLE_TOKENS, capabilities_for
    from modelopt.torch.quantization.config import QuantizeAlgorithmConfig
    from modelopt.torch.quantization.mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    class _CustomConfig(QuantizeAlgorithmConfig):
        method: Literal["my_custom_algo"] = "my_custom_algo"

    @CalibrateModeRegistry.register_mode
    class _CustomDescriptor(BaseCalibrateModeDescriptor):
        _calib_func = None

        @property
        def config_class(self):
            return _CustomConfig

    try:
        caps = capabilities_for("my_custom_algo")
        assert caps is not None, "a registered algorithm must have capabilities"
        assert caps.produces == ALL_WRITABLE_TOKENS, "assume it writes everything"
        assert not caps.honors_write_mask, "assume it cannot be scoped"
    finally:
        CalibrateModeRegistry.remove_mode("my_custom_algo_calibrate")


def test_calib_mutates_weights_false_is_rejected_for_every_weight_writing_algorithm(quantized):
    """The config-time guard is sourced from `produces`, so it covers every algorithm.

    `calib_mutates_weights=False` tells layerwise calibration not to write the layer's
    weights back after the stage, which silently drops the updates of any algorithm that
    writes `weight`. That used to be a hand-maintained `_mutates_weights` flag per config
    class -- a second statement of what `produces` already says, free to drift and easy to
    forget on a new algorithm. Now there is one statement, and this checks it holds for the
    whole registry rather than a hand-listed few.
    """
    from modelopt.torch.quantization.algo_cfg import WEIGHT, capabilities_for, known_algorithms
    from modelopt.torch.quantization.mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    checked = []
    for algo in known_algorithms():
        caps = capabilities_for(algo)
        if WEIGHT not in caps.produces:
            continue
        config_class = CalibrateModeRegistry[
            BaseCalibrateModeDescriptor._get_mode_name(algo)
        ].config_class
        with pytest.raises(ValidationError, match="mutates layer weights in-place"):
            config_class(layerwise={"enable": True, "calib_mutates_weights": False})
        checked.append(algo)

    assert checked, "no weight-writing algorithm found -- the check would pass vacuously"


def test_calib_mutates_weights_defaults_to_derived(quantized):
    """Unset means "derive from the algorithm", not "assume the safe value"."""
    from modelopt.torch.quantization.config import AWQLiteCalibConfig, MaxCalibConfig

    assert MaxCalibConfig().layerwise.calib_mutates_weights is None
    assert AWQLiteCalibConfig().layerwise.calib_mutates_weights is None
    # An amax-only algorithm may still opt out explicitly.
    assert (
        MaxCalibConfig(
            layerwise={"enable": True, "calib_mutates_weights": False}
        ).layerwise.calib_mutates_weights
        is False
    )


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


def test_range_search_then_gptq_is_recognized_as_a_handoff(quantized):
    """GPTQ rounds against an existing grid, so a preceding search feeds it, not dies."""
    plan = compile_algo_cfg(
        {
            "algo_cfg": [{"module_name": "*mlp*", "cfg": ["mse", {"method": "gptq"}]}],
            "algorithm": None,
        },
        quantized,
    )
    assert [stage.algo for stage in plan] == ["mse", "gptq"]
    assert derive_handoff(quantized, plan, 0) == {}
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_leading_gptq_still_initializes_its_own_amax(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["gptq"]}], "algorithm": None}, quantized
    )
    assert derive_handoff(quantized, plan, 0) == {}


def test_gptq_preserves_a_preceding_range_search():
    """The point of the chain: GPTQ compensates against the grid MSE searched.

    Prior art for this ordering is DeepCompressor's QoQ recipes, which run their weight
    range search before the GPTQ kernel rather than after.
    """
    gptq = {"method": "gptq", "block_size": 32}
    only_mse = _run_chain(["mse"])
    only_gptq = _run_chain([gptq])
    chained = _run_chain(["mse", gptq])

    probe = "layers.0.mlp.gate_proj.weight_quantizer"
    # GPTQ kept MSE's amax instead of re-deriving it from max ...
    assert torch.equal(chained[probe], only_mse[probe])
    # ... and the resulting model is not the one plain GPTQ produces.
    assert not torch.equal(chained[probe], only_gptq[probe])


def test_awq_full_is_exactly_its_two_stage_pipeline():
    """`awq_full` is a bundled composite; the plan surface expresses it as the pipeline.

    Bit-identical equivalence is the strongest available evidence that stage sequencing
    reproduces what an algorithm does internally today.
    """
    bundled = mtq.quantize(
        _model(), {"quant_cfg": QUANT_CFG, "algorithm": "awq_full"}, _forward_loop
    )
    pipeline = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"quantizer_name": "*", "cfg": ["awq_lite", "awq_clip"]}],
        },
        _forward_loop,
    )
    lite_only = mtq.quantize(
        _model(), {"quant_cfg": QUANT_CFG, "algorithm": "awq_lite"}, _forward_loop
    )
    bundled_amax, pipeline_amax = _weight_amax(bundled), _weight_amax(pipeline)

    assert set(bundled_amax) == set(pipeline_amax)
    assert all(torch.equal(bundled_amax[k], pipeline_amax[k]) for k in bundled_amax)
    # Guard against the assertion passing because awq_clip did nothing.
    lite_amax = _weight_amax(lite_only)
    assert any(not torch.equal(bundled_amax[k], lite_amax[k]) for k in bundled_amax)


def test_awq_then_mse_refines_the_smoothed_weights():
    """MSE re-searches the amax on AWQ's smoothed weights: a forward-free awq_clip."""
    only_awq = _run_chain(["awq_lite"])
    chained = _run_chain(["awq_lite", "mse"])

    assert set(only_awq) == set(chained)
    assert any(not torch.equal(only_awq[k], chained[k]) for k in only_awq)


def test_handoff_needs_coverage_not_just_overlap(quantized):
    """A narrow producer must not switch off a wide consumer's own initialization.

    `*layers.0.mlp*` produces amax for one layer only; the `*mlp*` consumer covers both. If the
    handoff fired, `layers.1.mlp.*` would end with no amax at all and no error.
    """
    plan = compile_algo_cfg(
        {
            "algo_cfg": [
                {"module_name": "*layers.0.mlp*", "cfg": ["mse"]},
                {"module_name": "*mlp*", "cfg": ["gptq"]},
            ],
            "algorithm": None,
        },
        quantized,
        strict=False,  # the narrow stage is dead under the wide one; not what this asserts
    )
    gptq_index = next(i for i, stage in enumerate(plan) if stage.algo == "gptq")
    assert derive_handoff(quantized, plan, gptq_index) == {}


def test_handoff_fires_when_the_producer_covers_the_consumer(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["mse", "gptq"]}], "algorithm": None},
        quantized,
    )
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_a_user_written_kwarg_beats_the_derived_handoff():
    """The handoff is an inference; an explicit value is an instruction."""
    model = mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [
                {
                    "module_name": "*mlp*",
                    "cfg": ["max", {"method": "mse", "skip_max_init": False}],
                }
            ],
        },
        _forward_loop,
    )
    assert _weight_amax(model)


def test_unknown_algorithm_is_fatal_even_when_not_strict(quantized):
    """`strict=False` downgrades judgement calls, not "there is nothing to dispatch to"."""
    with pytest.raises(AlgoCfgValidationError, match="unknown algorithm"):
        compile_algo_cfg(
            {"algo_cfg": [{"module_name": "*", "cfg": ["nope"]}], "algorithm": None},
            quantized,
            strict=False,
        )


def test_fused_siblings_compared_against_the_fallback_too(quantized):
    """Naming one sibling in an entry leaves the others on the fallback -- still a mismatch."""
    with pytest.raises(AlgoCfgValidationError, match="fusible siblings"):
        compile_algo_cfg(
            {"algo_cfg": [{"module_name": "*q_proj", "cfg": ["gptq"]}], "algorithm": "max"},
            quantized,
        )


def test_fused_siblings_with_no_entry_at_all_are_consistent(quantized):
    """Whole-model plans give every sibling the same pipeline, so they must not trip the rule."""
    plan = compile_algo_cfg({"algorithm": "max"}, quantized)
    assert [stage.algo for stage in plan] == ["max"]


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
    for name in ("layers.0.mlp.gate_proj", "layers.0.mlp.gate_proj.weight_quantizer"):
        assert should_process(quantized.get_submodule(name))
    for name in ("layers.0.self_attn.q_proj", "layers.0.self_attn.q_proj.weight_quantizer"):
        assert not should_process(quantized.get_submodule(name))


def test_stage_predicate_matches_on_identity_not_name(quantized):
    """`layerwise_calibrate` hands an algorithm a subtree, where names are relative.

    A name-based mask matches nothing there and the stage becomes a silent no-op, so the mask
    keys on the module object instead.
    """
    plan = compile_algo_cfg(
        {"algo_cfg": [{"module_name": "*mlp*", "cfg": ["max"]}], "algorithm": None}, quantized
    )
    should_process = stage_predicate(quantized, plan[0])

    layer = quantized.get_submodule("layers.0")
    subtree_names = dict(layer.named_modules())
    assert "mlp.gate_proj" in subtree_names, "subtree names are relative to the layer"
    # Same object, different name -- still in scope.
    assert should_process(subtree_names["mlp.gate_proj"])
    assert not should_process(subtree_names["self_attn.q_proj"])


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


def test_module_scope_reaches_sequential_quantizer_levels():
    """A SequentialQuantizer nests its levels below the linear, not directly under it."""
    model = mtq.quantize(_model(), {"quant_cfg": SEQUENTIAL_QUANT_CFG, "algorithm": None}, None)
    _, quantizers = resolve_targets(model, "*", "module_name")
    levels = [
        name
        for name, _ in model.named_modules()
        if name.split(".")[-1].isdigit() and "quantizer" in name
    ]
    assert levels, "fixture should produce sequential sub-quantizers"
    assert all(level in quantizers for level in levels)


def test_sequential_quantizers_are_calibrated_under_a_module_scope():
    scoped = mtq.quantize(
        _model(),
        {
            "quant_cfg": SEQUENTIAL_QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*", "cfg": ["max"]}],
        },
        _forward_loop,
    )
    legacy = mtq.quantize(
        _model(), {"quant_cfg": SEQUENTIAL_QUANT_CFG, "algorithm": "max"}, _forward_loop
    )
    assert _uncalibrated_weight_quantizers(scoped) == []
    assert _uncalibrated_weight_quantizers(legacy) == []


def test_quantizer_scoped_entry_leaves_the_fallback_able_to_calibrate_weights():
    """A `quantizer_name` entry claims quantizers, not the linears that own them.

    Subtracting the parent modules too would strip the fallback stage of every module, and
    with it `weight_only_quantize` -- invisible whenever a forward pass would have set the
    weight amax anyway, which is why this uses a weight-only config with no forward loop.
    """
    scoped = mtq.quantize(
        _model(),
        {
            "quant_cfg": WEIGHT_ONLY_QUANT_CFG,
            "algorithm": "max",
            "algo_cfg": [{"quantizer_name": "*input_quantizer", "cfg": ["max"]}],
        },
        None,
    )
    assert _uncalibrated_weight_quantizers(scoped) == []


def test_algorithms_that_ignore_the_write_mask_cannot_be_scoped(quantized):
    """`local_hessian` takes no `should_process`; scoping it would write outside its scope."""
    with pytest.raises(AlgoCfgValidationError, match="does not honour the scoping write-mask"):
        compile_algo_cfg(
            {
                "algo_cfg": [{"module_name": "*mlp*", "cfg": ["local_hessian"]}],
                "algorithm": None,
            },
            quantized,
        )


def test_algorithms_that_ignore_the_write_mask_are_still_usable_whole_model(quantized):
    plan = compile_algo_cfg(
        {"algo_cfg": [{"quantizer_name": "*", "cfg": ["local_hessian"]}], "algorithm": None},
        quantized,
    )
    assert [stage.algo for stage in plan] == ["local_hessian"]


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
