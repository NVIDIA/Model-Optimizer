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


def _compile(model, *entries, algorithm=None, **kwargs):
    return compile_algo_cfg({"algo_cfg": list(entries), "algorithm": algorithm}, model, **kwargs)


@pytest.fixture
def quantized():
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
    legacy = _compile(quantized, algorithm="max")
    explicit = _compile(quantized, {"quantizer_name": "*", "cfg": ["max"]})
    assert legacy == explicit


def test_pipeline_lowers_in_order_with_kwargs(quantized):
    plan = _compile(
        quantized, {"module_name": "*mlp*", "cfg": ["max", {"method": "mse", "step_size": 0.05}]}
    )
    assert [stage.algo for stage in plan] == ["max", "mse"]
    assert [stage.order for stage in plan] == [0, 1]
    assert plan[1].cfg["step_size"] == 0.05


def test_fallback_algorithm_excludes_scopes_claimed_by_entries(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["max"]}, algorithm="max")
    fallback = plan[-1]
    assert fallback.exclude == (("module_name", "*mlp*"),)

    _, fallback_quantizers = stage_targets(quantized, fallback)
    _, mlp_quantizers = resolve_targets(quantized, "*mlp*", "module_name")
    assert mlp_quantizers
    assert not (fallback_quantizers & mlp_quantizers)


# ---------------------------------------------------------------------------- validation

WHOLE_MODEL = {"quantizer_name": "*", "cfg": ["max"]}

REJECTIONS = [
    ("unknown algorithm", [{"module_name": "*", "cfg": ["awq_supreme"]}]),
    ("matches no target", [{"module_name": "*cross_attn*", "cfg": ["max"]}]),
    # `refines` is what an algorithm *improves*: mse seeds input amax via its max
    # bootstrap, but refines nothing on the input side, which is what was asked for.
    ("only improves weight quantizers", [{"quantizer_name": "*input_quantizer", "cfg": ["mse"]}]),
    # awq_lite writes the input quantizer too, so a weight-only scope cannot hold it.
    ("writes whole modules", [{"quantizer_name": "*weight_quantizer", "cfg": ["awq_lite"]}]),
    (
        "fusible siblings",
        [
            {"module_name": "*gate_proj", "cfg": ["awq_lite"]},
            {"module_name": "*up_proj", "cfg": ["max"]},
        ],
    ),
    ("is dead", [{"module_name": "*mlp*", "cfg": ["max", "mse", "max"]}]),
    # awq_lite folds 1/s into the weight; a second pass folds again without unfolding.
    ("pre_quant_scale", [{"module_name": "*mlp*", "cfg": ["awq_lite", "awq_lite"]}]),
]


@pytest.mark.parametrize(("match", "algo_cfg"), REJECTIONS, ids=[r[0] for r in REJECTIONS])
def test_invalid_plan_is_rejected(quantized, match, algo_cfg):
    with pytest.raises(AlgoCfgValidationError, match=match):
        _compile(quantized, *algo_cfg)


def test_whole_module_algorithm_accepts_a_scope_closed_over_its_modules(quantized):
    for entry in (
        {"quantizer_name": "*", "cfg": ["awq_lite"]},
        {"module_name": "*mlp*", "cfg": ["awq_lite"]},
    ):
        plan = _compile(quantized, entry)
        assert [stage.algo for stage in plan] == ["awq_lite"]


def test_awq_then_mse_then_awq_reports_both_problems(quantized):
    with pytest.raises(AlgoCfgValidationError) as excinfo:
        _compile(quantized, {"module_name": "*mlp*", "cfg": ["awq_lite", "mse", "awq_lite"]})
    message = str(excinfo.value)
    assert "2 problem(s)" in message
    assert "is dead" in message
    assert "pre_quant_scale" in message


def test_awq_then_mse_is_accepted(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["awq_lite", "mse"]})
    assert [stage.algo for stage in plan] == ["awq_lite", "mse"]


def test_stages_sharing_a_module_but_writing_different_roles_do_not_conflict(quantized):
    plan = _compile(
        quantized,
        {"module_name": "*mlp*", "cfg": ["max", "mse"]},
        {"quantizer_name": "*input_quantizer", "cfg": ["max"]},
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

    declared = set(capabilities_for(algo).may_write)
    assert written <= declared, f"{algo} writes {sorted(written - declared)}, undeclared"


@pytest.mark.parametrize(
    ("algo", "key"),
    [("lsq", "scale_algorithm"), ("nvfp4_act_headroom", "weight_scale_algorithm")],
)
def test_a_delegating_algorithm_inherits_its_sub_algorithms_capabilities(algo, key):
    from modelopt.torch.quantization.algo_cfg import ACTS, capabilities_for

    # An `fp8_scale_sweep` sub-algorithm searches stored per-block scales, so the grid has to
    # be static -- otherwise `prepare` never upgrades it and the search has nothing to read.
    assert capabilities_for(algo, {}).requires_weight_scales is None
    swept = capabilities_for(algo, {key: {"method": "mse", "fp8_scale_sweep": True}})
    assert swept.requires_weight_scales == "static"

    # `local_hessian` reads activations; the delegating algorithm must say so on its behalf.
    assert ACTS in capabilities_for(algo, {key: {"method": "local_hessian"}}).requires


def test_a_delegating_algorithm_falls_back_to_the_conservative_upper_bound():
    from modelopt.torch.quantization.algo_cfg import WRITABLE_TOKENS, capabilities_for

    caps = capabilities_for("lsq", {"scale_algorithm": {"method": "not_an_algorithm"}})
    assert caps.may_write >= WRITABLE_TOKENS


def test_algorithms_that_run_awq_lite_internally_require_a_dynamic_grid():
    from modelopt.torch.quantization.algo_cfg import capabilities_for

    for algo in ("awq_lite", "awq_full", "awq_clip", "svdquant"):
        assert capabilities_for(algo).requires_weight_scales == "dynamic", algo


def test_a_custom_algorithm_inherits_conservative_capabilities():
    from modelopt.torch.quantization.algo_cfg import WRITABLE_TOKENS, capabilities_for
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
        assert caps.may_write == WRITABLE_TOKENS, "assume it writes everything"
        assert not caps.scopable, "assume it cannot be scoped"
    finally:
        CalibrateModeRegistry.remove_mode("my_custom_algo_calibrate")


def test_calib_mutates_weights_false_is_rejected_for_every_weight_writing_algorithm(quantized):
    from modelopt.torch.quantization.algo_cfg import WEIGHT, capabilities_for, known_algorithms
    from modelopt.torch.quantization.mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    checked = []
    for algo in known_algorithms():
        caps = capabilities_for(algo)
        if WEIGHT not in caps.may_write:
            continue
        config_class = CalibrateModeRegistry[
            BaseCalibrateModeDescriptor._get_mode_name(algo)
        ].config_class
        with pytest.raises(ValidationError, match="mutates layer weights in-place"):
            config_class(layerwise={"enable": True, "calib_mutates_weights": False})
        checked.append(algo)

    assert checked, "no weight-writing algorithm found -- the check would pass vacuously"


def test_calib_mutates_weights_defaults_to_derived(quantized):
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
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["max", "mse"]})
    assert derive_handoff(quantized, plan, 0) == {}
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_handoff_is_dropped_for_algorithms_without_the_matching_knob():
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
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["mse", {"method": "gptq"}]})
    assert [stage.algo for stage in plan] == ["mse", "gptq"]
    assert derive_handoff(quantized, plan, 0) == {}
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_leading_gptq_still_initializes_its_own_amax(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["gptq"]})
    assert derive_handoff(quantized, plan, 0) == {}


def test_gptq_preserves_a_preceding_range_search():
    gptq = {"method": "gptq", "block_size": 32}
    only_mse = _run_chain(["mse"])
    only_gptq = _run_chain([gptq])
    chained = _run_chain(["mse", gptq])

    probe = "layers.0.mlp.gate_proj.weight_quantizer"
    # GPTQ kept MSE's amax instead of re-deriving it from max ...
    assert torch.equal(chained[probe], only_mse[probe])
    # ... and the resulting model is not the one plain GPTQ may_write.
    assert not torch.equal(chained[probe], only_gptq[probe])


def test_awq_full_is_exactly_its_two_stage_pipeline():
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
    only_awq = _run_chain(["awq_lite"])
    chained = _run_chain(["awq_lite", "mse"])

    assert set(only_awq) == set(chained)
    assert any(not torch.equal(only_awq[k], chained[k]) for k in only_awq)


def test_handoff_needs_coverage_not_just_overlap(quantized):
    plan = _compile(
        quantized,
        {"module_name": "*layers.0.mlp*", "cfg": ["mse"]},
        {"module_name": "*mlp*", "cfg": ["gptq"]},
        strict=False,
    )
    gptq_index = next(i for i, stage in enumerate(plan) if stage.algo == "gptq")
    assert derive_handoff(quantized, plan, gptq_index) == {}


def test_handoff_fires_when_the_producer_covers_the_consumer(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["mse", "gptq"]})
    assert derive_handoff(quantized, plan, 1) == {"skip_max_init": True}


def test_the_handoff_is_not_user_facing_config():
    import inspect

    from modelopt.torch.quantization.config import GPTQCalibConfig, MseCalibConfig
    from modelopt.torch.quantization.model_calib import gptq, local_hessian_calibrate, mse_calibrate

    # The handoff describes the plan, not the algorithm the user asked for, so it reaches the
    # calibration function as an argument (like `should_process`) and is not a settable field.
    for cfg in (MseCalibConfig, GPTQCalibConfig):
        assert "skip_max_init" not in cfg.model_fields
    for func in (mse_calibrate, gptq, local_hessian_calibrate):
        assert "skip_max_init" in inspect.signature(func).parameters
        assert "should_process" in inspect.signature(func).parameters


def test_unknown_algorithm_is_fatal_even_when_not_strict(quantized):
    with pytest.raises(AlgoCfgValidationError, match="unknown algorithm"):
        _compile(quantized, {"module_name": "*", "cfg": ["nope"]}, strict=False)


def test_fused_siblings_compared_against_the_fallback_too(quantized):
    with pytest.raises(AlgoCfgValidationError, match="fusible siblings"):
        _compile(quantized, {"module_name": "*q_proj", "cfg": ["gptq"]}, algorithm="max")


def test_fused_siblings_with_no_entry_at_all_are_consistent(quantized):
    plan = _compile(quantized, algorithm="max")
    assert [stage.algo for stage in plan] == ["max"]


def test_leading_mse_still_initializes_its_own_amax(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["mse"]})
    assert derive_handoff(quantized, plan, 0) == {}


# ---------------------------------------------------------------------------- scoping


def test_stage_predicate_matches_only_its_own_targets(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["max"]})
    should_process = stage_predicate(quantized, plan[0])
    for name in ("layers.0.mlp.gate_proj", "layers.0.mlp.gate_proj.weight_quantizer"):
        assert should_process(quantized.get_submodule(name))
    for name in ("layers.0.self_attn.q_proj", "layers.0.self_attn.q_proj.weight_quantizer"):
        assert not should_process(quantized.get_submodule(name))


def test_stage_predicate_matches_on_identity_not_name(quantized):
    plan = _compile(quantized, {"module_name": "*mlp*", "cfg": ["max"]})
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


def test_disabled_quantizers_are_not_targets(quantized):
    from modelopt.torch.quantization.algo_cfg import stage_targets

    # QUANT_CFG enables weight and input quantizers and leaves the output ones off.
    _, everything = stage_targets(quantized, _compile(quantized, WHOLE_MODEL)[0])
    assert everything and not any("output_quantizer" in q for q in everything)


def test_a_scope_matching_only_disabled_quantizers_warns_instead_of_failing():
    # An algo_cfg shared across numerics may name a role that this one turns off; that is
    # a no-op worth saying out loud, not a typo worth rejecting.
    model = mtq.quantize(_model(), {"quant_cfg": WEIGHT_ONLY_QUANT_CFG, "algorithm": None}, None)
    with pytest.warns(UserWarning, match="matches only disabled quantizers"):
        _compile(model, {"quantizer_name": "*input_quantizer", "cfg": ["max"]})

    # A glob naming nothing at all is still a rejection.
    with pytest.raises(AlgoCfgValidationError, match="matches no target"):
        _compile(model, {"quantizer_name": "*no_such_quantizer", "cfg": ["max"]})


def test_algorithms_that_ignore_the_write_mask_cannot_be_scoped(quantized):
    with pytest.raises(AlgoCfgValidationError, match="does not honour the scoping write-mask"):
        _compile(quantized, {"module_name": "*mlp*", "cfg": ["lsq"]})


def test_algorithms_that_ignore_the_write_mask_are_still_usable_whole_model(quantized):
    plan = _compile(quantized, {"quantizer_name": "*", "cfg": ["lsq"]})
    assert [stage.algo for stage in plan] == ["lsq"]


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


def test_each_stage_is_recorded_as_its_own_calibration_mode():
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
    assert modes == ["quantize", "max_calibrate", "mse_calibrate"]


def test_a_max_collect_after_mse_does_not_reenter_the_spent_calibrator():
    from modelopt.torch.quantization.model_calib import max_calibrate, mse_calibrate

    # mse installs a search calibrator for the duration of its amax search. Leaving it
    # installed makes any later stats collection re-enter a spent calibrator and fail on its
    # cleared `_initial_amax` -- which only a pipeline can reach.
    model = mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, _forward_loop)
    mse_calibrate(model, _forward_loop)
    max_calibrate(model, _forward_loop)
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


# ---------------------------------------------------------------------------- nvfp4 grid

NVFP4_DYN = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
NVFP4_STA = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}


def _nvfp4_model(wcfg, icfg=None):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 64, bias=False))
    mtq.quantize(
        model,
        {
            "quant_cfg": {
                "default": {"enable": False},
                "*input_quantizer": {**icfg, "enable": True} if icfg else {"enable": False},
                "*weight_quantizer": {**wcfg, "enable": True},
            },
            "algorithm": None,
        },
        forward_loop=None,
    )
    return model


def _prepare(model, stage):
    from modelopt.torch.quantization.algo_cfg import stage_targets
    from modelopt.torch.quantization.mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    descriptor = CalibrateModeRegistry[BaseCalibrateModeDescriptor._get_mode_name(stage.algo)]
    _, quantizers = stage_targets(model, stage)
    return type(descriptor).prepare(model, quantizers, stage.cfg)


def test_awq_needs_a_dynamic_grid_and_static_never_downgrades():
    with pytest.raises(AlgoCfgValidationError, match="needs a dynamic NVFP4 weight grid"):
        _compile(_nvfp4_model(NVFP4_STA), {"module_name": "*", "cfg": ["awq_lite"]})


SWEEP = {"method": "mse", "fp8_scale_sweep": True}


def test_fp8_scale_sweep_requires_static_and_a_dynamic_grid_is_upgraded():
    from modelopt.torch.quantization.algo_cfg import capabilities_for

    assert capabilities_for("mse", {"fp8_scale_sweep": True}).requires_weight_scales == "static"
    assert capabilities_for("mse").requires_weight_scales is None

    model = _nvfp4_model(NVFP4_DYN)
    stage = _compile(model, {"module_name": "*", "cfg": [SWEEP]})[0]
    assert model[0].weight_quantizer.block_sizes["type"] == "dynamic"
    assert _prepare(model, stage)
    assert model[0].weight_quantizer.block_sizes["type"] == "static"


def test_upgrading_the_weight_grid_leaves_activations_alone():
    # The shipped static presets pair a static weight grid with dynamic activations on
    # purpose; the upgrade must not reach across and convert the input quantizer too.
    model = _nvfp4_model(NVFP4_DYN, icfg=NVFP4_DYN)
    stage = _compile(model, {"quantizer_name": "*", "cfg": [SWEEP]})[0]
    _prepare(model, stage)
    assert model[0].weight_quantizer.block_sizes["type"] == "static"
    assert model[0].input_quantizer.block_sizes["type"] == "dynamic"


def test_a_grid_upgrade_earlier_in_the_plan_is_visible_to_later_stages():
    # Validating against the model as it is now would accept this: the weight grid is
    # dynamic until the sweep stage upgrades it, which happens before awq_clip runs.
    model = _nvfp4_model(NVFP4_DYN)
    with pytest.raises(AlgoCfgValidationError, match="'mse' upgraded them"):
        _compile(model, {"module_name": "*", "cfg": [SWEEP, "awq_clip"]})

    # ... and the same pair in the order that works is still accepted.
    _compile(_nvfp4_model(NVFP4_DYN), {"module_name": "*", "cfg": ["awq_clip", SWEEP]})


def test_a_static_activation_grid_does_not_block_a_weight_side_algorithm():
    model = _nvfp4_model(NVFP4_DYN, icfg=NVFP4_STA)
    _compile(model, {"module_name": "*", "cfg": ["awq_clip"]})
