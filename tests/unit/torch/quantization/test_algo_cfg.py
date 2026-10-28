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
    _index_model,
    compile_algo_cfg,
    derive_handoff,
    resolve_targets,
    stage_predicate,
    stage_targets,
)
from modelopt.torch.quantization.config import AlgoCfgEntry
from modelopt.torch.quantization.model_calib import smoothquant
from modelopt.torch.quantization.nn import (
    GroupedQuantizer,
    QuantModule,
    SequentialQuantizer,
    TensorQuantizer,
)

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


# NVFP4 with a *static* weight grid -- the shape that carries a shared `global_amax` between
# fused siblings. Other numerics have no such state, so separating their siblings is ordinary
# scoping, not a split group.
NVFP4_STATIC_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {
        "quantizer_name": "*weight_quantizer",
        "cfg": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
        },
    },
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


class _MLA(nn.Module):
    """The pair ``_sync_mla_fused_down_proj_amax`` reconciles, found by attribute name."""

    def __init__(self, d=32):
        super().__init__()
        self.linear_q_down_proj = nn.Linear(d, d, bias=False)
        self.linear_kv_down_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.linear_q_down_proj(x) + self.linear_kv_down_proj(x)


class _SparseMoE(nn.Module):
    """Carries the hook ``max_calibrate`` calls to reconcile a layer's local experts.

    The container is named per framework: HF uses ``experts``, Megatron ``local_experts``.
    """

    def __init__(self, d=32, n=2, container="experts"):
        super().__init__()
        setattr(self, container, nn.ModuleList([_MLP(d) for _ in range(n)]))
        self._container = container

    def forward(self, x):
        return sum(e(x) for e in getattr(self, self._container))

    def layer_sync_moe_local_experts_amax(self, sync_weight_amax=False):
        return None


class _FusedExperts(QuantModule):
    """Per-projection quantizers, so ``is_quantized_linear`` is False -- an HF fused wrapper."""

    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizers = nn.ModuleList([TensorQuantizer() for _ in range(2)])
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizers = nn.ModuleList([TensorQuantizer() for _ in range(2)])

    def forward(self, x):
        return x


def _wrap(part, **kwargs):
    """A one-block model around ``part``, so every scope glob has a prefix to match."""

    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = part(**kwargs)

        def forward(self, x):
            return self.block(x)

    torch.manual_seed(0)
    return _Wrapped().eval()


def _fused_expert_model():
    """A fused wrapper whose quantizers exist but whose module is not a quantized linear."""

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, 8, 8))

        def forward(self, x):
            return x

    class _MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _Holder()
            self.experts.__class__ = _FusedExperts
            self.experts._setup()

        def forward(self, x):
            return self.experts(x)

    return _MoE()


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


def test_an_unscopable_algorithm_runs_without_being_handed_a_predicate():
    """Compiling is not enough: the executor must not pass ``should_process`` to one.

    A non-scopable calib func does not take the argument, and most only tolerate it by
    swallowing ``**kwargs``. ``nvfp4_act_headroom`` does not, so it raises ``TypeError``.
    """
    mtq.quantize(
        _model(),
        {
            "quant_cfg": QUANT_CFG,
            "algorithm": None,
            "algo_cfg": [{"module_name": "*", "cfg": ["nvfp4_act_headroom"]}],
        },
        _forward_loop,
    )


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


# --- Unsupported combinations are refused, or warn where refusing is not possible -----


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


def test_a_stages_kwargs_are_validated_when_the_plan_compiles(quantized):
    """Otherwise a typo in stage 2 surfaces only after stage 1 has calibrated and mutated."""
    with pytest.raises(AlgoCfgValidationError, match="block_sizee"):
        _compile(
            quantized,
            AlgoCfgEntry(module_name="*", cfg=["mse", {"method": "gptq", "block_sizee": 64}]),
        )


# --- Layouts the resolver must still reach --------------------------------------------


def test_an_empty_stage_does_not_run():
    """An empty scope is warned about, not refused, so the stage still reaches the executor.
    An unscopable algorithm drops the predicate there and would run over the whole model --
    the opposite of the nothing the warning promised.
    """
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 32}}},
    ]
    torch.manual_seed(0)
    model = mtq.quantize(_model(), {"quant_cfg": quant_cfg, "algorithm": None}, None)
    quantizer = model.layers[0].self_attn.o_proj.weight_quantizer
    assert getattr(quantizer, "_amax", None) is None

    # `*input_quantizer` matches only disabled quantizers here, so the stage is empty.
    with pytest.warns(UserWarning, match="will do nothing"):
        mtq.quantize(
            model,
            {
                "quant_cfg": quant_cfg,
                "algorithm": None,
                "algo_cfg": [{"quantizer_name": "*input_quantizer", "cfg": ["nvfp4_act_headroom"]}],
            },
            _forward_loop,
        )

    assert getattr(quantizer, "_amax", None) is None, "an empty stage calibrated out of scope"


def test_a_disabled_level_does_not_look_like_a_split_chain():
    """A disabled level is not a target of any scope, so it cannot be one a selector left out.
    Counting it made every whole-model scope look like it split the chain, and no widening
    could satisfy the check.
    """
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [{"num_bits": 4, "block_sizes": {-1: 32}}, {"num_bits": (4, 3), "axis": None}],
        },
    ]
    model = mtq.quantize(_model(), {"quant_cfg": quant_cfg, "algorithm": None}, None)
    for _, module in model.named_modules():
        if isinstance(module, SequentialQuantizer):
            module[0].disable()

    plan = _compile(model, AlgoCfgEntry(module_name="*", cfg=["max"]))
    assert plan


def test_a_disabled_fused_expert_role_is_an_empty_scope_not_an_error():
    """`_matches_only_disabled` must use the same matcher as everything else, or a recipe naming
    a disabled fused-expert role is rejected as a typo instead of warned about.
    """

    class _FusedExperts(QuantModule):
        def _setup(self):
            self.gate_up_proj_input_quantizers = nn.ModuleList(
                [TensorQuantizer() for _ in range(2)]
            )
            self.gate_up_proj_weight_quantizers = nn.ModuleList(
                [TensorQuantizer() for _ in range(2)]
            )

        def forward(self, x):
            return x

    class _Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, 8, 8))

        def forward(self, x):
            return x

    class _MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _Experts()
            self.experts.__class__ = _FusedExperts
            self.experts._setup()

        def forward(self, x):
            return self.experts(x)

    model = _MoE()
    for quantizer in model.experts.gate_up_proj_weight_quantizers:
        quantizer.enable()
    for quantizer in model.experts.gate_up_proj_input_quantizers:
        quantizer.disable()

    with pytest.warns(UserWarning, match="will do nothing"):
        _compile(model, AlgoCfgEntry(quantizer_name="experts.*input_quantizer", cfg=["max"]))


def test_a_scope_giving_smoothquant_one_role_compiles_and_says_so():
    """SmoothQuant divides the activation and multiplies the weight, so a module whose scope
    admits only one side is skipped at runtime -- and the stage still reports success.
    """
    model = mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)

    assert _compile(model, AlgoCfgEntry(quantizer_name="*input_quantizer", cfg=["smoothquant"]))
    assert _compile(model, AlgoCfgEntry(module_name="*", cfg=["smoothquant"]))

    stage = _compile(model, AlgoCfgEntry(quantizer_name="*input_quantizer", cfg=["smoothquant"]))[0]
    with pytest.warns(UserWarning, match="could not be smoothed"):
        smoothquant(model, _forward_loop, should_process=stage_predicate(model, stage))


def test_a_weight_only_config_still_pairs_for_awq():
    """A disabled input quantizer is not a split pair: nothing indexes it, and weight-only AWQ
    is exactly the configuration that must keep working.
    """
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 32}}},
    ]
    model = mtq.quantize(_model(), {"quant_cfg": quant_cfg, "algorithm": None}, None)

    assert _compile(model, AlgoCfgEntry(module_name="*mlp*", cfg=["awq_lite"]))


# --- Group refusals: a scope may not split what calibration reconciles ------------------

INPUTS_ONLY = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]
SEQ_WEIGHTS_ONLY = [
    {"quantizer_name": "*", "enable": False},
    {
        "quantizer_name": "*weight_quantizer",
        "cfg": [{"num_bits": 4, "block_sizes": {-1: 32}}, {"num_bits": (4, 3), "axis": None}],
    },
]
WITH_OUTPUTS = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
    {"quantizer_name": "*output_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]
PER_CHANNEL_WEIGHTS_ONLY = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
]

# (id, model, quant cfg, selector, scope, refused_matching)
#
# `refused_matching` is the text the refusal must carry, or None when the plan is legitimate.
# Accepting is as much a claim as refusing: every rule here has at some point rejected a plan
# the write-mask handles correctly, which no widening could fix.
GROUP_CASES = [
    # q/k/v share one NVFP4-static `global_amax`; nothing else shares anything.
    ("qkv/shared/split", _model, NVFP4_STATIC_QUANT_CFG, "module_name", "*q_proj*", "fused amax"),
    ("qkv/shared/whole", _model, NVFP4_STATIC_QUANT_CFG, "module_name", "*self_attn*", None),
    ("qkv/int4/split", _model, QUANT_CFG, "module_name", "*q_proj*", None),
    ("qkv/inputs/split", _model, INPUTS_ONLY, "module_name", "*q_proj*", None),
    # The MLA pair is reconciled per-tensor, for whichever roles hold a scalar range.
    ("mla/sequential/split", _MLA, SEQ_WEIGHTS_ONLY, "module_name", "*q_down_proj*", "fused amax"),
    ("mla/per_channel/split", _MLA, PER_CHANNEL_WEIGHTS_ONLY, "module_name", "*q_down_proj*", None),
    # Local experts: input amax is reconciled across the layer, and missing weight amax filled.
    ("experts/hf/one", _SparseMoE, QUANT_CFG, "module_name", "*experts.0*", "local experts"),
    ("experts/hf/all", _SparseMoE, QUANT_CFG, "module_name", "*experts*", None),
    (
        "experts/hf/one_role",
        _SparseMoE,
        QUANT_CFG,
        "quantizer_name",
        "*input_quantizer",
        "local experts",
    ),
    # Megatron names the same container differently; missing it lets its scopes through.
    (
        "experts/megatron/one",
        _SparseMoE,
        QUANT_CFG,
        "module_name",
        "*local_experts.0*",
        "local experts",
    ),
    # The sync keeps a sequential chain's per-tensor range in its *last* level, so the group
    # must name that leaf. Naming the container instead leaves a scope on `.1` intersecting
    # nothing, and it passes validation while the sibling's `.1` gets rewritten.
    (
        "mla/sequential/level_1",
        _MLA,
        SEQ_WEIGHTS_ONLY,
        "quantizer_name",
        "*linear_q_down_proj.weight_quantizer.1",
        "fused amax",
    ),
    # A role nothing shares: the weight `global_amax` is fused across q/k/v, the inputs are not.
    (
        "qkv/shared/one_input",
        _model,
        NVFP4_STATIC_QUANT_CFG,
        "quantizer_name",
        "*q_proj.input_quantizer",
        None,
    ),
    # Enabled output quantizers must not widen what the group demands: `sync_moe_expert_amax`
    # writes the input and weight roles only. No single glob selects those two without the
    # outputs, so this covers the whole layer and pins that the extra role changes nothing.
    (
        "experts/with_outputs/all",
        _SparseMoE,
        WITH_OUTPUTS,
        "module_name",
        "*experts*",
        None,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "part", "quant_cfg", "selector", "scope", "refused_matching"),
    GROUP_CASES,
    ids=[c[0] for c in GROUP_CASES],
)
def test_a_group_scope_is_refused_only_when_something_is_shared(
    case_id, part, quant_cfg, selector, scope, refused_matching
):
    kwargs = {"container": "local_experts"} if "megatron" in case_id else {}
    model = _wrap(part, **kwargs) if part is not _model else _model()
    model = mtq.quantize(model, {"quant_cfg": quant_cfg, "algorithm": None}, None)
    entry = AlgoCfgEntry(**{selector: scope, "cfg": ["max"]})

    if refused_matching:
        with pytest.raises(AlgoCfgValidationError, match=refused_matching):
            _compile(model, entry, algorithm="max")
    else:
        assert _compile(model, entry, algorithm="max")


def test_a_wholly_disabled_sibling_does_not_make_a_group_unsatisfiable():
    """Reconciliation skips disabled quantizers, so such a sibling is never written. Counting
    it would reject even `module_name: "*"`, which no widening can fix.
    """
    model = mtq.quantize(_wrap(_MLA), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)
    for quantizer in model.block.linear_kv_down_proj.modules():
        if isinstance(quantizer, TensorQuantizer):
            quantizer.disable()

    assert _compile(model, AlgoCfgEntry(module_name="*", cfg=["max"]))


# --- Resolution: a scope reaches the quantizers that do the work ------------------------

# (id, model, quant cfg, selector, scope, module names, quantizer-name suffixes)
#
# Each case is a layout where the resolver once reached the wrong thing: a container without
# its leaves, a leaf without its owner, a fused wrapper owning neither.
RESOLVE_CASES = [
    (
        "sequential/by_module",
        lambda: _model(),
        SEQUENTIAL_QUANT_CFG,
        "module_name",
        "*mlp*",
        None,
        (".0", ".1"),
    ),
    (
        "sequential/by_container",
        lambda: _model(),
        SEQUENTIAL_QUANT_CFG,
        "quantizer_name",
        "*weight_quantizer",
        None,
        (".0", ".1"),
    ),
    (
        "fused_wrapper/by_quantizer",
        _fused_expert_model,
        None,
        "quantizer_name",
        "*down_proj_weight_quantizers*",
        {"experts"},
        (),
    ),
    (
        "fused_wrapper/singular_glob",
        _fused_expert_model,
        None,
        "quantizer_name",
        "*weight_quantizer",
        {"experts"},
        ("weight_quantizers.0",),
    ),
]


@pytest.mark.parametrize(
    ("case_id", "build", "quant_cfg", "selector", "scope", "modules", "suffixes"),
    RESOLVE_CASES,
    ids=[c[0] for c in RESOLVE_CASES],
)
def test_a_scope_resolves_the_quantizers_that_do_the_work(
    case_id, build, quant_cfg, selector, scope, modules, suffixes
):
    model = build()
    if quant_cfg is not None:
        model = mtq.quantize(model, {"quant_cfg": quant_cfg, "algorithm": None}, None)
    else:
        for quantizer in model.modules():
            if isinstance(quantizer, TensorQuantizer):
                quantizer.enable()

    plan = _compile(model, AlgoCfgEntry(**{selector: scope, "cfg": ["max"]}))
    got_modules, got_quantizers = stage_targets(model, plan[0])

    assert got_quantizers, f"{case_id}: nothing resolved"
    if modules is not None:
        assert got_modules == modules, f"{case_id}: owners {got_modules}"
    for suffix in suffixes:
        assert any(q.endswith(suffix) for q in got_quantizers), (
            f"{case_id}: nothing ending {suffix!r} in {sorted(got_quantizers)}"
        )
    predicate = stage_predicate(model, plan[0])
    assert all(predicate(model.get_submodule(q)) for q in got_quantizers)


def test_a_bare_quantized_linear_is_its_own_scope():
    """`named_modules()` names the root "", a depth the ancestor walk must still consider."""
    model = mtq.quantize(nn.Linear(32, 32), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)
    plan = _compile(model, AlgoCfgEntry(module_name="*", cfg=["max"]))
    modules, quantizers = stage_targets(model, plan[0])

    assert modules == {""} and quantizers
    assert stage_predicate(model, plan[0])(model)


def test_an_unscopable_algorithm_may_only_run_whole_model():
    """It ignores the mask, so a partial scope would write outside itself while claiming not to."""
    model = mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)

    with pytest.raises(AlgoCfgValidationError, match="does not honour the scoping write-mask"):
        _compile(model, AlgoCfgEntry(module_name="*mlp*", cfg=["nvfp4_act_headroom"]))
    assert _compile(model, AlgoCfgEntry(module_name="*", cfg=["nvfp4_act_headroom"]))


# --- The plan must be the same on every rank ------------------------------------------


def _plan_digest(model, config):
    """What a stage will do, reduced to something comparable across processes.

    Not the plan object: it holds predicates and module references, which are per-process. The
    order, the algorithm, and the resolved target *names* are what must agree -- those decide
    which quantizers each rank writes, and so which collectives its algorithms reach.
    """
    plan = compile_algo_cfg(config, model)
    return [
        (i, stage.algo, stage.selector, stage.scope, tuple(sorted(stage_targets(model, stage)[1])))
        for i, stage in enumerate(plan)
    ]


def _plan_is_rank_identical_job(rank, size):
    import torch.distributed as dist

    torch.manual_seed(0)
    model = mtq.quantize(_model(), {"quant_cfg": QUANT_CFG, "algorithm": None}, None)
    config = {
        "algo_cfg": [
            {"module_name": "*self_attn*", "cfg": ["max"]},
            {"quantizer_name": "*mlp*weight_quantizer", "cfg": ["max"]},
        ],
        "algorithm": "max",
    }
    digest = _plan_digest(model, config)

    gathered = [None] * size
    dist.all_gather_object(gathered, digest)
    assert all(other == gathered[0] for other in gathered), (
        f"rank {rank} compiled a different plan:\n  {digest}\n  vs rank 0:\n  {gathered[0]}"
    )


def test_the_plan_is_identical_on_every_rank(cpu_dist_workers):
    """Scoping must never make one rank disagree with another about what runs.

    Several calibration algorithms reach a collective, and a rank that resolved a different
    target set reaches a different one -- which hangs rather than errors. Compiling is a pure
    function of the config and the model's structure precisely so this cannot happen; this
    gathers each rank's plan and compares, rather than running a collective that could hang.
    """
    cpu_dist_workers.run(_plan_is_rank_identical_job)
