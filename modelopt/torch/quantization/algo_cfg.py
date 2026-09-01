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

"""Compile a quantize config into an ordered list of scoped calibration stages.

This is the *compile* half of the calibration plan (the ``calibration_plan`` mode in
:mod:`~modelopt.torch.quantization.mode` is the *execute* half).  It is deliberately
side-effect free: it reads the already-quantized model's **structure** — quantizer and
linear names — to resolve globs and validate, but it mutates nothing, runs no forward and
touches no data.  Consequences the design leans on:

* bad configs fail fast, before any expensive calibration runs;
* it is testable without running a model;
* the resulting plan is a pure function of ``(config, model structure)``, so it is
  identical on every rank — which is what keeps predicate scoping from desynchronizing
  collectives in distributed calibration.

Both surfaces lower here.  ``algorithm="max"`` becomes the single all-``"*"`` stage, so
the legacy whole-model path is a special case of the scoped one rather than a second
engine.
"""

import fnmatch
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch.nn as nn

from .config import AlgoCfgEntry, QuantizeAlgorithmConfig, QuantizeConfig

__all__ = [
    "AlgoCapabilities",
    "AlgoCfgValidationError",
    "AlgoStage",
    "CalibrationPlan",
    "compile_algo_cfg",
    "describe_plan",
    "plan_hash",
    "stage_predicate",
]


class AlgoCfgValidationError(ValueError):
    """Raised when an ``algo_cfg`` cannot be lowered into a valid plan."""


# --------------------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgoCapabilities:
    """What one calibration algorithm consumes, produces and needs to run.

    Only the fields the compiler actually uses today are declared.  ``produces`` /
    ``requires`` are a small open vocabulary of state tokens:

    ``weight``, ``weight_amax``, ``input_amax``, ``pre_quant_scale``, ``acts``.
    """

    granularity: str  # "tensor" | "module"
    role: str  # "weight" | "input" | "both" — which quantizers it may write
    requires: frozenset[str] = frozenset()
    produces: frozenset[str] = frozenset()
    needs_forward: bool = True
    self_forwards: bool = False  # needs its own forward pass, cannot share one
    # State tokens that must NOT already be present for this algorithm to be correct.
    # ``awq_lite`` folds ``1/s`` into the weight and assumes it is starting from an
    # unsmoothed weight; running it twice folds twice while keeping only the last
    # activation-side scale (see ``apply_pre_quant_scale_and_smooth``).
    requires_absent: frozenset[str] = frozenset()

    @property
    def shareable_forward(self) -> bool:
        """Whether this stage could ride a forward pass shared with other stages."""
        return self.needs_forward and not self.self_forwards


_W_AMAX = "weight_amax"
_I_AMAX = "input_amax"
_PQS = "pre_quant_scale"
_W = "weight"

#: Declared capabilities per algorithm.  Phase 1 of the design defers this contract, but
#: the two failure modes that make ``awq_lite -> mse -> awq_lite`` wrong are only
#: detectable with it, so a minimal table ships here.  ``None`` (no calibration) is absent
#: on purpose — it compiles to an empty plan.
ALGO_CAPABILITIES: dict[str, AlgoCapabilities] = {
    "max": AlgoCapabilities(
        granularity="tensor", role="both", produces=frozenset({_W_AMAX, _I_AMAX})
    ),
    "mse": AlgoCapabilities(
        granularity="tensor",
        role="weight",
        requires=frozenset({_W, _W_AMAX}),
        produces=frozenset({_W_AMAX}),
    ),
    "nvfp4_act_headroom": AlgoCapabilities(
        granularity="tensor",
        role="input",
        requires=frozenset({"acts"}),
        produces=frozenset({_I_AMAX}),
    ),
    "local_hessian": AlgoCapabilities(
        granularity="module",
        role="weight",
        requires=frozenset({_W, _W_AMAX, "acts"}),
        produces=frozenset({_W_AMAX}),
    ),
    "smoothquant": AlgoCapabilities(
        granularity="module",
        role="both",
        requires=frozenset({"acts"}),
        produces=frozenset({_PQS, _I_AMAX, _W, _W_AMAX}),
        requires_absent=frozenset({_PQS}),
    ),
    "awq_lite": AlgoCapabilities(
        granularity="module",
        role="both",
        requires=frozenset({"acts", _W}),
        produces=frozenset({_PQS, _W_AMAX, _I_AMAX}),
        self_forwards=True,
        requires_absent=frozenset({_PQS}),
    ),
    "awq_clip": AlgoCapabilities(
        granularity="module",
        role="weight",
        requires=frozenset({"acts", _W, _W_AMAX}),
        produces=frozenset({_W_AMAX}),
        self_forwards=True,
    ),
    "awq_full": AlgoCapabilities(
        granularity="module",
        role="both",
        requires=frozenset({"acts", _W}),
        produces=frozenset({_PQS, _W_AMAX, _I_AMAX}),
        self_forwards=True,
        requires_absent=frozenset({_PQS}),
    ),
    "gptq": AlgoCapabilities(
        granularity="module",
        role="weight",
        requires=frozenset({_W, "acts"}),
        produces=frozenset({_W, _W_AMAX}),
        self_forwards=True,
    ),
    "svdquant": AlgoCapabilities(
        granularity="module",
        role="both",
        requires=frozenset({"acts", _W}),
        produces=frozenset({_PQS, _W, _W_AMAX, _I_AMAX}),
        self_forwards=True,
        requires_absent=frozenset({_PQS}),
    ),
    "lsq": AlgoCapabilities(
        granularity="tensor",
        role="weight",
        requires=frozenset({_W, _W_AMAX}),
        produces=frozenset({_W_AMAX}),
    ),
}


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgoStage:
    """One algorithm applied to one scope — the unit of work in a calibration plan."""

    algo: str | None
    cfg: dict  # kwargs for the algorithm, including its "method" key
    scope: str  # the glob
    selector: str  # "module_name" | "quantizer_name"
    order: int  # position within its entry's pipeline
    entry: int  # which algo_cfg entry it came from (-1 = the `algorithm` fallback)
    # Scopes this stage must NOT touch, as ``(selector, glob)`` pairs.  The model-wide
    # ``algorithm`` fallback covers "everything an algo_cfg entry did not match", which is a
    # complement and so cannot be written as a glob.  Keeping it as globs-minus-globs (rather
    # than a resolved name list) preserves the property the plan depends on: it is derived
    # from the config alone, so every rank computes the same thing.
    exclude: tuple[tuple[str, str], ...] = ()

    @property
    def capabilities(self) -> AlgoCapabilities | None:
        """Declared capabilities of this stage's algorithm, or ``None`` if undeclared.

        Looked up rather than copied onto the stage: capabilities describe the *algorithm*,
        so a stage that carried its own copy could drift from the registry.
        """
        return ALGO_CAPABILITIES.get(self.algo) if self.algo else None

    def key(self) -> tuple:
        """Execution-relevant identity, used for the plan hash.

        ``entry`` is deliberately excluded: it records *where in the config* a stage came
        from, which is provenance, not behaviour. Leaving it out is what makes the legacy
        ``algorithm="max"`` plan and the explicit ``[{"quantizer_name": "*", "cfg": ["max"]}]``
        plan hash identical -- the same execution, written two ways.
        """
        return (
            self.algo,
            tuple(sorted(self.cfg.items(), key=str)),
            self.scope,
            self.selector,
            self.order,
            tuple(sorted(self.exclude)),
        )

    def __str__(self) -> str:
        extra = {k: v for k, v in self.cfg.items() if k != "method"}
        extra_s = f" {extra}" if extra else ""
        excl = f" minus {[g for _, g in self.exclude]}" if self.exclude else ""
        return (
            f"{self.algo or 'none'} @ {self.selector}={self.scope!r}{excl} (#{self.order}){extra_s}"
        )


CalibrationPlan = list[AlgoStage]


# --------------------------------------------------------------------------------------
# Target resolution
# --------------------------------------------------------------------------------------


@dataclass
class _ModelIndex:
    """Names of the things a scope can select, read once from the model structure."""

    linears: list[str] = field(default_factory=list)
    quantizers: list[str] = field(default_factory=list)
    quantizers_of: dict[str, list[str]] = field(default_factory=dict)  # linear -> quantizers
    parent_of: dict[str, str] = field(default_factory=dict)  # quantizer -> linear


def _index_model(model: nn.Module) -> _ModelIndex:
    # Imported lazily: `mode` imports this module while `modelopt.torch.quantization` is
    # still initializing, and `.nn` pulls in the quantized-tensor backends.
    from .nn import SequentialQuantizer, TensorQuantizer
    from .utils import is_quantized_linear

    index = _ModelIndex()
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            index.linears.append(name)
            index.quantizers_of[name] = []
        elif isinstance(module, (TensorQuantizer, SequentialQuantizer)):
            index.quantizers.append(name)
    for q in index.quantizers:
        parent = q.rsplit(".", 1)[0] if "." in q else ""
        if parent in index.quantizers_of:
            index.quantizers_of[parent].append(q)
            index.parent_of[q] = parent
    return index


def resolve_targets(model: nn.Module, scope: str, selector: str) -> tuple[set[str], set[str]]:
    """Resolve a scope into ``(module names, quantizer names)``.

    A ``module_name`` scope pulls in that module's quantizers; a ``quantizer_name`` scope
    pulls in the owning modules, so a stage's write-mask covers whichever name its
    algorithm happens to iterate over.
    """
    index = _index_model(model)
    if selector == "module_name":
        modules = {n for n in index.linears if fnmatch.fnmatch(n, scope)}
        quantizers = {q for m in modules for q in index.quantizers_of[m]}
    else:
        quantizers = {n for n in index.quantizers if fnmatch.fnmatch(n, scope)}
        modules = {index.parent_of[q] for q in quantizers if q in index.parent_of}
    return modules, quantizers


#: Which quantizer a state token lives on.  ``weight`` (the tensor itself) rides with the
#: weight quantizer for scoping purposes.
TOKEN_ROLE: dict[str, str] = {
    "weight": "weight",
    "weight_amax": "weight",
    "input_amax": "input",
    "pre_quant_scale": "input",
}


def stage_targets(model: nn.Module, stage: AlgoStage) -> tuple[set[str], set[str]]:
    """``(modules, quantizers)`` a stage may write, after subtracting its exclusions."""
    modules, quantizers = resolve_targets(model, stage.scope, stage.selector)
    for selector, glob in stage.exclude:
        ex_modules, ex_quantizers = resolve_targets(model, glob, selector)
        modules -= ex_modules
        quantizers -= ex_quantizers
    return modules, quantizers


def role_quantizers(model: nn.Module, stage: AlgoStage) -> dict[str, set[str]]:
    """The quantizers a stage may write, split by role.

    A stage's scope can pull in quantizers its algorithm will never touch — a
    ``module_name`` scope resolves to both the weight and the input quantizer, but ``mse``
    only writes weights.  Overlap between stages has to be judged on what they can actually
    write, otherwise two stages that share a module but write different roles look like they
    conflict when they do not.
    """
    _, quantizers = stage_targets(model, stage)
    caps = stage.capabilities
    role = caps.role if caps else "both"
    weight = {q for q in quantizers if "weight_quantizer" in q}
    inp = quantizers - weight
    return {
        "weight": weight if role in ("weight", "both") else set(),
        "input": inp if role in ("input", "both") else set(),
    }


def effective_produces(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Tokens a stage actually writes here — declared ``produces`` minus roles it cannot reach."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.produces if by_role[TOKEN_ROLE.get(t, "weight")]}


def effective_requires(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Non-ambient tokens a stage reads here."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.requires - AMBIENT_TOKENS if by_role[TOKEN_ROLE.get(t, "weight")]}


def _token_overlap(model: nn.Module, a: AlgoStage, b: AlgoStage, token: str) -> bool:
    """Whether two stages can write the same ``token`` on the same quantizers."""
    role = TOKEN_ROLE.get(token, "weight")
    return bool(role_quantizers(model, a)[role] & role_quantizers(model, b)[role])


def stage_predicate(model: nn.Module, stage: AlgoStage) -> Callable[[str], bool]:
    """Build the ``should_process`` write-mask for a stage.

    The predicate is AND-ed into each algorithm's existing ``is_enabled`` filter, so a
    stage **writes only its targets and never toggles enable-state** — reads (and hence
    the activations seen by search-based algorithms like AWQ and GPTQ) are unchanged.
    """
    modules, quantizers = stage_targets(model, stage)
    allowed = modules | quantizers
    return lambda name: name in allowed


# --------------------------------------------------------------------------------------
# Lowering
# --------------------------------------------------------------------------------------


def _algo_to_name_and_cfg(algo) -> tuple[str | None, dict]:
    """Normalize one pipeline element to ``(algo_name, kwargs)``."""
    if isinstance(algo, QuantizeAlgorithmConfig):
        algo = algo.model_dump()
    if algo is None or isinstance(algo, str):
        return algo, {"method": algo}
    if isinstance(algo, dict):
        if "method" not in algo:
            raise AlgoCfgValidationError(
                f"Algorithm dict must have a 'method' key; got {sorted(algo)}. Entry: {algo!r}"
            )
        return algo["method"], dict(algo)
    raise AlgoCfgValidationError(f"Invalid algorithm config type {type(algo)}: {algo!r}")


def _lower(entries: Iterable[AlgoCfgEntry], algorithm) -> CalibrationPlan:
    """Config -> stages.  No model needed; validation of names happens separately."""
    plan: CalibrationPlan = []
    for e_idx, entry in enumerate(entries):
        selector, scope = entry.selector
        for order, algo in enumerate(entry.cfg):
            name, cfg = _algo_to_name_and_cfg(algo)
            plan.append(AlgoStage(name, cfg, scope, selector, order, e_idx))

    # The model-wide `algorithm` is the same thing at scope "*" -- one engine, not two.
    # It is the *fallback*, so it must not re-run over targets an entry already claimed;
    # otherwise the default would silently overwrite every scoped pipeline.
    if algorithm is not None:
        claimed = tuple(entry.selector for entry in entries)
        algos = algorithm if isinstance(algorithm, list) else [algorithm]
        for order, algo in enumerate(algos):
            name, cfg = _algo_to_name_and_cfg(algo)
            if name is None:
                continue
            plan.append(AlgoStage(name, cfg, "*", "quantizer_name", order, -1, exclude=claimed))
    return plan


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

#: Sibling linears that are fused into one kernel at export and therefore must share a
#: single weight scale — so they must also share one pipeline.
FUSED_SIBLING_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("w1", "w3"),
)


def _report(msg: str, strict: bool = True, sink: list[str] | None = None) -> None:
    """Record a validation violation.

    Violations are collected rather than raised on the first hit so one compile reports
    everything wrong with a config -- a config with three mistakes should not take three
    round trips to fix. ``strict=False`` turns the whole set into warnings.
    """
    if sink is not None:
        sink.append(msg)
        return
    if strict:
        raise AlgoCfgValidationError(msg)
    warnings.warn(f"algo_cfg: {msg}", stacklevel=3)


def known_algorithms() -> list[str]:
    """Algorithm names currently registered in the calibrate-mode registry."""
    from .mode import CalibrateModeRegistry

    names = getattr(CalibrateModeRegistry, "_name2descriptor", {})
    return sorted(
        n.removesuffix("_calibrate")
        for n in names
        if n.endswith("_calibrate") and not n.startswith("_")
    )


def _validate_config_only(plan: CalibrationPlan, strict: bool) -> None:
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    for stage in plan:
        mode_name = BaseCalibrateModeDescriptor._get_mode_name(stage.algo)
        if mode_name not in CalibrateModeRegistry:
            _report(
                f"unknown algorithm {stage.algo!r}. Known algorithms: {known_algorithms()}",
                strict=True,
            )


def _validate_scopes(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    for stage in plan:
        modules, quantizers = resolve_targets(model, stage.scope, stage.selector)
        if not modules and not quantizers:
            _report(
                f"scope {stage.selector}={stage.scope!r} (stage {stage}) matches no target in "
                "the model. Check the glob against the quantized module/quantizer names.",
                sink=sink,
            )
            continue

        caps = stage.capabilities
        if caps is None:
            continue
        # Role check: a weight-only algorithm pointed at input quantizers writes nothing.
        if stage.selector == "quantizer_name":
            roles = {"weight" if "weight_quantizer" in q else "input" for q in quantizers}
            if caps.role != "both" and roles and caps.role not in roles:
                _report(
                    f"{stage.algo!r} only writes {caps.role} quantizers but "
                    f"{stage.selector}={stage.scope!r} matches only {sorted(roles)} quantizers "
                    "— the stage would be a no-op.",
                    sink=sink,
                )


def _validate_fused_siblings(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """Fusible siblings must share one pipeline: one fused kernel, one weight scale."""
    pipeline_of: dict[str, tuple] = {}
    for stage in plan:
        if stage.entry < 0:
            continue  # the "*" fallback covers everything equally
        modules, _ = resolve_targets(model, stage.scope, stage.selector)
        for m in modules:
            pipeline_of.setdefault(m, ())
    for stage in plan:
        if stage.entry < 0:
            continue
        modules, _ = resolve_targets(model, stage.scope, stage.selector)
        for m in modules:
            pipeline_of[m] = (*pipeline_of[m], (stage.algo, stage.entry))

    index = _index_model(model)
    for group in FUSED_SIBLING_GROUPS:
        by_parent: dict[str, dict[str, tuple]] = {}
        for linear in index.linears:
            leaf = linear.rsplit(".", 1)[-1]
            if leaf in group and linear in pipeline_of:
                by_parent.setdefault(linear.rsplit(".", 1)[0], {})[leaf] = pipeline_of[linear]
        for parent, members in by_parent.items():
            distinct = {tuple(a for a, _ in v) for v in members.values()}
            if len(distinct) > 1:
                _report(
                    f"fusible siblings under {parent!r} got different pipelines "
                    f"({ {k: [a for a, _ in v] for k, v in members.items()} }). They export to one "
                    "fused kernel and must share a single weight scale, so they must share one "
                    "pipeline.",
                    sink=sink,
                )


def _validate_dependencies(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """Capability-derived checks: non-composable repeats and dead stages.

    Both are judged **per token and per role**: two stages conflict only when they write the
    same state token on the same quantizers. A stage whose scope happens to include a module
    another stage also touches is not a conflict if the two write different roles.
    """
    for i, stage in enumerate(plan):
        caps = stage.capabilities
        if caps is None:
            continue

        # (1) Non-composable repeat: an earlier stage already produced a token this
        #     algorithm needs to be *absent* to be correct.
        for j in range(i):
            prev = plan[j]
            if prev.capabilities is None:
                continue
            clash = {
                t
                for t in caps.requires_absent & effective_produces(model, prev)
                if _token_overlap(model, stage, prev, t)
            }
            if clash:
                _report(
                    f"stage {i} ({stage}) cannot follow stage {j} ({prev}) on overlapping "
                    f"targets: {stage.algo!r} assumes {sorted(clash)} is not already set, but "
                    f"{prev.algo!r} produces it. Re-running it folds the scale a second time "
                    "while keeping only the last activation-side scale. Insert an explicit "
                    "unfold (disable_pre_quant_scale_and_resmooth) between them, or drop the "
                    "repeat.",
                    sink=sink,
                )

        # (2) Dead stage: every token it writes is overwritten downstream before anyone
        #     reads it, so the stage cannot affect the final model.
        produced = effective_produces(model, stage)
        if not produced:
            continue
        overwriters: dict[str, AlgoStage] = {}
        for token in produced:
            for j in range(i + 1, len(plan)):
                later = plan[j]
                if later.capabilities is None or not _token_overlap(model, stage, later, token):
                    continue
                if token in effective_requires(model, later):
                    break  # somebody read it -- not dead
                if token in effective_produces(model, later):
                    overwriters[token] = later
                    break
        if set(overwriters) == produced:
            first = next(iter(overwriters.values()))
            _report(
                f"stage {i} ({stage}) is dead: everything it produces ({sorted(produced)}) is "
                f"overwritten by a later stage ({first}) on the same quantizers, without being "
                "read in between. Remove it, or move it after the stage that overwrites it.",
                sink=sink,
            )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def compile_algo_cfg(
    config: QuantizeConfig | dict,
    model: nn.Module | None = None,
    strict: bool = True,
) -> CalibrationPlan:
    """Lower a quantize config into an ordered, validated list of scoped stages.

    Pure: reads model structure only, mutates nothing, runs no forward.

    Args:
        config: a :class:`QuantizeConfig` or a mapping with ``algo_cfg`` / ``algorithm``.
        model: the already-quantized model.  Required for the model-aware validation
            (scope resolution, roles, fused siblings, dependencies); when ``None`` only
            the config-only rules run.
        strict: raise :class:`AlgoCfgValidationError` on violations.  ``False`` downgrades
            them to warnings, which is what lets a knowingly-broken pipeline be run for
            demonstration purposes.

    Returns:
        The ordered plan.  Stages run in list order.
    """
    if isinstance(config, QuantizeConfig):
        entries, algorithm = config.algo_cfg or [], config.algorithm
    else:
        raw_entries = config.get("algo_cfg") or []
        entries = [e if isinstance(e, AlgoCfgEntry) else AlgoCfgEntry(**e) for e in raw_entries]
        algorithm = config.get("algorithm", "max")

    # An explicit algo_cfg suppresses the implicit whole-model default: entries are the
    # plan, and `algorithm` only fills in what they do not cover.  Keeping the "*" stage
    # unconditionally would silently re-calibrate every scoped target.
    if entries and algorithm is not None:
        covered = _coverage_is_total(model, entries) if model is not None else False
        if covered:
            algorithm = None

    plan = _lower(entries, algorithm)
    _validate_config_only(plan, strict)
    if model is not None:
        violations: list[str] = []
        _validate_scopes(model, plan, violations)
        _validate_fused_siblings(model, plan, violations)
        _validate_dependencies(model, plan, violations)
        if violations:
            body = "\n".join(f"  {i + 1}. {v}" for i, v in enumerate(violations))
            msg = f"invalid algo_cfg ({len(violations)} problem(s)):\n{body}"
            if strict:
                raise AlgoCfgValidationError(msg)
            warnings.warn(f"algo_cfg: {msg}", stacklevel=2)
    return plan


def _coverage_is_total(model: nn.Module, entries: list[AlgoCfgEntry]) -> bool:
    """Whether the entries already cover every quantizer, making `algorithm` redundant."""
    index = _index_model(model)
    covered: set[str] = set()
    for entry in entries:
        selector, scope = entry.selector
        _, quantizers = resolve_targets(model, scope, selector)
        covered |= quantizers
    return covered >= set(index.quantizers)


#: State tokens that are always available and therefore never need to be *produced* by a
#: stage: the weight tensor is part of the model, and activations come from the forward
#: loop rather than from another algorithm.
AMBIENT_TOKENS = frozenset({"weight", "acts"})


def derive_handoff(model: nn.Module, plan: CalibrationPlan, i: int) -> dict:
    """Extra kwargs for stage ``i`` implied by what earlier stages already produced.

    This is the general form of the hard-coded ``skip_max_init`` flag: when every non-ambient
    token a stage requires was already produced by an earlier stage on the same quantizers,
    the stage should refine that state rather than re-initialize it. Derived from the declared
    capabilities, not from a table of algorithm pairs.
    """
    stage = plan[i]
    if stage.capabilities is None:
        return {}
    needed = effective_requires(model, stage)
    if not needed:
        return {}

    satisfied = {
        token
        for token in needed
        for j in range(i)
        if plan[j].capabilities is not None
        and token in effective_produces(model, plan[j])
        and _token_overlap(model, stage, plan[j], token)
    }
    return {"skip_max_init": True} if needed <= satisfied else {}


def _stage_targets(model: nn.Module, stage: AlgoStage) -> set[str]:
    modules, quantizers = stage_targets(model, stage)
    return modules | quantizers


def plan_hash(plan: CalibrationPlan) -> str:
    """A stable hash of the plan.

    Distributed calibration is safe only if every rank runs the *same* stages over the
    *same* scopes — otherwise a predicate skips a quantizer on one rank and its amax
    all-reduce never matches, which hangs.  Comparing this hash across ranks turns that
    silent deadlock into a clear error.
    """
    import hashlib

    payload = "|".join(str(s.key()) for s in plan)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe_plan(plan: CalibrationPlan, model: nn.Module | None = None) -> str:
    """Human-readable plan dump, used by the demos and for debugging."""
    if not plan:
        return "  (empty plan — no calibration)"
    lines = []
    for i, stage in enumerate(plan):
        suffix = ""
        if model is not None:
            modules, quantizers = stage_targets(model, stage)
            suffix = f"  -> {len(modules)} module(s), {len(quantizers)} quantizer(s)"
        lines.append(f"  [{i}] {stage}{suffix}")
    return "\n".join(lines)
