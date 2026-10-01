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

"""Compile a quantize config into an ordered list of scoped calibration stages."""

import fnmatch
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn

from .config import AlgoCfgEntry, QuantizeAlgorithmConfig, QuantizeConfig

__all__ = [
    "WRITABLE_TOKENS",
    "AlgoCapabilities",
    "AlgoCfgValidationError",
    "AlgoStage",
    "CalibrationPlan",
    "capabilities_for",
    "compile_algo_cfg",
    "stage_predicate",
]


class AlgoCfgValidationError(ValueError):
    """Raised when an ``algo_cfg`` cannot be lowered into a valid plan."""


# --- Capabilities ---


@dataclass(frozen=True)
class AlgoCapabilities:
    """What one calibration algorithm reads, writes and assumes."""

    #: Writes *every* quantizer of each linear it touches, not one quantizer at a time.
    writes_whole_module: bool
    #: Role this algorithm *improves*. Narrower than what it writes: weight-side algorithms
    #: also seed input amax via an internal `max_calibrate`, which `may_write` records.
    refines: Literal["weight", "input", "both"]
    #: Tokens this algorithm reads. ``weight`` and ``acts`` are ambient, so never counted.
    #: Not a precondition: every algorithm listing ``weight_amax`` also seeds its own via an
    #: internal ``max_calibrate``, so a plan starting with one compiles. Declaring a token means
    #: two things -- an earlier stage producing it on *every* target lets this one skip that
    #: init (``derive_handoff``), and it keeps that earlier write from counting as dead.
    requires: frozenset[str] = frozenset()
    #: Tokens this may write. An *upper* bound: it may write fewer on a given model (smoothquant
    #: only touches INT8 layers), never more. Over-declaring is safe for conflict detection and
    #: unsafe for the hand-off, which is why the hand-off also checks coverage.
    may_write: frozenset[str] = frozenset()
    #: Tokens whose presence makes this algorithm incorrect: ``awq_lite`` folds a scale into
    #: the weight assuming an unsmoothed start, so it conflicts with ``pre_quant_scale``.
    invalid_if_present: frozenset[str] = frozenset()
    #: Can be restricted to a scope, i.e. threads the ``should_process`` write-mask through
    #: everything it writes. ``False`` forces whole-model scope; the compiler rejects the rest.
    scopable: bool = True
    #: NVFP4 weight block scales this algorithm needs: ``"static"`` (stored per-block amax it
    #: can search) or ``"dynamic"`` (derived in-kernel). ``None`` = works on either. Dynamic
    #: upgrades to static as a prep step; static never downgrades, since that discards a search.
    requires_weight_scales: Literal["static", "dynamic"] | None = None


WEIGHT_AMAX = "weight_amax"
INPUT_AMAX = "input_amax"
PRE_QUANT_SCALE = "pre_quant_scale"
WEIGHT = "weight"
ACTS = "acts"

#: Conservative default for an algorithm that declares nothing: over-reporting conflicts is
#: the safe direction.
WRITABLE_TOKENS = frozenset({WEIGHT, WEIGHT_AMAX, INPUT_AMAX, PRE_QUANT_SCALE})


def capabilities_for(algo: str | None, cfg: dict | None = None) -> AlgoCapabilities | None:
    """Capabilities of ``algo``, read off its calibrate-mode descriptor."""
    if algo is None:
        return None
    # Imported lazily: `mode` imports this module while the package is still initializing.
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    descriptor = CalibrateModeRegistry.get(BaseCalibrateModeDescriptor._get_mode_name(algo))
    if descriptor is None:
        return None
    return type(descriptor).capabilities_for_cfg(cfg or {})


# --- Stages ---


@dataclass(frozen=True)
class AlgoStage:
    """One algorithm applied to one scope — the unit of work in a calibration plan."""

    algo: str | None
    cfg: dict  # kwargs for the algorithm, including its "method" key
    scope: str  # the glob
    selector: str  # "module_name" | "quantizer_name"
    order: int  # position within its entry's pipeline
    # Scopes this stage must NOT touch. The fallback's coverage is a complement, so it cannot
    # be a glob; globs-minus-globs keeps the plan a pure function of the config (rank-identical).
    exclude: tuple[tuple[str, str], ...] = ()

    @property
    def capabilities(self) -> AlgoCapabilities | None:
        """Declared capabilities of this stage's algorithm, or ``None`` if undeclared."""
        return capabilities_for(self.algo, self.cfg)

    def __str__(self) -> str:
        extra = {k: v for k, v in self.cfg.items() if k != "method"}
        extra_s = f" {extra}" if extra else ""
        excl = f" minus {[g for _, g in self.exclude]}" if self.exclude else ""
        return (
            f"{self.algo or 'none'} @ {self.selector}={self.scope!r}{excl} (#{self.order}){extra_s}"
        )


CalibrationPlan = list[AlgoStage]


# --- Target resolution ---


@dataclass
class _ModelIndex:
    """Names of the things a scope can select, read once from the model structure."""

    linears: list[str] = field(default_factory=list)
    quantizers: list[str] = field(default_factory=list)
    quantizers_of: dict[str, list[str]] = field(default_factory=dict)  # linear -> quantizers
    parent_of: dict[str, str] = field(default_factory=dict)  # quantizer -> linear


def _index_model(model: nn.Module) -> _ModelIndex:
    """Structural index of the quantized model: linears, quantizers, ownership."""
    # Imported lazily: `mode` imports this module while `modelopt.torch.quantization` is
    # still initializing, and `.nn` pulls in the quantized-tensor backends.
    from .nn import SequentialQuantizer, TensorQuantizer
    from .utils import is_quantized_linear

    index = _ModelIndex()
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            index.linears.append(name)
            index.quantizers_of[name] = []
        # Disabled quantizers are not targets: nothing can be written to them, so a scope
        # reaching only disabled ones is as empty as one matching nothing at all.
        elif isinstance(module, TensorQuantizer | SequentialQuantizer) and module.is_enabled:
            index.quantizers.append(name)
    for q in index.quantizers:
        # Nearest enclosing linear, not the direct parent: a SequentialQuantizer (W4A8,
        # INT4-AWQ) nests levels as `<linear>.weight_quantizer.0`, a grandchild.
        parts = q.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            ancestor = ".".join(parts[:depth])
            if ancestor in index.quantizers_of:
                index.quantizers_of[ancestor].append(q)
                index.parent_of[q] = ancestor
                break
    # A linear whose quantizers are all disabled is likewise not a target.
    index.linears = [n for n in index.linears if index.quantizers_of[n]]
    index.quantizers_of = {n: qs for n, qs in index.quantizers_of.items() if qs}
    return index


def resolve_targets(model: nn.Module, scope: str, selector: str) -> tuple[set[str], set[str]]:
    """Resolve a scope into ``(module names, quantizer names)``."""
    index = _index_model(model)
    if selector == "module_name":
        modules = {n for n in index.linears if fnmatch.fnmatch(n, scope)}
        quantizers = {q for m in modules for q in index.quantizers_of[m]}
    else:
        quantizers = {n for n in index.quantizers if fnmatch.fnmatch(n, scope)}
        modules = {index.parent_of[q] for q in quantizers if q in index.parent_of}
    return modules, quantizers


#: Which quantizer a state token lives on.
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
        quantizers -= ex_quantizers
        if selector == "module_name":
            modules -= ex_modules
    if stage.exclude:
        owned = _index_model(model).quantizers_of
        modules = {m for m in modules if quantizers.intersection(owned.get(m, ()))}
    return modules, quantizers


def is_weight_quantizer(name: str) -> bool:
    """Whether ``name`` is a weight-side quantizer, by the convention the module keys on."""
    return "weight_quantizer" in name


def role_quantizers(model: nn.Module, stage: AlgoStage) -> dict[str, set[str]]:
    """The stage's in-scope quantizers, split by role."""
    _, quantizers = stage_targets(model, stage)
    weight = {q for q in quantizers if is_weight_quantizer(q)}
    return {"weight": weight, "input": quantizers - weight}


def effective_writes(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Tokens a stage actually writes here — declared ``may_write`` minus roles it cannot reach."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.may_write if by_role[TOKEN_ROLE.get(t, "weight")]}


def effective_requires(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Non-ambient tokens a stage reads here."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.requires - AMBIENT_TOKENS if by_role[TOKEN_ROLE.get(t, "weight")]}


def token_targets(model: nn.Module, stage: AlgoStage, token: str) -> set[str]:
    """The quantizers on which ``stage`` reads or writes ``token``."""
    return role_quantizers(model, stage)[TOKEN_ROLE.get(token, "weight")]


def _token_overlap(model: nn.Module, a: AlgoStage, b: AlgoStage, token: str) -> bool:
    return bool(token_targets(model, a, token) & token_targets(model, b, token))


def stage_predicate(model: nn.Module, stage: AlgoStage) -> Callable[[nn.Module], bool]:
    """Build the ``should_process`` write-mask for a stage."""
    modules, quantizers = stage_targets(model, stage)
    allowed = {id(model.get_submodule(name)) for name in modules | quantizers}
    return lambda module: id(module) in allowed


# --- Lowering ---


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
            plan.append(AlgoStage(name, cfg, scope, selector, order))

    # `algorithm` is the same thing at scope "*", as a fallback: it must not re-run over
    # targets an entry already claimed.
    if algorithm is not None:
        claimed = tuple(entry.selector for entry in entries)
        algos = algorithm if isinstance(algorithm, list) else [algorithm]
        for order, algo in enumerate(algos):
            name, cfg = _algo_to_name_and_cfg(algo)
            if name is None:
                continue
            plan.append(AlgoStage(name, cfg, "*", "quantizer_name", order, exclude=claimed))
    return plan


# --- Validation ---

#: Linears fused into one kernel at export: they share a weight scale, so also a pipeline.
FUSED_SIBLING_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("w1", "w3"),
)


def _report(msg: str, strict: bool = True, sink: list[str] | None = None) -> None:
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


def _validate_config_only(plan: CalibrationPlan) -> None:
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    for stage in plan:
        mode_name = BaseCalibrateModeDescriptor._get_mode_name(stage.algo)
        if mode_name not in CalibrateModeRegistry:
            raise AlgoCfgValidationError(
                f"unknown algorithm {stage.algo!r}. Known algorithms: {known_algorithms()}"
            )


def _scope_is_empty(model: nn.Module, stage: AlgoStage) -> bool:
    """An unmatched scope is reported by its own rule; the rest stay quiet about it."""
    modules, quantizers = resolve_targets(model, stage.scope, stage.selector)
    return not modules and not quantizers


def _matches_only_disabled(model: nn.Module, stage: AlgoStage) -> bool:
    """The scope named quantizers that exist but are all disabled."""
    from .nn import SequentialQuantizer, TensorQuantizer

    return any(
        isinstance(module, TensorQuantizer | SequentialQuantizer)
        and not module.is_enabled
        and fnmatch.fnmatch(name, stage.scope)
        for name, module in model.named_modules()
    )


def _reject_empty_scope(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """A glob that matches nothing is almost always a typo, and silently does no work."""
    for stage in plan:
        modules, quantizers = resolve_targets(model, stage.scope, stage.selector)
        if modules or quantizers:
            continue
        # Matching only *disabled* quantizers is different: an algo_cfg shared across
        # numerics is allowed to name a role that this one turns off. Say so and move on.
        if stage.selector == "quantizer_name" and _matches_only_disabled(model, stage):
            warnings.warn(
                f"scope {stage.selector}={stage.scope!r} (stage {stage}) matches only "
                "disabled quantizers; the stage will do nothing.",
                stacklevel=2,
            )
            continue
        _report(
            f"scope {stage.selector}={stage.scope!r} (stage {stage}) matches no target "
            "in the model.",
            sink=sink,
        )


def _reject_unscopable_with_scope(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """An algorithm that ignores the write-mask would write outside its scope."""
    for stage in plan:
        caps = stage.capabilities
        if caps is None or caps.scopable or _scope_is_empty(model, stage):
            continue
        everything = set(_index_model(model).quantizers)
        _, in_scope = stage_targets(model, stage)
        if in_scope != everything:
            _report(
                f"{stage.algo!r} does not honour the scoping write-mask, so it cannot be "
                f"restricted to {stage.selector}={stage.scope!r} ({len(in_scope)} of "
                f"{len(everything)} quantizers). Use it at whole-model scope.",
                sink=sink,
            )


def _reject_partial_module_scope(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """A whole-module algorithm needs a scope closed under module ownership.

    Otherwise it writes the quantizers left out anyway, and ``effective_writes`` understates
    what the stage touched, hiding real conflicts with later stages.
    """
    for stage in plan:
        caps = stage.capabilities
        if caps is None or not caps.writes_whole_module or _scope_is_empty(model, stage):
            continue
        modules, quantizers = stage_targets(model, stage)
        owned = _index_model(model).quantizers_of
        unreachable = {q for m in modules for q in owned.get(m, ()) if q not in quantizers}
        if unreachable:
            _report(
                f"{stage.algo!r} writes whole modules, but {stage.selector}="
                f"{stage.scope!r} leaves {len(unreachable)} of their quantizers out of "
                f"scope (e.g. {sorted(unreachable)[0]!r}). Select them with `module_name`.",
                sink=sink,
            )


def _reject_role_mismatch(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """A weight-refining algorithm aimed only at input quantizers refines nothing."""
    for stage in plan:
        caps = stage.capabilities
        if caps is None or stage.selector != "quantizer_name" or caps.refines == "both":
            continue
        _, quantizers = stage_targets(model, stage)
        roles = {"weight" if "weight_quantizer" in q else "input" for q in quantizers}
        if roles and caps.refines not in roles:
            _report(
                f"{stage.algo!r} only improves {caps.refines} quantizers but "
                f"{stage.selector}={stage.scope!r} matches only {sorted(roles)}: a no-op.",
                sink=sink,
            )


def _reject_wrong_weight_scales(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """A stage needing a dynamic weight grid runs after that grid has gone static.

    Walks the plan in order rather than checking the model once: a stage needing static
    upgrades its targets through ``prepare``, so what a later stage sees is the initial
    layout plus every upgrade before it.
    """
    static_now = {
        name
        for name, module in model.named_modules()
        if is_weight_quantizer(name) and getattr(module, "is_nvfp4_static", False)
    }
    upgraded_by: dict[str, AlgoStage] = {}

    for stage in plan:
        caps = stage.capabilities
        if caps is None:
            continue
        weights = role_quantizers(model, stage)["weight"]
        if caps.requires_weight_scales == "dynamic":
            clash = sorted(weights & static_now)
            if clash:
                culprit = upgraded_by.get(clash[0])
                cause = (
                    f"{culprit.algo!r} upgraded them"
                    if culprit is not None
                    else "they are `type: static` in quant_cfg"
                )
                _report(
                    f"{stage.algo!r} needs a dynamic NVFP4 weight grid but {len(clash)} "
                    f"target(s) are static (e.g. {clash[0]!r}) because {cause}, and static "
                    "never downgrades. Move it before the stage that needs static, or "
                    "declare `type: dynamic` in quant_cfg.",
                    sink=sink,
                )
        elif caps.requires_weight_scales == "static":
            for name in weights - static_now:
                upgraded_by[name] = stage
            static_now |= weights


def _reject_split_fused_siblings(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    pipeline_of: dict[str, tuple[str | None, ...]] = {}
    for stage in plan:
        modules, _ = stage_targets(model, stage)
        for m in modules:
            pipeline_of[m] = (*pipeline_of.get(m, ()), stage.algo)

    for group in FUSED_SIBLING_GROUPS:
        by_parent: dict[str, dict[str, tuple]] = {}
        for linear in _index_model(model).linears:
            leaf = linear.rsplit(".", 1)[-1]
            if leaf in group:
                by_parent.setdefault(linear.rsplit(".", 1)[0], {})[leaf] = pipeline_of.get(
                    linear, ()
                )
        for parent, members in by_parent.items():
            if len(set(members.values())) > 1:
                _report(
                    f"fusible siblings under {parent!r} got different pipelines "
                    f"({ {k: list(v) for k, v in members.items()} }): they export to one fused "
                    "kernel and must share a single pipeline.",
                    sink=sink,
                )


def _reject_noncomposable_repeat(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """An earlier stage wrote a token this algorithm needs absent to be correct."""
    for i, stage in enumerate(plan):
        caps = stage.capabilities
        if caps is None:
            continue
        for j in range(i):
            prev = plan[j]
            if prev.capabilities is None:
                continue
            clash = {
                t
                for t in caps.invalid_if_present & effective_writes(model, prev)
                if _token_overlap(model, stage, prev, t)
            }
            if clash:
                _report(
                    f"stage {i} ({stage}) cannot follow stage {j} ({prev}): {stage.algo!r} "
                    f"assumes {sorted(clash)} is unset but {prev.algo!r} writes it. Unfold "
                    "with disable_pre_quant_scale_and_resmooth between them, or drop the repeat.",
                    sink=sink,
                )


def _reject_dead_stage(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """Everything the stage writes is overwritten by a later stage before anyone reads it."""
    for i, stage in enumerate(plan):
        if stage.capabilities is None:
            continue
        produced = effective_writes(model, stage)
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
                if token in effective_writes(model, later):
                    overwriters[token] = later
                    break
        if set(overwriters) == produced:
            first = next(iter(overwriters.values()))
            _report(
                f"stage {i} ({stage}) is dead: everything it writes ({sorted(produced)}) is "
                f"overwritten unread by a later stage ({first}). Remove or reorder it.",
                sink=sink,
            )


#: Every rule the compiler enforces against the model structure, in the order they run.
#: Adding a rule is one function plus one entry here; this tuple is the list of what is checked.
_MODEL_RULES = (
    _reject_empty_scope,
    _reject_unscopable_with_scope,
    _reject_partial_module_scope,
    _reject_role_mismatch,
    _reject_wrong_weight_scales,
    _reject_split_fused_siblings,
    _reject_noncomposable_repeat,
    _reject_dead_stage,
)


# --- Entry point ---


def compile_algo_cfg(
    config: QuantizeConfig | dict,
    model: nn.Module | None = None,
    strict: bool = True,
) -> CalibrationPlan:
    """Lower a quantize config into an ordered, validated list of scoped stages."""
    if isinstance(config, QuantizeConfig):
        entries, algorithm = config.algo_cfg or [], config.algorithm
    else:
        raw_entries = config.get("algo_cfg") or []
        entries = [e if isinstance(e, AlgoCfgEntry) else AlgoCfgEntry(**e) for e in raw_entries]
        algorithm = config.get("algorithm", "max")

    # An explicit algo_cfg suppresses the implicit whole-model default; `algorithm` only
    # fills in what entries do not cover.
    if entries and algorithm is not None:
        covered = _coverage_is_total(model, entries) if model is not None else False
        if covered:
            algorithm = None

    plan = _lower(entries, algorithm)
    _validate_config_only(plan)
    if model is not None:
        violations: list[str] = []
        for rule in _MODEL_RULES:
            rule(model, plan, violations)
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


#: Tokens that never need producing: the weight is part of the model, acts come from the
#: forward loop.
AMBIENT_TOKENS = frozenset({WEIGHT, ACTS})


def derive_handoff(model: nn.Module, plan: CalibrationPlan, i: int) -> dict:
    """Extra kwargs for stage ``i`` implied by what earlier stages already produced."""
    stage = plan[i]
    if stage.capabilities is None:
        return {}
    needed = effective_requires(model, stage)
    if not needed:
        return {}

    # Coverage, not overlap: skipping init is only safe if *every* target already has the
    # state. A narrow producer before a wide consumer would leave some with no amax.
    for token in needed:
        produced_on: set[str] = set()
        for j in range(i):
            if plan[j].capabilities is None:
                continue
            if token in effective_writes(model, plan[j]):
                produced_on |= token_targets(model, plan[j], token)
        if not token_targets(model, stage, token) <= produced_on:
            return {}
    return {"skip_max_init": True}


def _stage_targets(model: nn.Module, stage: AlgoStage) -> set[str]:
    modules, quantizers = stage_targets(model, stage)
    return modules | quantizers
