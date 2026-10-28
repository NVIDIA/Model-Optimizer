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

import torch.nn as nn

from .config import AlgoCfgEntry, QuantizeAlgorithmConfig, QuantizeConfig

__all__ = [
    "AlgoCfgValidationError",
    "AlgoStage",
    "CalibrationPlan",
    "compile_algo_cfg",
    "derive_handoff",
    "stage_predicate",
]


class AlgoCfgValidationError(ValueError):
    """Raised when an ``algo_cfg`` cannot be lowered into a valid plan."""


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
    def descriptor(self):
        """The calibrate-mode descriptor class for this stage's algorithm, or ``None``."""
        return descriptor_for(self.algo)

    def __str__(self) -> str:
        extra = {k: v for k, v in self.cfg.items() if k != "method"}
        extra_s = f" {extra}" if extra else ""
        excl = f" minus {[g for _, g in self.exclude]}" if self.exclude else ""
        return (
            f"{self.algo or 'none'} @ {self.selector}={self.scope!r}{excl} (#{self.order}){extra_s}"
        )


CalibrationPlan = list[AlgoStage]


def descriptor_for(algo: str | None):
    """The registered descriptor *class* for ``algo``, or ``None`` if there is none."""
    if algo is None:
        return None
    # Imported lazily: `mode` imports this module while the package is still initializing.
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    descriptor = CalibrateModeRegistry.get(BaseCalibrateModeDescriptor._get_mode_name(algo))
    return None if descriptor is None else type(descriptor)


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


def stage_predicate(model: nn.Module, stage: AlgoStage) -> Callable[[nn.Module], bool]:
    """Build the ``should_process`` write-mask for a stage.

    Keyed on module identity, not name: ``layerwise_calibrate`` hands an algorithm a reparented
    decoder layer whose ``named_modules()`` names are subtree-relative. Admits both the parent
    modules and their quantizers, since the mask is applied to each in different call sites.
    """
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
    entries = list(entries)
    for entry in entries:
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
#
# Two rules, not the eight an earlier draft carried. Rejecting every combination that cannot be
# right needs a model of what each algorithm reads and writes, and that matrix has to be
# re-reasoned about for every new algorithm and fails *open* when it is wrong. The combinations
# we ship are curated recipes, which are validated by having been run. What survives here is
# what a recipe cannot catch for you: a glob that silently selects nothing, and an algorithm
# that would ignore the scope you gave it.


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
    """Reject an unregistered algorithm name. Needs no model."""
    for stage in plan:
        if stage.algo is not None and stage.descriptor is None:
            raise AlgoCfgValidationError(
                f"unknown algorithm {stage.algo!r}. Known algorithms: {known_algorithms()}"
            )


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
                f"algo_cfg: scope {stage.selector}={stage.scope!r} (stage {stage}) matches only "
                "disabled quantizers; the stage will do nothing.",
                stacklevel=2,
            )
            continue
        sink.append(
            f"scope {stage.selector}={stage.scope!r} (stage {stage}) matches no target "
            "in the model."
        )


def _selector_universe(model: nn.Module, selector: str) -> set[str]:
    """Every quantizer a glob of this selector kind could reach.

    Not simply "all quantizers": a ``module_name`` glob can only reach quantizers owned by a
    quantized linear, so KV ``*_bmm_quantizer`` -- which hang off the attention module -- are
    outside its reach. Measuring whole-model scope against all of them would reject
    ``module_name: "*"`` on any model with KV quantization enabled.
    """
    index = _index_model(model)
    if selector == "module_name":
        return {q for qs in index.quantizers_of.values() for q in qs}
    return set(index.quantizers)


def _reject_unscopable_with_scope(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """An algorithm that ignores the write-mask would write outside its scope."""
    for stage in plan:
        descriptor = stage.descriptor
        if descriptor is None or descriptor.scopable:
            continue
        _, in_scope = stage_targets(model, stage)
        if not in_scope:  # an empty scope is the other rule's business
            continue
        everything = _selector_universe(model, stage.selector)
        if in_scope != everything:
            sink.append(
                f"{stage.algo!r} does not honour the scoping write-mask, so it cannot be "
                f"restricted to {stage.selector}={stage.scope!r} ({len(in_scope)} of "
                f"{len(everything)} quantizers). Use it at whole-model scope."
            )


_MODEL_RULES = (_reject_empty_scope, _reject_unscopable_with_scope)


def _scoped_stages(plan: CalibrationPlan) -> list[AlgoStage]:
    """Stages that actually narrow the model; the auto-generated fallback does not."""
    return [s for s in plan if not (s.scope == "*" and s.selector == "quantizer_name")]


def _warn_if_splits_a_shared_amax_group(model: nn.Module, plan: CalibrationPlan) -> None:
    """A scope finer than a fused sibling group rewrites amax those siblings share.

    ``max_calibrate`` finishes with ``_finalize_with_shared_state``, which populates the
    ``global_amax`` that fusible siblings (``q/k/v_proj``, ``gate/up_proj``) are tied to, and
    with a TP sync -- both over the whole model, because they reconcile state that is shared
    *between* modules and so cannot honour a mask that splits the group. Scoping inside such a
    group is therefore unsupported: the stage rewrites scales its out-of-scope siblings depend
    on, and can trip the grouped-amax consistency check.
    """
    from .utils.shared_input import SHARED_PATTERNS, find_shared_input_groups

    name_of = {id(module): name for name, module in model.named_modules()}
    groups = [
        {name_of[id(m)] for m in members if id(m) in name_of}
        for _, members in find_shared_input_groups(model, SHARED_PATTERNS)
    ]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        return
    for stage in _scoped_stages(plan):
        modules, _ = stage_targets(model, stage)
        for members in groups:
            hit = members & modules
            if hit and hit != members:
                warnings.warn(
                    f"algo_cfg: scope {stage.selector}={stage.scope!r} covers {sorted(hit)} "
                    f"but not {sorted(members - hit)}, which share a fused weight amax group. "
                    "Calibration reconciles that group over the whole model, so this stage "
                    "will rewrite scales its out-of-scope siblings rely on. Widen the scope "
                    "to cover the whole group.",
                    stacklevel=2,
                )
                return


def _warn_if_expert_parallel(model: nn.Module, plan: CalibrationPlan) -> None:
    """Expert parallelism is not supported with a scoped plan; say so loudly.

    The plan is a pure function of ``(config, model structure)`` and therefore identical on
    every rank -- but only while every rank *sees* the same structure. Expert parallelism
    breaks that: each rank holds a different slice of the experts, so a glob can match on one
    rank and not another. Calibration's amax syncs are collectives, and a per-rank disagreement
    about which quantizers participate mismatches them, which NCCL hangs on rather than
    reporting.
    """
    from .model_calib import _has_expert_parallelism

    if not any(_has_expert_parallelism(m) for m in model.modules()):
        return
    scoped = _scoped_stages(plan)
    if not scoped:
        return
    warnings.warn(
        "algo_cfg: this model uses expert parallelism, which scoped calibration does not "
        "support in this release. Each rank holds a different slice of the experts, so a scope "
        f"glob (e.g. {scoped[0].selector}={scoped[0].scope!r}) can resolve differently per rank "
        "and mismatch the amax collectives. Run the plan at whole-model scope, or calibrate "
        "without expert parallelism.",
        stacklevel=2,
    )


# --- Entry point ---


def compile_algo_cfg(
    config: QuantizeConfig | dict,
    model: nn.Module | None = None,
    strict: bool = True,
) -> CalibrationPlan:
    """Lower a quantize config into an ordered, validated list of scoped stages.

    Pure: it reads the model's *structure* to resolve globs and validate, but mutates nothing,
    runs no forward and touches no data. So a bad config fails before any expensive calibration,
    and the plan is testable without running a model.
    """
    if isinstance(config, QuantizeConfig):
        entries, algorithm = config.algo_cfg or [], config.algorithm
    else:
        raw_entries = config.get("algo_cfg") or []
        entries = [e if isinstance(e, AlgoCfgEntry) else AlgoCfgEntry(**e) for e in raw_entries]
        algorithm = config.get("algorithm", "max")

    # An explicit algo_cfg suppresses the implicit whole-model default; `algorithm` only
    # fills in what entries do not cover.
    if entries and algorithm is not None and model is not None:
        if _coverage_is_total(model, entries):
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
        _warn_if_splits_a_shared_amax_group(model, plan)
        _warn_if_expert_parallel(model, plan)
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


def derive_handoff(model: nn.Module, plan: CalibrationPlan, i: int) -> dict:
    """Extra kwargs for stage ``i`` implied by what earlier stages already produced.

    One fact, not five: did an earlier stage establish amax on *every* target this one will
    touch? Coverage, not overlap -- a narrow producer before a wide consumer would leave some
    targets with no amax at all, and skipping init there would read uninitialized state.
    """
    stage = plan[i]
    descriptor = stage.descriptor
    if descriptor is None:
        return {}
    # Only algorithms that bootstrap their own amax can be told to skip it.
    if "skip_max_init" not in _calib_params(descriptor):
        return {}
    _, targets = stage_targets(model, stage)
    if not targets:
        return {}

    produced: set[str] = set()
    for j in range(i):
        earlier = plan[j].descriptor
        if earlier is not None and earlier.writes_amax:
            produced |= stage_targets(model, plan[j])[1]
    return {"skip_max_init": True} if targets <= produced else {}


def _calib_params(descriptor) -> frozenset[str]:
    """Parameter names of a descriptor's calibration function."""
    import inspect

    func = getattr(descriptor, "_calib_func", None)
    if func is None:
        return frozenset()
    return frozenset(inspect.signature(func).parameters)
