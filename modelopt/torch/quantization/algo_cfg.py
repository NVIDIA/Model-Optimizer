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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch.nn as nn

from .config import AlgoCfgEntry, QuantizeAlgorithmConfig, QuantizeConfig

__all__ = [
    "ALL_WRITABLE_TOKENS",
    "AlgoCapabilities",
    "AlgoCfgValidationError",
    "AlgoStage",
    "CalibrationPlan",
    "capabilities_for",
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
    """What one calibration algorithm reads, writes and assumes.

    Every field here is a **claim about the implementation** that some compile rule trusts, so a
    wrong entry is a silent bug rather than a loud one. ``demos/05_conformance.py`` checks the
    claims against what the algorithms actually do.

    ``produces`` / ``consumes`` are a small open vocabulary of state tokens: ``weight``,
    ``weight_amax``, ``input_amax``, ``pre_quant_scale``, and ``acts`` (activations from the
    forward loop). Which quantizer a token lives on is fixed by :data:`TOKEN_ROLE`, so the
    declarations do not repeat it.
    """

    #: ``True`` when the algorithm writes *every* quantizer of each linear it touches, rather
    #: than one quantizer at a time. The scope-closure rule enforces exactly this.
    writes_whole_module: bool
    #: The quantizer role this algorithm actually *improves* -- ``"weight"``, ``"input"`` or
    #: ``"both"``. Aiming a weight-refining algorithm at input quantizers is a no-op worth an
    #: error. Deliberately narrower than what the algorithm *writes*: most weight-side algorithms
    #: also seed input amax through an internal `max_calibrate`, which `produces` records.
    optimizes: str
    #: Tokens read. ``weight`` and ``acts`` are always available (see
    #: :data:`ALWAYS_AVAILABLE`).
    consumes: frozenset[str] = frozenset()
    #: Tokens written. An **upper bound**: an algorithm may legitimately write less on a given
    #: model (smoothquant only touches INT8 layers), but must never write more.
    produces: frozenset[str] = frozenset()
    #: Tokens whose presence makes this algorithm incorrect: it must not run after anything
    #: that produced one. ``awq_lite`` folds a scale into the weight assuming an unsmoothed
    #: starting point, so it conflicts with ``pre_quant_scale``.
    conflicts_with: frozenset[str] = frozenset()
    #: Whether the calibration function honours the ``should_process`` write-mask. One that does
    #: not must never be scoped: it would either raise or silently write outside its scope.
    honors_write_mask: bool = True


WEIGHT_AMAX = "weight_amax"
INPUT_AMAX = "input_amax"
PRE_QUANT_SCALE = "pre_quant_scale"
WEIGHT = "weight"

#: Every token an algorithm can write. Used as the conservative default for an algorithm that
#: declares nothing -- assuming it writes everything over-reports conflicts, which is the safe
#: direction; assuming it writes nothing would hide them.
ALL_WRITABLE_TOKENS = frozenset({WEIGHT, WEIGHT_AMAX, INPUT_AMAX, PRE_QUANT_SCALE})


def capabilities_for(algo: str | None, cfg: dict | None = None) -> AlgoCapabilities | None:
    """Capabilities of ``algo``, read off its calibrate-mode descriptor.

    The descriptor is already the one-object-per-algorithm registry, and already carries
    implementation traits of this kind (``_supports_layerwise``). Keeping capabilities there
    rather than in a second table means a user-registered custom algorithm inherits the
    conservative default instead of falling off the end of a dict -- which used to disable
    every validation rule for it, silently.

    ``cfg`` is the algorithm's own kwargs from the plan. A few algorithms' capabilities depend
    on them: ``lsq`` delegates to a configurable sub-algorithm, so whether it writes ``weight``
    is a property of ``scale_algorithm``, not of ``lsq``. Note this derives capabilities *from*
    the user's parameters -- it is not a way to override them. The declarations are statements
    about the implementation, and letting a caller assert "this does not write weights" would
    switch off the analysis that exists to catch exactly that mistake.
    """
    if algo is None:
        return None
    # Imported lazily: `mode` imports this module while the package is still initializing.
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    descriptor = CalibrateModeRegistry.get(BaseCalibrateModeDescriptor._get_mode_name(algo))
    if descriptor is None:
        return None
    return type(descriptor).capabilities_for_cfg(cfg or {})


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
        return capabilities_for(self.algo, self.cfg)

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


#: Set only for the duration of :func:`_reusing_model_index`. The index is a pure function of
#: the model's structure, so caching it is safe exactly as long as nothing mutates the module
#: tree -- which is why the cache is scoped to a single pure computation rather than global.
_CACHED_INDEX: tuple[nn.Module, "_ModelIndex"] | None = None


@contextmanager
def _reusing_model_index(model: nn.Module) -> Iterator[None]:
    """Build the model index once for a block of structural queries.

    Validation compares every stage against every other stage, and each comparison used to
    rebuild the index by walking ``named_modules()`` from scratch -- quadratic in the number of
    stages, which defeats the point of compile being the cheap step.
    """
    global _CACHED_INDEX
    previous = _CACHED_INDEX
    _CACHED_INDEX = (model, _build_model_index(model))
    try:
        yield
    finally:
        _CACHED_INDEX = previous


def _index_model(model: nn.Module) -> _ModelIndex:
    """Cached structural index; see :func:`_reusing_model_index`."""
    if _CACHED_INDEX is not None and _CACHED_INDEX[0] is model:
        return _CACHED_INDEX[1]
    return _build_model_index(model)


def _build_model_index(model: nn.Module) -> _ModelIndex:
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
        # Walk up to the nearest enclosing quantized linear rather than assuming the
        # quantizer is its direct child: a SequentialQuantizer (W4A8, INT4-AWQ) nests its
        # levels as `<linear>.weight_quantizer.0`, which is a *grandchild*. Attaching only
        # direct children silently drops every sub-quantizer from module-scoped stages.
        parts = q.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            ancestor = ".".join(parts[:depth])
            if ancestor in index.quantizers_of:
                index.quantizers_of[ancestor].append(q)
                index.parent_of[q] = ancestor
                break
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
    """``(modules, quantizers)`` a stage may write, after subtracting its exclusions.

    Exclusions are subtracted at the granularity the excluding entry actually claimed. A
    ``quantizer_name`` entry claims *quantizers*, not the linears that own them, so removing
    its parent modules as well would strip the fallback stage of every module -- and with it
    the weight calibration that only runs per module (`weight_only_quantize`). A module drops
    out only when nothing it owns is left in scope.
    """
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


def role_quantizers(model: nn.Module, stage: AlgoStage) -> dict[str, set[str]]:
    """The stage's in-scope quantizers, split by role.

    Which role a given token lands on is already fixed by :data:`TOKEN_ROLE`, so this does not
    consult the algorithm: `weight_amax` goes to weight quantizers whoever writes it. Gating
    this by a declared role used to hide the fact that most weight-side algorithms also seed
    `input_amax` through an internal `max_calibrate`.
    """
    _, quantizers = stage_targets(model, stage)
    weight = {q for q in quantizers if "weight_quantizer" in q}
    return {"weight": weight, "input": quantizers - weight}


def effective_produces(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Tokens a stage actually writes here — declared ``produces`` minus roles it cannot reach."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.produces if by_role[TOKEN_ROLE.get(t, "weight")]}


def effective_consumes(model: nn.Module, stage: AlgoStage) -> set[str]:
    """Non-ambient tokens a stage reads here."""
    caps = stage.capabilities
    if caps is None:
        return set()
    by_role = role_quantizers(model, stage)
    return {t for t in caps.consumes - ALWAYS_AVAILABLE if by_role[TOKEN_ROLE.get(t, "weight")]}


def token_targets(model: nn.Module, stage: AlgoStage, token: str) -> set[str]:
    """The quantizers on which ``stage`` reads or writes ``token``."""
    return role_quantizers(model, stage)[TOKEN_ROLE.get(token, "weight")]


def _token_overlap(model: nn.Module, a: AlgoStage, b: AlgoStage, token: str) -> bool:
    """Whether two stages touch ``token`` on *any* of the same quantizers.

    This is the relation the conflict rules need -- one shared quantizer is enough for a stage
    to clobber another. It is deliberately **not** the relation the handoff needs; see
    :func:`derive_handoff`.
    """
    return bool(token_targets(model, a, token) & token_targets(model, b, token))


def stage_predicate(model: nn.Module, stage: AlgoStage) -> Callable[[nn.Module], bool]:
    """Build the ``should_process`` write-mask for a stage.

    The predicate is AND-ed into each algorithm's existing ``is_enabled`` filter, so a stage
    **writes only its targets and never toggles enable-state** — reads (and hence the
    activations seen by search-based algorithms like AWQ and GPTQ) are unchanged.

    It matches on **module identity, not name**. Names are relative to whatever root they were
    produced from, so a name-based mask silently matches nothing when an algorithm is invoked on
    a subtree: ``layerwise_calibrate`` calls each algorithm with a single decoder layer, whose
    ``named_modules()`` yields ``mlp.gate_proj`` where the plan resolved
    ``layers.0.mlp.gate_proj``. Identity is invariant under that reparenting.
    """
    modules, quantizers = stage_targets(model, stage)
    allowed = {id(model.get_submodule(name)) for name in modules | quantizers}
    return lambda module: id(module) in allowed


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


def _validate_config_only(plan: CalibrationPlan) -> None:
    """Checks that need no model, and that no ``strict=False`` can downgrade.

    An unknown algorithm is not a judgement call: the executor has nothing to dispatch to, so
    warning and continuing would only move the failure to a less informative place.
    """
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    for stage in plan:
        mode_name = BaseCalibrateModeDescriptor._get_mode_name(stage.algo)
        if mode_name not in CalibrateModeRegistry:
            raise AlgoCfgValidationError(
                f"unknown algorithm {stage.algo!r}. Known algorithms: {known_algorithms()}"
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

        if not caps.honors_write_mask:
            everything = set(_index_model(model).quantizers)
            _, in_scope = stage_targets(model, stage)
            if in_scope != everything:
                _report(
                    f"{stage.algo!r} does not honour the scoping write-mask, so it cannot be "
                    f"restricted to {stage.selector}={stage.scope!r} ({len(in_scope)} of "
                    f"{len(everything)} quantizers) -- it would write outside its scope and "
                    "clobber other stages. Use it at whole-model scope, or add "
                    "`should_process` support to its calibration function first.",
                    sink=sink,
                )
                continue
        # An algorithm that writes whole modules touches *every* quantizer of those linears:
        # `awq_lite` sets a pre_quant_scale on the input quantizer as well as the weight amax.
        # Its scope therefore has to be closed under module ownership. A `quantizer_name` scope
        # that selects only some of a module's quantizers cannot constrain such an algorithm --
        # the predicate admits the parent module, the algorithm writes both roles, and the write
        # -mask is silently violated. Worse, `effective_produces` is derived from the declared
        # role sets, so the compiler would *understate* what the stage writes and miss real
        # conflicts with later stages.
        if caps.writes_whole_module:
            modules, quantizers = stage_targets(model, stage)
            owned = _index_model(model).quantizers_of
            unreachable = {q for m in modules for q in owned.get(m, ()) if q not in quantizers}
            if unreachable:
                _report(
                    f"{stage.algo!r} writes whole modules: it touches every quantizer of "
                    f"the modules it touches, but {stage.selector}={stage.scope!r} leaves "
                    f"{len(unreachable)} of them out of scope (e.g. "
                    f"{sorted(unreachable)[0]!r}). It would write them anyway, outside the "
                    "write-mask. Select the modules instead, with `module_name`.",
                    sink=sink,
                )
                continue

        # Role check: a weight-only algorithm pointed at input quantizers writes nothing.
        if stage.selector == "quantizer_name":
            roles = {"weight" if "weight_quantizer" in q else "input" for q in quantizers}
            if caps.optimizes != "both" and roles and caps.optimizes not in roles:
                _report(
                    f"{stage.algo!r} only improves {caps.optimizes} quantizers but "
                    f"{stage.selector}={stage.scope!r} matches only {sorted(roles)} quantizers "
                    "— the stage would be a no-op.",
                    sink=sink,
                )


def _validate_fused_siblings(model: nn.Module, plan: CalibrationPlan, sink: list[str]) -> None:
    """Fusible siblings must share one pipeline: one fused kernel, one weight scale.

    Every stage counts, including the fallback the top-level ``algorithm`` lowers to, and a
    sibling matched by *nothing* counts as the empty pipeline. Skipping either -- as this check
    originally did for the fallback -- hides the most likely way to get the bug: name one sibling
    in an entry and let the others fall through to a different default.
    """
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
                    f"({ {k: list(v) for k, v in members.items()} }). They export to one fused "
                    "kernel and must share a single weight scale, so they must share one "
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
                for t in caps.conflicts_with & effective_produces(model, prev)
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
                if token in effective_consumes(model, later):
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
    _validate_config_only(plan)
    if model is not None:
        violations: list[str] = []
        with _reusing_model_index(model):
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
ALWAYS_AVAILABLE = frozenset({"weight", "acts"})


def derive_handoff(model: nn.Module, plan: CalibrationPlan, i: int) -> dict:
    """Extra kwargs for stage ``i`` implied by what earlier stages already produced.

    This is the general form of the hard-coded ``skip_max_init`` flag: when every non-ambient
    token a stage consumes was already produced by an earlier stage on the same quantizers,
    the stage should refine that state rather than re-initialize it. Derived from the declared
    capabilities, not from a table of algorithm pairs.
    """
    stage = plan[i]
    if stage.capabilities is None:
        return {}
    with _reusing_model_index(model):
        needed = effective_consumes(model, stage)
        if not needed:
            return {}

        # Coverage, not overlap. Telling a stage to skip its own initialization is only safe if
        # every target it will touch already has the state -- a narrow producer in front of a
        # wide consumer leaves the uncovered targets with no amax at all, silently. Conflict
        # detection asks the opposite question ("do these share *any* target?"), which is why
        # the two use different set relations over the same per-token target sets.
        for token in needed:
            produced_on: set[str] = set()
            for j in range(i):
                if plan[j].capabilities is None:
                    continue
                if token in effective_produces(model, plan[j]):
                    produced_on |= token_targets(model, plan[j], token)
            if not token_targets(model, stage, token) <= produced_on:
                return {}
    return {"skip_max_init": True}


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
