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
from contextlib import contextmanager
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch.nn as nn
from pydantic import ValidationError as PydanticValidationError

from .config import AlgoCfgEntry, QuantizeAlgorithmConfig, QuantizeConfig
from .conversion import quantizer_name_matches
from .utils import is_weight_quantizer

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

    # Any module that owns quantizers, not only quantized linears: fused wrappers own theirs
    # under per-projection names and are the module a scope must admit for them.
    owners: list[str] = field(default_factory=list)
    quantizers: list[str] = field(default_factory=list)
    quantizers_of: dict[str, list[str]] = field(default_factory=dict)  # owner -> quantizers
    parent_of: dict[str, str] = field(default_factory=dict)  # quantizer -> owner


#: Set only for the duration of one `compile_algo_cfg` call; see `_reuse_index`.
_INDEX_CACHE: dict[int, "_ModelIndex"] = {}


@contextmanager
def _reuse_index(model: nn.Module):
    """Build the structural index once for one compile, instead of once per rule.

    Safe because compiling is pure: it reads structure and mutates nothing, so the index cannot
    go stale inside one call. Scoped to that call rather than memoized globally, because between
    calls a stage *does* mutate the model -- `prepare()` rewrites block scales, algorithms add
    and remove quantizers -- and a cache that outlived a compile would hand back a model that no
    longer exists. Eight full `named_modules()` walks for a two-entry plan become one.
    """
    key = id(model)
    had = key in _INDEX_CACHE
    if not had:
        _INDEX_CACHE[key] = _build_index(model)
    try:
        yield
    finally:
        if not had:
            _INDEX_CACHE.pop(key, None)


def _index_model(model: nn.Module) -> _ModelIndex:
    """Structural index of the quantized model: owners, quantizers, ownership."""
    cached = _INDEX_CACHE.get(id(model))
    return cached if cached is not None else _build_index(model)


def _build_index(model: nn.Module) -> _ModelIndex:
    # Imported lazily: `mode` imports this module while `modelopt.torch.quantization` is
    # still initializing, and `.nn` pulls in the quantized-tensor backends.
    from .nn import QuantModule, TensorQuantizer
    from .nn.modules.tensor_quantizer import _QuantizerContainerBase

    index = _ModelIndex()
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer | _QuantizerContainerBase):
            # Asked of the members, not the container: `is_enabled` delegates to the first,
            # so [disabled, enabled] would read as disabled and hide its live sibling.
            if any(
                isinstance(leaf, TensorQuantizer) and leaf.is_enabled for leaf in module.modules()
            ):
                index.quantizers.append(name)
        elif isinstance(module, QuantModule):
            # Any QuantModule, not just `is_quantized_linear`: a fused wrapper (HF MoE
            # experts) names its quantizers per projection and fails that check, and left
            # unowned its module is excluded while its quantizers are admitted.
            index.owners.append(name)
            index.quantizers_of[name] = []
    for q in index.quantizers:
        # Nearest enclosing owner, not the direct parent: a SequentialQuantizer (W4A8,
        # INT4-AWQ) nests levels as `<linear>.weight_quantizer.0`, a grandchild. Depth 0 is
        # the root, whose name is "" when a bare Linear is quantized on its own.
        parts = q.split(".")
        for depth in range(len(parts) - 1, -1, -1):
            ancestor = ".".join(parts[:depth])
            if ancestor in index.quantizers_of:
                index.quantizers_of[ancestor].append(q)
                index.parent_of[q] = ancestor
                break
    # An owner whose quantizers are all disabled is likewise not a target.
    index.owners = [n for n in index.owners if index.quantizers_of[n]]
    index.quantizers_of = {n: qs for n, qs in index.quantizers_of.items() if qs}
    return index


def resolve_targets(model: nn.Module, scope: str, selector: str) -> tuple[set[str], set[str]]:
    """Resolve a scope into ``(module names, quantizer names)``."""
    index = _index_model(model)
    if selector == "module_name":
        modules = {n for n in index.owners if fnmatch.fnmatch(n, scope)}
        quantizers = {q for m in modules for q in index.quantizers_of[m]}
    else:
        # The canonical matcher, not a bare fnmatch: it also tries the normalized form, without
        # which a pattern written against `...weight_quantizer` matches none of a fused
        # wrapper's `..._weight_quantizers.0` and those experts quietly fall through to whatever
        # the fallback algorithm is.
        quantizers = {n for n in index.quantizers if quantizer_name_matches(scope, n)}
        # Expand a matched container to its leaves: `named_modules()` names a
        # SequentialQuantizer `...weight_quantizer` and its members `...weight_quantizer.0`,
        # so a glob ending in `weight_quantizer` matches only the container. The leaves hold
        # the state every calibration loop writes, so without this the stage calibrates none.
        quantizers |= {
            leaf
            for container in quantizers
            for leaf in index.quantizers
            if leaf.startswith(f"{container}.")
        }
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
# Only what a curated recipe cannot catch. Each rule here refuses a plan that would *write
# outside its scope*; a plan that merely does less work than asked is documented and warned.


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
    """Reject an unregistered algorithm name, or kwargs its config class refuses. Needs no model.

    The schema check otherwise happens in ``apply_mode``, i.e. per stage as it runs, so a typo in
    the *second* stage's kwargs would surface only after the first had calibrated and mutated the
    model. Compiling is meant to be the point where a bad config fails, so it happens here.
    """
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    for stage in plan:
        if stage.algo is None:
            continue
        if stage.descriptor is None:
            raise AlgoCfgValidationError(
                f"unknown algorithm {stage.algo!r}. Known algorithms: {known_algorithms()}"
            )
        # Through the registry instance: `config_class` is an instance property, not a classattr.
        mode_name = BaseCalibrateModeDescriptor._get_mode_name(stage.algo)
        config_class = CalibrateModeRegistry[mode_name].config_class
        try:
            config_class.model_validate(stage.cfg)
        except PydanticValidationError as e:
            raise AlgoCfgValidationError(f"{stage.algo!r}: {e}") from e


def _matches_only_disabled(model: nn.Module, stage: AlgoStage) -> bool:
    """The scope named quantizers that exist but are all disabled."""
    from .nn import SequentialQuantizer, TensorQuantizer

    return any(
        isinstance(module, TensorQuantizer | SequentialQuantizer)
        and not module.is_enabled
        and quantizer_name_matches(stage.scope, name)
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


def _shares_weight_global_amax(module: nn.Module) -> bool:
    """Whether this module's weight quantizer carries the shared NVFP4 ``global_amax``."""
    from .nn import TensorQuantizer

    quantizer = getattr(module, "weight_quantizer", None)
    if quantizer is None:
        return False
    leaves = [q for q in quantizer.modules() if isinstance(q, TensorQuantizer)]
    return any(getattr(q, "is_nvfp4_static", False) for q in leaves or [quantizer])


def _reject_split_local_expert_group(
    model: nn.Module, plan: CalibrationPlan, sink: list[str]
) -> None:
    """A scope may not take some of a layer's local experts and leave the rest.

    ``max_calibrate`` reconciles input amax across every expert in a layer through
    ``layer_sync_moe_local_experts_amax``, ungated because it is a group operation. So a stage
    scoped to one expert still rewrites its siblings' ranges, including experts this stage was
    told to leave alone. The reconciliation cannot honour a mask, so the plan is refused.
    """
    index = _index_model(model)
    owned = index.quantizers_of
    groups = []
    for name, module in model.named_modules():
        if not hasattr(module, "layer_sync_moe_local_experts_amax"):
            continue
        # Mirrors the guard inside that method: with a calibration ratio set, each expert is
        # calibrated independently and nothing is reconciled between them.
        if getattr(module, "_moe_calib_experts_ratio", None) is not None:
            continue
        # Both spellings: the HF wrapper syncs `self.experts`, Megatron's SequentialMLP syncs
        # `self.local_experts`. Missing one lets that framework's scopes through unchecked.
        for attr in ("experts", "local_experts"):
            if getattr(module, attr, None) is None:
                continue
            prefix = f"{name}.{attr}." if name else f"{attr}."
            # Quantizers, not owning modules: `sync_moe_expert_amax` reconciles input amax and
            # initializes any missing weight amax, so covering every expert *module* is not
            # enough -- a scope naming one role leaves the other written from outside itself.
            # Input and weight only: `sync_moe_expert_amax` reconciles those two roles and
            # leaves everything else -- an output quantizer, a KV bmm quantizer -- alone, so
            # demanding them too refuses a scope that covers everything actually written.
            members = {
                q
                for owner, qs in owned.items()
                if owner.startswith(prefix)
                for q in qs
                if is_weight_quantizer(q) or "input_quantizer" in q
            }
            if len({owner for owner in owned if owner.startswith(prefix)}) > 1:
                groups.append(members)
    if not groups:
        return
    for stage in _scoped_stages(plan):
        _, quantizers = stage_targets(model, stage)
        for members in groups:
            hit = members & quantizers
            if hit and hit != members:
                sink.append(
                    f"scope {stage.selector}={stage.scope!r} (stage {stage}) covers "
                    f"{sorted(hit)[:4]} but not {sorted(members - hit)[:4]}, which belong to the "
                    "local experts of one layer. Calibration reconciles input amax across the "
                    "layer and initializes any missing weight amax, so this stage would write "
                    "ranges outside itself. Cover every expert's quantizers, or none."
                )


def _holds_a_per_tensor_range(quantizer) -> bool:
    """Whether this quantizer's ``amax`` is the single scalar the MLA sync requires.

    That sync skips any role whose amax is not scalar, so this decides whether the pair shares
    anything at all. Measured: NVFP4 *dynamic* keeps a scalar amax despite having block sizes
    (the per-block scales are derived in-kernel), while NVFP4 *static* stores one amax per block
    and per-channel INT8 one per channel. Testing ``block_sizes is None`` therefore misses the
    dynamic case entirely. ``axis`` alone would do once set, but it is filled in by first-use
    normalization, which has not happened when a plan compiles -- so both are checked.
    """
    from .nn import TensorQuantizer

    if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
        return False
    if quantizer.axis is not None:
        return False
    block_sizes = quantizer.block_sizes
    return block_sizes is None or block_sizes.get("type") != "static"


def _mla_reconciled_roles(members: list[nn.Module]) -> list[str]:
    """The roles ``_sync_mla_fused_down_proj_amax`` reconciles for this pair.

    It gives the pair one *per-tensor* range per role, and skips a role whose amax is not
    scalar. A per-channel or blocked quantizer therefore has nothing reconciled across the
    pair, and splitting it is ordinary scoping rather than a shared-state write.
    """
    from .nn import SequentialQuantizer

    roles = []
    for role in ("input_quantizer", "weight_quantizer"):
        quantizers = []
        for member in members:
            quantizer = getattr(member, role, None)
            # W4A8 keeps its per-tensor FP8 range in the last level, as the sync itself does.
            if isinstance(quantizer, SequentialQuantizer):
                quantizer = quantizer[-1]
            quantizers.append(quantizer)
        if all(_holds_a_per_tensor_range(q) for q in quantizers):
            roles.append(role)
    return roles


# Populated below: _reject_split_shared_amax_group is defined after the helpers it needs.


def _scoped_stages(plan: CalibrationPlan) -> list[AlgoStage]:
    """Stages that actually narrow the model; the auto-generated fallback does not."""
    return [s for s in plan if not (s.scope == "*" and s.selector == "quantizer_name")]


def _reconciled_leaf(model: nn.Module, module_name: str, attr: str) -> str | None:
    """The quantizer name reconciliation actually writes for ``module_name``'s ``attr`` role.

    A sequential chain keeps its per-tensor range in the *last* level, and that is the leaf
    ``_sync_mla_fused_down_proj_amax`` reads and writes -- so naming the container here would
    make a scope that selects the level the sync touches look like it hit nothing.
    """
    from .nn import SequentialQuantizer

    quantizer = getattr(model.get_submodule(module_name), attr, None)
    if quantizer is None:
        return None
    if isinstance(quantizer, SequentialQuantizer):
        return f"{module_name}.{attr}.{len(quantizer) - 1}"
    return f"{module_name}.{attr}"


def _shared_amax_groups(model: nn.Module, shared_states) -> list[set[str]]:
    """Quantizer groups that calibration reconciles across modules, under ``shared_states``.

    Named per quantizer rather than per module: a group fuses one *role*, so a scope naming the
    other role of the same modules splits nothing. Resolved from the caller's configuration
    because the runtime does the same -- defaults would describe groups a configured stage
    never forms.
    """
    from .utils.shared_input import SharedWeightGlobalAmaxState, find_shared_input_groups

    name_of = {id(module): name for name, module in model.named_modules()}
    index = _index_model(model)
    live = set(index.owners)
    owned = {q for owner in live for q in index.quantizers_of.get(owner, ())}

    def role(members, attr):
        names = (name_of[id(m)] for m in members if id(m) in name_of)
        return {leaf for n in names if (leaf := _reconciled_leaf(model, n, attr))} & owned

    patterns = SharedWeightGlobalAmaxState.resolve_patterns(shared_states=shared_states)
    groups = [
        role(members, "weight_quantizer")
        for _, members in find_shared_input_groups(model, patterns)
        if any(_shares_weight_global_amax(m) for m in members)
    ]
    # Plus the MLA pair, found by attribute as `_sync_mla_fused_down_proj_amax` finds it. One
    # group spanning every role it reconciles, since it writes them on the same pass.
    for module in model.modules():
        members = [
            getattr(module, attr, None) for attr in ("linear_q_down_proj", "linear_kv_down_proj")
        ]
        if all(member is not None for member in members):
            reconciled = set()
            for attr in _mla_reconciled_roles(members):
                reconciled |= role(members, attr)
            if reconciled:
                groups.append(reconciled)
    groups = [g for g in groups if len(g) > 1]
    return [g for i, g in enumerate(groups) if g not in groups[:i]]


def _reject_split_shared_amax_group(
    model: nn.Module, plan: CalibrationPlan, sink: list[str]
) -> None:
    """A scope finer than a fused sibling group rewrites amax those siblings share.

    ``_finalize_with_shared_state`` reconciles state shared *between* modules, so it runs over
    the whole model and cannot honour a mask that splits a group. Groups are recomputed per
    stage: each stage reconciles under its own ``shared_states``, so one stage's configuration
    does not describe another's.
    """
    for stage in _scoped_stages(plan):
        _, quantizers = stage_targets(model, stage)
        for members in _shared_amax_groups(model, stage.cfg.get("shared_states")):
            hit = members & quantizers
            if hit and hit != members:
                sink.append(
                    f"scope {stage.selector}={stage.scope!r} covers {sorted(hit)} but not "
                    f"{sorted(members - hit)}, which share a fused amax group. "
                    "Calibration reconciles that group over the whole model, so this stage "
                    "would rewrite scales its out-of-scope siblings rely on. Widen the scope "
                    "to cover the whole group."
                )
                return


_MODEL_RULES = (
    _reject_empty_scope,
    _reject_unscopable_with_scope,
    _reject_split_shared_amax_group,
    _reject_split_local_expert_group,
)


def _warn_if_expert_parallel(model: nn.Module, plan: CalibrationPlan) -> None:
    """Expert parallelism is not supported with a scoped plan; say so loudly.

    The plan is identical on every rank only while every rank sees the same structure, which
    expert parallelism breaks -- and a per-rank disagreement in front of a collective hangs.
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
        with _reuse_index(model):
            _validate_against(model, plan, strict)
    return plan


def _validate_against(model: nn.Module, plan: CalibrationPlan, strict: bool) -> None:
    """Run every model-aware rule, then report all violations together.

    Every rule runs before anything is raised, so one compile tells you everything wrong with
    the plan rather than making you fix them one at a time.
    """
    violations: list[str] = []
    for rule in _MODEL_RULES:
        rule(model, plan, violations)
    if violations:
        body = "\n".join(f"  {i + 1}. {v}" for i, v in enumerate(violations))
        msg = f"invalid algo_cfg ({len(violations)} problem(s)):\n{body}"
        if strict:
            raise AlgoCfgValidationError(msg)
        warnings.warn(f"algo_cfg: {msg}", stacklevel=2)
    _warn_if_expert_parallel(model, plan)


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
        # Derived rather than declared: anything that calibrates at all establishes amax on what
        # it touched, and only the `algorithm=None` no-op has no calibration function. A flag
        # here would be true for every real algorithm and wrong-but-silent if one mis-set it.
        if earlier is not None and getattr(earlier, "_calib_func", None) is not None:
            produced |= stage_targets(model, plan[j])[1]
    return {"skip_max_init": True} if targets <= produced else {}


def _calib_params(descriptor) -> frozenset[str]:
    """Parameter names of a descriptor's calibration function."""
    import inspect

    func = getattr(descriptor, "_calib_func", None)
    if func is None:
        return frozenset()
    return frozenset(inspect.signature(func).parameters)
