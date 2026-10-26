# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared quantization state for fusible sibling modules.

Weight ``global_amax`` must be unified across modules that get **fused** at export
(q/k/v -> qkv, gate/up -> gate_up) so they quantize with one per-tensor scale.
:func:`find_shared_input_groups` discovers these groups by regex over module FQNs;
``SHARED_PATTERNS`` covers the standard q/k/v, gate/up and per-expert w1/w3 names.

Discovery is name/pattern-based (not input-hook-based) on purpose: "shares an input
tensor" is broader than "gets fused" — e.g. a ``shared_expert_gate`` reads the same
hidden states as the GLU pair but is never fused with it, so a hook would over-group
it. Patterns match exactly the roles export fuses.

The shared-state abstraction is intentionally stronger than a post-processing helper:
concrete states own the canonical tensor(s), tie member quantizers to the same buffer
object, and can install parent-level hooks for future runtime shared computation.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import torch
import torch.distributed as dist
import torch.nn as nn

from modelopt.torch.utils.distributed import ParallelState

from .core_utils import quantizer_attr_names, reduce_amax

__all__ = [
    "SHARED_PATTERNS",
    "SHARED_PATTERNS_ACROSS_EXPERTS",
    "SharedQuantState",
    "SharedWeightGlobalAmaxState",
    "find_shared_input_groups",
    "iter_shared_quant_states",
]

# Groups export fuses: q/k/v -> qkv, gate/up (incl. Mixtral w1/w3) -> gate_up.
# Regexes are ``re.fullmatch``-ed against module FQNs; ``(?:(.*)\.)?`` captures the
# immediate parent so MoE ``w1``/``w3`` groups are per-expert.
SHARED_PATTERNS = (
    r"(?:(.*)\.)?(?:q_proj|k_proj|v_proj)",
    r"(?:(.*)\.)?(?:gate_proj|up_proj)",
    r"(?:(.*)\.)?(?:w1|w3)",
)

# Future variant for states that intentionally share one group across all experts in the
# same ``experts``/``local_experts`` container. Kept unused until a state opts into it.
SHARED_PATTERNS_ACROSS_EXPERTS = (
    r"(?:(.*)\.)?(?:q_proj|k_proj|v_proj)",
    r"(?:(.*)\.)?(?:gate_proj|up_proj)",
    r"(?:(.*)\.)?(?:experts|local_experts)\.\d+\.(?:w1|w3)",
)


def _clone_tied_buffers_for_serialization(module, state_dict, prefix, local_metadata):
    """Clone a quantizer's tied managed buffers so the serialized tensors are independent.

    ``state_dict`` hook. Members alias one storage at runtime (single source of truth);
    ``torch.save`` dedups that, but safetensors (HF ``save_pretrained``) rejects shared storage.
    Restore re-ties.
    """
    for attr in module.__dict__.get("_shared_quant_tied_attrs", ()):
        key = prefix + attr
        tensor = state_dict.get(key)
        if tensor is not None:
            state_dict[key] = tensor.detach().clone()


class SharedQuantState(nn.Module, ABC):
    """Base class for shared quantization state owned by a group parent.

    Subclasses define when and how their canonical state is initialized. Runtime states
    can override :meth:`install_hooks` to compute/cache group-level values at the parent
    instead of doing the same work in every member.
    """

    name: ClassVar[str]
    managed_attrs: ClassVar[tuple[str, ...]] = ()
    target_quantizer_kind: ClassVar[str] = "weight"
    default_patterns: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        """Initialize an empty shared-state owner."""
        super().__init__()
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_members", ())
        object.__setattr__(self, "_hook_handles", [])

    def set_members(self, parent: nn.Module, members: Sequence[nn.Module]) -> None:
        """Set the owning parent and linked member modules."""
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_members", tuple(members))

    @property
    def members(self) -> tuple[nn.Module, ...]:
        """Return linked member modules."""
        return cast("tuple[nn.Module, ...]", self.__dict__["_members"])

    def install_hooks(self) -> None:
        """Install parent/member hooks for runtime shared computation."""

    def remove_hooks(self) -> None:
        """Remove hooks installed by :meth:`install_hooks`."""
        for handle in cast("list[Any]", self.__dict__["_hook_handles"]):
            handle.remove()
        object.__setattr__(self, "_hook_handles", [])

    @abstractmethod
    def sync(self, parallel_state: ParallelState | None = None) -> None:
        """Synchronize canonical state across distributed process groups."""

    def finalize(self) -> bool:
        """Whether the managed buffer(s) are populated; the finalize hook-produced states inherit.

        A state whose value is produced *during the forward* (e.g. the shared input-amax state's
        parent hook) needs only this readiness gate — the value already exists by now. States
        that produce on demand override it: weight aggregates member ``_amax``, SVDQuant runs an
        SVD. :meth:`populate` skips a state whose ``finalize`` returns ``False`` (uncalibrated /
        meta / forward never ran).
        """
        return any(
            (value := getattr(self, attr, None)) is not None and not value.is_meta
            for attr in self.managed_attrs
        )

    def _target_quantizer(self, member: nn.Module) -> nn.Module | None:
        attr = getattr(quantizer_attr_names(), f"{self.target_quantizer_kind}_quantizer")
        return getattr(member, attr, None)

    def _member_quantizers(self) -> list[nn.Module]:
        return [q for m in self.members if (q := self._target_quantizer(m)) is not None]

    def _set_state_buffer(self, state_attr: str, value: torch.Tensor | None) -> None:
        if state_attr not in self._buffers:
            self.register_buffer(state_attr, value, persistent=False)
        else:
            self._buffers[state_attr] = value
            self._non_persistent_buffers_set.add(state_attr)

    def tie_member_quantizer(self, quantizer: nn.Module) -> bool:
        """Alias a member quantizer's managed buffers to this state's canonical buffers.

        For each managed attr, point ``quantizer._buffers[attr]`` at the *same tensor object*
        as ``self.<attr>`` (register it if absent, else replace) so the member and the state
        share one storage, not a copy. Records the attr in the quantizer's
        ``_shared_quant_tied_attrs`` so ``TensorQuantizer.__setattr__`` rejects a later rebind.
        Returns whether anything was tied.
        """
        tied_any = False
        for attr in self.managed_attrs:
            value = getattr(self, attr, None)
            if value is None:
                continue
            if attr not in quantizer._buffers:
                quantizer.register_buffer(attr, value)
            else:
                quantizer._buffers[attr] = value
            tied = quantizer.__dict__.setdefault("_shared_quant_tied_attrs", set())
            tied.add(attr)
            tied_any = True
        # Serialized buffers must be independent: torch.save dedups shared storage, but
        # safetensors (HF save_pretrained) forbids it. The hook clones the tied buffers in the
        # output state_dict only; the runtime buffers stay aliased and restore re-ties.
        if (
            tied_any
            and _clone_tied_buffers_for_serialization not in quantizer._state_dict_hooks.values()
        ):
            quantizer._register_state_dict_hook(_clone_tied_buffers_for_serialization)
        return tied_any

    def tie_member_quantizers(self) -> None:
        """Tie all eligible member quantizers to the canonical state buffers."""
        for quantizer in self._member_quantizers():
            self.tie_member_quantizer(quantizer)

    def restore_from_members(self) -> bool:
        """Rebuild the canonical buffer from members' restored buffers and re-tie.

        Used only on checkpoint restore: the state is non-persistent, so it is absent until
        rebuilt here from the members' (persistent, just-loaded) buffers.
        """
        restored = False
        for attr in self.managed_attrs:
            for quantizer in self._member_quantizers():
                value = getattr(quantizer, attr, None)
                if value is None or value.is_meta:
                    continue
                self._set_state_buffer(attr, value)
                restored = True
                break
        if restored:
            self.tie_member_quantizers()
        return restored

    def _post_apply(self) -> None:
        """Per-state fixup after an ``_apply`` (dtype/device) move, before members are re-tied.

        No-op by default; override to do whatever the state needs (restore a dtype, re-slice a
        shared tensor, recompute, ...).
        """

    def _apply(self, fn, recurse=True):
        # ``_apply`` (.to/.cuda/.half/...) allocates fresh tensors and breaks the member
        # aliases; run the per-state fixup, then re-tie so members keep sharing one buffer.
        module = super()._apply(fn, recurse=recurse)
        self._post_apply()
        self.tie_member_quantizers()
        return module

    @classmethod
    def attach(cls, model: nn.Module, patterns: Sequence[str] | None = None) -> int:
        """Create this state on each discovered group's parent."""
        n_created = 0
        for parent, members in find_shared_input_groups(
            model,
            patterns=patterns,
            target_quantizer_kind=cls.target_quantizer_kind,
        ):
            state = cls()
            state.set_members(parent, members)
            created = _register_parent_shared_state(parent, state)
            n_created += int(created)
            if created:
                state.install_hooks()
        return n_created

    @classmethod
    def resolve_patterns(
        cls, shared_states: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    ) -> list[str]:
        """Resolve the max-calibration config into grouping patterns for this state."""
        if shared_states is not None:
            state_cfg = shared_states.get(cls.name, {})
            return list(state_cfg.get("patterns", cls.default_patterns))
        return list(cls.default_patterns)

    @classmethod
    @torch.no_grad()
    def populate(cls, model: nn.Module) -> int:
        """Finalize and sync every state of this type in ``model``; return the count populated."""
        n_groups = 0
        for state in iter_shared_quant_states(model, cls):
            if not state.finalize():
                continue
            state.sync(_first_parallel_state(state))
            state.remove_hooks()  # calibration done; drop any forward hooks (no-op if none)
            n_groups += 1
        return n_groups

    @classmethod
    def restore(cls, model: nn.Module, patterns: Sequence[str] | None = None) -> None:
        """Re-attach states and rebuild member aliases from members' restored buffers."""
        cls.attach(model, patterns=patterns)
        for state in iter_shared_quant_states(model, cls):
            state.restore_from_members()

    @classmethod
    def metadata(cls, model: nn.Module) -> dict[str, bool]:
        """Return restore metadata for this state when present in ``model``."""
        if any(iter_shared_quant_states(model, cls)):
            return {cls.name: True}
        return {}


class SharedWeightGlobalAmaxState(SharedQuantState):
    """Canonical shared weight ``global_amax`` for one fusible sibling group."""

    name: ClassVar[str] = "weight_global_amax"
    managed_attrs: ClassVar[tuple[str, ...]] = ("_global_amax",)
    target_quantizer_kind: ClassVar[str] = "weight"
    default_patterns: ClassVar[tuple[str, ...]] = SHARED_PATTERNS

    def __init__(self) -> None:
        """Initialize with an unset canonical ``global_amax`` buffer."""
        super().__init__()
        # Non-persistent canonical runtime buffer. It is serialized through the tied
        # member quantizer buffers (``*_weight_quantizer._global_amax``), so fresh-model
        # restore can rebuild the alias graph before ``load_state_dict``.
        self.register_buffer("_global_amax", None, persistent=False)

    @property
    def global_amax(self):
        """Return the canonical shared global amax."""
        return getattr(self, "_global_amax", None)

    @global_amax.setter
    def global_amax(self, value):
        self._set_state_buffer("_global_amax", value)

    def tie_member_quantizer(self, quantizer: nn.Module) -> bool:
        """Tie one member quantizer to the shared ``_global_amax`` buffer when eligible."""
        if not hasattr(quantizer, "global_amax") or self.global_amax is None:
            return False
        return super().tie_member_quantizer(quantizer)

    def finalize(self) -> bool:
        """Set ``global_amax`` to the max over members' calibrated ``_amax``."""
        child_maxes: list[torch.Tensor] = []
        for quantizer in self._member_quantizers():
            amax = getattr(quantizer, "_amax", None)
            if amax is None or amax.is_meta:
                continue
            child_maxes.append(reduce_amax(amax, axis=None))

        if not child_maxes:
            return False

        self.global_amax = torch.max(torch.stack(child_maxes)).clone().detach().to(torch.float32)
        return True

    def sync(self, parallel_state: ParallelState | None = None) -> None:
        """All-reduce (MAX) ``global_amax`` across EP, plus TP defensively."""
        if self.global_amax is None or parallel_state is None:
            return
        for group in (
            parallel_state.expert_model_parallel_group,
            parallel_state.tensor_parallel_group,
        ):
            if group is None or not group.is_initialized():
                continue
            try:
                dist.all_reduce(
                    self.global_amax,
                    op=dist.ReduceOp.MAX,
                    group=group.group,
                )
            except RuntimeError as e:
                raise RuntimeError("Failed to sync shared weight global_amax") from e

    def _post_apply(self) -> None:
        """Keep the NVFP4 scale in fp32 regardless of the model dtype after an ``_apply`` move."""
        if self.global_amax is not None:
            self.global_amax = self.global_amax.to(dtype=torch.float32)


def _has_enabled_quantizer(child: nn.Module, quantizer_attr: str) -> bool:
    # Membership is structural (independent of calibration), so this intentionally does NOT
    # require ``_amax`` — letting attach run before ``weight_only_quantize``.
    quantizer = getattr(child, quantizer_attr, None)
    return quantizer is not None and hasattr(quantizer, "_disabled") and not quantizer._disabled


def _build_parent_map(model: nn.Module) -> dict[nn.Module, nn.Module]:
    parent_map: dict[nn.Module, nn.Module] = {}
    for parent in model.modules():
        for child in parent.children():
            parent_map[child] = parent
    return parent_map


def _climb_past_modulelist(
    module: nn.Module,
    parent_map: dict[nn.Module, nn.Module],
    fallback: nn.Module,
) -> nn.Module:
    # Attaching shared states to a ModuleList registers them in the container's ``_modules``
    # and corrupts its iteration/length, so climb to the first non-ModuleList ancestor.
    # (Only ModuleList today; extend to ModuleDict etc. if needed.)
    cur = module
    while isinstance(cur, nn.ModuleList):
        parent = parent_map.get(cur)
        if parent is None or parent is cur:
            return fallback
        cur = parent
    return cur


def _lowest_common_ancestor(
    members: Sequence[nn.Module],
    parent_map: dict[nn.Module, nn.Module],
    fallback: nn.Module,
) -> nn.Module:
    if not members:
        return fallback

    def ancestors(m: nn.Module) -> list[nn.Module]:
        chain = []
        cur = m
        while cur in parent_map:
            cur = parent_map[cur]
            chain.append(cur)
        return chain

    chains = [ancestors(m) for m in members]
    if not chains[0]:
        return fallback
    common = set(chains[0])
    for c in chains[1:]:
        common &= set(c)
    # Deepest common ancestor: first in member[0]'s chain that's in every chain.
    for a in chains[0]:
        if a in common:
            return _climb_past_modulelist(a, parent_map, fallback)
    return fallback


def find_shared_input_groups(
    model: nn.Module,
    patterns: Sequence[str] | None = None,
    target_quantizer_kind: str = "weight",
) -> list[tuple[nn.Module, list[nn.Module]]]:
    r"""Find fusible sibling groups by regex over module FQNs; capture groups define the key.

    Each pattern is ``re.fullmatch``-ed against every quantized module's fully-qualified
    name; modules whose match yields the same capture-group tuple form one group, parented
    at their LCA. Granularity is set by *what you capture*:

    - Capture the immediate parent -> per-parent grouping: q/k/v per attention block, and
      **per-expert** ``w1``/``w3`` (each expert is the immediate parent), e.g.
      ``r"(.*)\.(?:w1|w3)$"``.
    - Capture only a level above the expert index, leaving the index uncaptured -> one
      **cross-expert** group, e.g. ``r"(.*)\.experts\.\d+\.(?:w1|w3)$"``.

    Roles to fuse together go in a non-capturing alternation ``(?:w1|w3)`` so they don't
    split the key; what you wrap in ``(...)`` is the group boundary. Pass
    ``SHARED_PATTERNS`` for the standard q/k/v + gate/up groups, or override via
    ``MaxCalibConfig.shared_states``. The caller selects which quantizer these groups
    apply to. Returns ``(parent, members)`` tuples; empty when no patterns are given.
    """
    if not patterns:
        return []
    quantizer_attr = getattr(quantizer_attr_names(), f"{target_quantizer_kind}_quantizer")
    compiled = [re.compile(p) for p in patterns]
    buckets: dict[tuple, list[nn.Module]] = {}
    order: list[tuple] = []
    for name, module in model.named_modules():
        if not _has_enabled_quantizer(module, quantizer_attr):
            continue
        for pattern_idx, regex in enumerate(compiled):
            match = regex.fullmatch(name)
            if match is not None:
                # include pattern_idx in case 2+ patterns yield the same capture tuple
                key = (pattern_idx, match.groups())
                if key not in buckets:
                    buckets[key] = []
                    order.append(key)
                buckets[key].append(module)
                break  # each module belongs to its first matching pattern
    parent_map = _build_parent_map(model)
    groups: list[tuple[nn.Module, list[nn.Module]]] = []
    for key in order:
        members = buckets[key]
        if len(members) >= 2:
            parent = _lowest_common_ancestor(members, parent_map, fallback=model)
            groups.append((parent, members))
    return groups


# ---------------------------------------------------------------------------
# Attach / populate lifecycle
# ---------------------------------------------------------------------------


def _members_key(members: Sequence[nn.Module]) -> tuple[int, ...]:
    return tuple(id(m) for m in members)


def _shared_state_list(parent: nn.Module) -> nn.ModuleList:
    states = getattr(parent, "_shared_quant_states", None)
    if states is None:
        states = nn.ModuleList()
        parent._shared_quant_states = states
    return states


def _register_parent_shared_state(
    parent: nn.Module,
    state: SharedQuantState,
) -> bool:
    """Register ``state`` on ``parent`` if an equivalent state is not already owned."""
    state_cls = type(state)
    wanted = _members_key(state.members)
    states = _shared_state_list(parent)
    for existing_state in states:
        if isinstance(existing_state, state_cls) and _members_key(existing_state.members) == wanted:
            return False

    states.append(state)
    return True


def iter_shared_quant_states(
    model: nn.Module,
    state_cls: type[SharedQuantState] = SharedQuantState,
):
    """Yield shared quant states owned within ``model``."""
    for module in model.modules():
        for state in getattr(module, "_shared_quant_states", ()):
            if isinstance(state, state_cls):
                yield state


def _first_parallel_state(state: SharedQuantState) -> ParallelState | None:
    for member in state.members:
        parallel_state = getattr(member, "parallel_state", None)
        if parallel_state is not None:
            return parallel_state
    return None
