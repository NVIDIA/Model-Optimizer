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

"""Per-group shared quantization state for fusible sibling modules.

Weight ``global_amax`` must be unified across modules that get **fused** at export
(q/k/v -> qkv, gate/up -> gate_up) so they quantize with one per-tensor scale.
:func:`find_shared_input_groups` discovers these groups by regex over module FQNs;
:data:`DEFAULT_WEIGHT_SHARED_PATTERNS` covers the standard q/k/v, gate/up and w1/w3
names, and callers may override per quantizer kind via ``MaxCalibConfig.shared_patterns``.

Discovery is name/pattern-based (not input-hook-based) on purpose: "shares an input
tensor" is broader than "gets fused" — e.g. a ``shared_expert_gate`` reads the same
hidden states as the GLU pair but is never fused with it, so a hook would over-group
it. Patterns match exactly the roles export fuses.

:class:`SharedQuantState` is an ``nn.Module`` attached to a group's parent so its
tensors ride along in ``state_dict``. Holds only ``weight_global_amax`` today;
designed to grow (act scales, LoRA factors, ...). The ``(parent, members)`` tuples
from :func:`find_shared_input_groups` are consumed by
:func:`attach_shared_quant_states` and :func:`populate_shared_state`.
"""

import re
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from modelopt.torch.utils.distributed import ParallelState

from .core_utils import quantizer_attr_names, reduce_amax

__all__ = [
    "DEFAULT_WEIGHT_SHARED_PATTERNS",
    "SharedQuantState",
    "attach_shared_quant_states",
    "find_shared_input_groups",
    "populate_shared_state",
]

# Default fusible-sibling patterns for WEIGHT global_amax — the groups export fuses:
# q/k/v -> qkv, gate/up (incl. Mixtral w1/w3) -> gate_up. These reproduce the legacy
# name-based grouping exactly. Regexes are ``re.fullmatch``-ed against module FQNs;
# ``(?:(.*)\.)?`` captures the immediate parent so grouping is per-parent (per-expert
# for MoE experts). Override per quantizer kind via ``MaxCalibConfig.shared_patterns``.
DEFAULT_WEIGHT_SHARED_PATTERNS = [
    r"(?:(.*)\.)?(?:q_proj|k_proj|v_proj)",
    r"(?:(.*)\.)?(?:gate_proj|up_proj)",
    r"(?:(.*)\.)?(?:w1|w3)",
]


# ---------------------------------------------------------------------------
# Per-group shared state
# ---------------------------------------------------------------------------


class SharedQuantState(nn.Module):
    """State shared across a sibling group of quantized modules.

    Attached to the group's parent (e.g. ``self_attn``, ``block_sparse_moe``) as a
    submodule, but its buffers are **non-persistent**: this is a calibration-time
    artifact, not part of the checkpoint. Members that quantize the same input resolve
    shared values here during calibration; the resolved value is then baked into each
    member's promoted quantizer (``NVFP4StaticQuantizer._global_amax``, which *is*
    serialized). So the scale survives save/restore via the members, and restore need
    not re-create this submodule (it isn't in ``state_dict``) — see
    :func:`attach_shared_quant_states`.

    Holds only ``weight_global_amax`` today, and it is mirrored onto every member. Any
    future field that is **not** mirrored on a member would not survive save/restore as
    a non-persistent buffer and would need its own restore path.
    """

    def __init__(self) -> None:
        """Initialize with an unset ``weight_global_amax`` and no registered members."""
        super().__init__()
        # NVFP4 two-level FP8 grid scale = max over members' per-block ``_amax``.
        # Non-persistent: a calibration-time artifact. The resolved value is baked into
        # each member's ``NVFP4StaticQuantizer._global_amax`` (which IS serialized), so
        # restore rebuilds scales from members and need not re-create this submodule.
        self.register_buffer("weight_global_amax", None, persistent=False)
        # Back-references to member modules. ``object.__setattr__`` keeps them out of
        # ``_modules`` so members' params don't re-enter our ``state_dict``.
        object.__setattr__(self, "_members", [])

    def sync_weight_global_amax(self, parallel_state: ParallelState | None) -> None:
        """All-reduce (MAX) ``weight_global_amax`` across EP, plus TP defensively.

        Weights are DP-replicated (no DP sync needed). EP sync is required since
        ranks hold different experts; TP sync guards against per-child ``_amax`` TP
        sync skipping block-quantized weights. Raises on a failed all-reduce: a silent
        failure would leave ranks with different scales and still promote/export them.
        """
        if self.weight_global_amax is None or parallel_state is None:
            return
        for group in (
            parallel_state.expert_model_parallel_group,
            parallel_state.tensor_parallel_group,
        ):
            if group is None or not group.is_initialized():
                continue
            try:
                dist.all_reduce(
                    self.weight_global_amax,
                    op=dist.ReduceOp.MAX,
                    group=group.group,
                )
            except RuntimeError as e:
                raise RuntimeError("Failed to sync shared weight_global_amax") from e


# ---------------------------------------------------------------------------
# Group discovery (regex over FQNs) → (parent, members) tuples
# ---------------------------------------------------------------------------


def _has_calibratable_weight_quantizer(child: nn.Module, wq_attr: str) -> bool:
    """A child is eligible if its weight quantizer is enabled and has ``_amax`` set."""
    wq = getattr(child, wq_attr, None)
    if wq is None or not hasattr(wq, "_disabled") or wq._disabled:
        return False
    return getattr(wq, "_amax", None) is not None


def _build_parent_map(model: nn.Module) -> dict[nn.Module, nn.Module]:
    """Build a ``{child_module: direct_parent}`` map by walking ``named_children``."""
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
    """Walk up past any ``nn.ModuleList`` ancestors to a regular module.

    Attaching ``SharedQuantState`` to a ``ModuleList`` registers it in that
    container's ``_modules`` and corrupts its iteration/length (the state shows up
    alongside the experts), so attach to the first non-ModuleList ancestor.

    Only ``nn.ModuleList`` is handled today. It can be extended in the future to include modules
    like nn.ModuleDict``.
    """
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
    """LCA of ``members`` in the module tree (``fallback`` if none).

    Climbs past ``nn.ModuleList`` ancestors so the result can host
    ``SharedQuantState`` as a submodule.
    """
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
    :data:`DEFAULT_WEIGHT_SHARED_PATTERNS` for the standard q/k/v + gate/up groups, or
    override via ``MaxCalibConfig.shared_patterns``. The caller selects which quantizer
    these groups apply to (today only the weight quantizer). Returns ``(parent, members)``
    tuples; empty when no patterns are given.
    """
    if not patterns:
        return []
    wq_attr = quantizer_attr_names("weight").weight_quantizer
    compiled = [re.compile(p) for p in patterns]
    buckets: dict[tuple, list[nn.Module]] = {}
    order: list[tuple] = []
    for name, module in model.named_modules():
        if not _has_calibratable_weight_quantizer(module, wq_attr):
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


def attach_shared_quant_states(
    model: nn.Module,
    patterns: Sequence[str] | None = None,
) -> int:
    """Create ``SharedQuantState`` on each group's parent and link members.

    Groups are discovered by ``patterns`` (regexes over module FQNs; see
    :func:`find_shared_input_groups`). The parent owns the state under
    ``_shared_quant_state`` (normal setattr → a registered submodule, so its buffer
    rides along in ``state_dict``). Each member's weight quantizer — the only consumer,
    via ``promote_nvfp4_static_quantizers`` — gets a back-reference under the distinct
    name ``_shared_quant_state_ref`` set with ``object.__setattr__`` (not a submodule,
    so the buffer isn't duplicated per member). The distinct names let
    ``populate_shared_state`` select owners with a plain ``getattr``.

    Idempotent (reuses an existing parent state). Returns the number created.
    """
    n_created = 0
    wq_attr = quantizer_attr_names("weight").weight_quantizer
    for parent, members in find_shared_input_groups(model, patterns=patterns):
        if not hasattr(parent, "_shared_quant_state"):
            parent._shared_quant_state = SharedQuantState()
            n_created += 1
        state = parent._shared_quant_state
        # Record members so populate_shared_state needn't re-run discovery.
        object.__setattr__(state, "_members", list(members))
        for child in members:
            wq = getattr(child, wq_attr, None)
            if wq is None:
                continue
            # Groups are disjoint after merging, so each quantizer gets one state per
            # call and a re-attach reuses the same object; a different existing state
            # would mean an inconsistent re-attach.
            existing = getattr(wq, "_shared_quant_state_ref", None)
            assert existing is None or existing is state, (
                f"{type(wq).__name__} already belongs to a different shared-input "
                "group; groups should be disjoint after merging."
            )
            object.__setattr__(wq, "_shared_quant_state_ref", state)
    return n_created


@torch.no_grad()
def populate_shared_state(model: nn.Module) -> int:
    """Aggregate per-member stats into each group's ``SharedQuantState``.

    Currently sets ``weight_global_amax`` = max over members' reduced ``_amax``,
    EP-synced so all ranks agree, then writes it back to each member's
    ``global_amax`` (overriding any stale value from an earlier promotion). Future
    fields plug in here as extra aggregation steps.

    Call after members' ``_amax`` is cross-rank consistent (post TP/DP/EP sync in
    ``max_calibrate``). Members not yet promoted to ``NVFP4StaticQuantizer`` are
    skipped on write-back; the next promotion reads the shared value instead.
    Returns the number of groups populated.
    """
    from modelopt.torch.quantization.nn import NVFP4StaticQuantizer

    wq_attr = quantizer_attr_names("weight").weight_quantizer
    n_groups = 0

    for parent in model.modules():
        # Owners hold the state under ``_shared_quant_state`` (members use the
        # distinct ``_shared_quant_state_ref``), so getattr matches owners only.
        state = getattr(parent, "_shared_quant_state", None)
        if not isinstance(state, SharedQuantState):
            continue

        members = getattr(state, "_members", [])
        if not members:
            continue

        child_maxes: list[torch.Tensor] = []
        parallel_state: ParallelState | None = None
        for child in members:
            wq = getattr(child, wq_attr, None)
            if wq is None or getattr(wq, "_amax", None) is None:
                continue
            child_maxes.append(reduce_amax(wq._amax, axis=None))
            if parallel_state is None:
                parallel_state = getattr(child, "parallel_state", None)

        if not child_maxes:
            continue

        local_max = torch.max(torch.stack(child_maxes))
        state.weight_global_amax = local_max
        state.sync_weight_global_amax(parallel_state)

        synced = state.weight_global_amax
        for child in members:
            wq = getattr(child, wq_attr, None)
            if isinstance(wq, NVFP4StaticQuantizer):
                wq.global_amax = synced
        n_groups += 1

    return n_groups
