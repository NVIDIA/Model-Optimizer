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

"""Discovery and per-group shared state for modules that consume the same input.

- :func:`collect_shared_input_modules` runs a probe forward with hooks and
  returns the modules that received the same input tensor. Used by both
  calibration (here) and export (``unified_export_hf``).
- :class:`SharedQuantState` is an ``nn.Module`` attached to a sibling group's
  parent so its tensors ride along in ``state_dict``. Holds only
  ``weight_global_amax`` today; designed to grow (act scales, LoRA factors, ...).

:func:`find_shared_input_groups` (hooks + name patterns) produces the
``(parent, members)`` tuples consumed by :func:`attach_shared_quant_states` and
:func:`populate_shared_state`.
"""

import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from modelopt.torch.utils.distributed import ParallelState

from .core_utils import is_quantized_linear, quantizer_attr_names, reduce_amax

ForwardLoop = Callable[[nn.Module], None]

__all__ = [
    "SharedQuantState",
    "attach_shared_quant_states",
    "collect_shared_input_modules",
    "find_shared_input_groups",
    "populate_shared_state",
]


# ---------------------------------------------------------------------------
# Discovery primitive
# ---------------------------------------------------------------------------


def collect_shared_input_modules(
    model: nn.Module,
    forward_fn: Callable[[], None],
    module_filter: Callable[[nn.Module], bool] | None = None,
    output_filter: Callable[[nn.Module], bool] | None = None,
) -> tuple[dict, dict | None]:
    """Hook the model, run a probe forward, group modules by shared input tensor.

    Args:
        model: model to probe.
        forward_fn: zero-arg callable running a forward pass on ``model``;
            quantizers are disabled during it so probe outputs aren't perturbed.
        module_filter: which modules to hook on input (default
            :func:`is_quantized_linear`).
        output_filter: optional, which modules to hook on output. AWQ export uses
            it to map a layernorm's output to itself so the pre_quant_scale can be
            folded into the layernorm; ``None`` skips output tracking.

    Returns:
        ``(input_to_modules, output_to_modules)``: input tensor -> modules that
        received it (the shared-input group), and output tensor -> producing
        module (``None`` when ``output_filter`` is not given).
    """
    # Inline import to avoid a cycle (conversion/dataset_utils import from utils);
    # safe because this runs at calibration/export time, not module load.
    from modelopt.torch.quantization.conversion import set_quantizer_by_cfg_context
    from modelopt.torch.utils.dataset_utils import _disable_use_cache

    if module_filter is None:
        module_filter = is_quantized_linear

    input_to_modules: dict = defaultdict(list)
    output_to_modules: dict | None = defaultdict(lambda: None) if output_filter else None

    def _input_hook(module, args, output):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_to_modules[args[0]].append(module)

    def _output_hook(module, args, output):
        if output_to_modules is not None and isinstance(output, torch.Tensor):
            output_to_modules[output] = module

    handles = []
    for name, module in model.named_modules():
        if output_filter is not None and output_filter(module):
            module.name = name
            handles.append(module.register_forward_hook(_output_hook))
        elif module_filter(module):
            module.name = name
            handles.append(module.register_forward_hook(_input_hook))

    if not handles:
        return input_to_modules, output_to_modules

    try:
        with (
            torch.no_grad(),
            set_quantizer_by_cfg_context(model, [{"quantizer_name": "*", "enable": False}]),
            _disable_use_cache(model),
        ):
            forward_fn()
    finally:
        for handle in handles:
            handle.remove()

    return input_to_modules, output_to_modules


# ---------------------------------------------------------------------------
# Per-group shared state
# ---------------------------------------------------------------------------


class SharedQuantState(nn.Module):
    """State shared across a sibling group of quantized modules.

    Attached to the group's parent (e.g. ``self_attn``, ``block_sparse_moe``) as a
    registered submodule so its buffers ride along in ``state_dict``. Members that
    quantize the same input resolve shared values here instead of computing them
    independently and reconciling at export.

    Holds only ``weight_global_amax`` today; new tensor fields (act scales, AWQ
    ``pre_quant_scale``, SVDQuant/FlatQuant factors) should use ``register_buffer``
    so they serialize too.
    """

    def __init__(self) -> None:
        """Initialize with an unset ``weight_global_amax`` and no registered members."""
        super().__init__()
        # NVFP4 two-level FP8 grid scale = max over members' per-block ``_amax``.
        # Unset for non-NVFP4 configs.
        self.register_buffer("weight_global_amax", None)
        # Back-references to member modules. ``object.__setattr__`` keeps them out of
        # ``_modules`` so members' params don't re-enter our ``state_dict``.
        object.__setattr__(self, "_members", [])

    def sync_weight_global_amax(self, parallel_state: ParallelState | None) -> None:
        """All-reduce (MAX) ``weight_global_amax`` across EP, plus TP defensively.

        Weights are DP-replicated (no DP sync needed). EP sync is required since
        ranks hold different experts; TP sync guards against per-child ``_amax`` TP
        sync skipping block-quantized weights.
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
                warnings.warn(f"Failed to sync shared weight_global_amax: {e}")


# ---------------------------------------------------------------------------
# Group discovery (hook + pattern) → (parent, members) tuples
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


def _groups_from_hooks(
    model: nn.Module,
    forward_loop: ForwardLoop,
) -> list[tuple[nn.Module, list[nn.Module]]]:
    """Discover sibling groups from a hook-based probe forward.

    Modules whose first input is the same tensor object form a group, parented at
    their LCA. Only catches the *literal same tensor*, so cross-expert MoE sharing
    (one input per expert) is missed — patterns cover that.
    """
    input_to_modules, _ = collect_shared_input_modules(
        model, forward_fn=lambda: forward_loop(model)
    )
    if not input_to_modules:
        return []

    parent_map = _build_parent_map(model)
    wq_attr = quantizer_attr_names("weight").weight_quantizer
    groups: list[tuple[nn.Module, list[nn.Module]]] = []
    for members in input_to_modules.values():
        # Dedup (a module may be hooked twice) and keep calibrated members only.
        unique: list[nn.Module] = []
        seen: set[int] = set()
        for m in members:
            if id(m) in seen:
                continue
            if not _has_calibratable_weight_quantizer(m, wq_attr):
                continue
            seen.add(id(m))
            unique.append(m)
        if len(unique) >= 2:
            parent = _lowest_common_ancestor(unique, parent_map, fallback=model)
            groups.append((parent, unique))
    return groups


def _groups_from_patterns(
    model: nn.Module,
    patterns: Sequence[str],
) -> list[tuple[nn.Module, list[nn.Module]]]:
    r"""Discover groups by regex over module FQNs; capture groups define the grouping key.

    Each pattern is a regex ``re.fullmatch``-ed against every quantized module's
    fully-qualified name; modules whose match yields the same capture-group tuple form
    one group, parented at their LCA. Granularity is set by *what you capture*:

    - Capture the immediate parent -> per-parent grouping: q/k/v per attention block,
      and **per-expert** ``w1``/``w3`` (each expert is the immediate parent), e.g.
      ``r"(.*)\.(?:w1|w3)$"``.
    - Capture only a level above the expert index, leaving the index uncaptured ->
      one **cross-expert** group, e.g. ``r"(.*)\.experts\.\d+\.(?:w1|w3)$"``.

    Roles to fuse together go in a non-capturing alternation ``(?:w1|w3)`` so they
    don't split the key; what you wrap in ``(...)`` is the group boundary.
    """
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
                key = (
                    pattern_idx,
                    match.groups(),
                )  # include pattern_idx in case 1+ partterns collide.
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


def find_shared_input_groups(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    patterns: Sequence[str] | None = None,
) -> list[tuple[nn.Module, list[nn.Module]]]:
    """Find sibling groups from regex patterns if given, else from a hook probe.

    Patterns and hooks are mutually exclusive: when ``patterns`` is set it is the
    sole source (and must list every group you want); otherwise hook discovery runs
    on the ``forward_loop`` probe.

    - ``patterns`` — regexes over module FQNs whose capture groups define the grouping
      key (see :func:`_groups_from_patterns`); the capture boundary chooses per-expert
      vs cross-expert granularity. The caller selects which quantizer these groups apply
      to (today only the weight quantizer; see ``MaxCalibConfig.shared_patterns``).
    - ``forward_loop`` — hook probe grouping modules that receive the *literal same
      tensor* (Q/K/V, gate/up within one block/expert; per-expert for MoE).

    Returns ``(parent, members)`` tuples.
    """
    if patterns:
        return _groups_from_patterns(model, patterns)
    if forward_loop is not None:
        return _groups_from_hooks(model, forward_loop)
    return []


# ---------------------------------------------------------------------------
# Attach / populate lifecycle
# ---------------------------------------------------------------------------


def attach_shared_quant_states(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    patterns: Sequence[str] | None = None,
) -> int:
    """Create ``SharedQuantState`` on each group's parent and link members.

    The parent owns the state under ``_shared_quant_state`` (normal setattr → a
    registered submodule, so its buffer rides along in ``state_dict``). Each
    member's weight quantizer — the only consumer, via
    ``promote_nvfp4_static_quantizers`` — gets a back-reference under the distinct
    name ``_shared_quant_state_ref`` set with ``object.__setattr__`` (not a
    submodule, so the buffer isn't duplicated per member). The distinct names let
    ``populate_shared_state`` select owners with a plain ``getattr``.

    Idempotent (reuses an existing parent state). Returns the number created.
    """
    n_created = 0
    wq_attr = quantizer_attr_names("weight").weight_quantizer
    for parent, members in find_shared_input_groups(
        model, forward_loop=forward_loop, patterns=patterns
    ):
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
