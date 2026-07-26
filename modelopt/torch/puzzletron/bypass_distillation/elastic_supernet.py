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

"""Nested (Matryoshka) elastic bypass: train all removal-pruning variants in one supernet run.

Extends the existing blockwise-LOCAL, teacher-forced bypass (each block independently distills its
output to the teacher's, given the teacher's input) with random per-block prefixes. Each minibatch
samples a size per prunable subblock (``p ~ 1/num_params``, full size included), masks each block to
that prefix (D3 differentiable masks), and takes the local loss vs the teacher — so all blocks at
all sizes train **independently** (teacher-forcing decouples them) in one pass. The resulting
elastic checkpoint is sliced (now *trained*) instead of running N per-variant bypass runs.

This file provides the model-agnostic per-step sampling and masking used by the
native AutoModel local-distillation recipe.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import torch

from ..pruning.dynamic_block_prune import register_mask_hook
from ..pruning.elastic_sampling import ElasticSizeSampler

logger = logging.getLogger(__name__)

ATTENTION_NO_OP_TARGET = (0, 0)

__all__ = [
    "SubblockElastic",
    "ElasticSupernetMasker",
    "CanonicalBlockElastic",
    "CanonicalCandidateMasker",
    "build_subblock_elastics",
    "build_canonical_block_elastics",
    "logical_data_lane_from_peer_sets",
    "make_param_fn",
    "validate_lane_architecture_assignments",
]


def logical_data_lane_from_peer_sets(
    rank: int,
    peer_sets_by_rank,
) -> tuple[int, int]:
    """Return the model-parallel connected component for one global rank."""

    world_size = len(peer_sets_by_rank)
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    parents = list(range(world_size))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for peer_sets in peer_sets_by_rank:
        for peers in peer_sets:
            peers = tuple(int(peer) for peer in peers)
            if any(peer < 0 or peer >= world_size for peer in peers):
                raise ValueError(f"invalid model-parallel peer set {peers}")
            for peer in peers[1:]:
                union(peers[0], peer)

    components: dict[int, list[int]] = defaultdict(list)
    for global_rank in range(world_size):
        components[find(global_rank)].append(global_rank)
    ordered = sorted(components.values(), key=lambda members: min(members))
    lane_by_rank = {
        member: lane
        for lane, members in enumerate(ordered)
        for member in members
    }
    return lane_by_rank[int(rank)], len(ordered)


def validate_lane_architecture_assignments(assignments) -> None:
    """Require one architecture digest for every logical data lane."""

    by_lane: dict[int, set[str]] = defaultdict(set)
    for lane, digest in assignments:
        by_lane[int(lane)].add(str(digest))
    conflicts = {
        lane: sorted(digests)
        for lane, digests in sorted(by_lane.items())
        if len(digests) != 1
    }
    if conflicts:
        details = "; ".join(
            f"lane {lane}: {digests}" for lane, digests in conflicts.items()
        )
        raise RuntimeError(
            f"model-parallel peers selected different architectures for {details}"
        )


@dataclass
class SubblockElastic:
    """One prunable subblock's elastic spec: layout + the size sampler for it."""

    layer_idx: int
    kind: str  # "ffn" | "attn" | "gdn"
    sampler: ElasticSizeSampler


@dataclass
class CanonicalBlockElastic:
    """Complete legal candidates sampled for one decoder layer."""

    layer_idx: int
    parent_block_config: object
    sampler: ElasticSizeSampler


class CanonicalCandidateMasker:
    """Sample and apply one canonical complete block candidate per layer."""

    def __init__(
        self,
        elastics: list[CanonicalBlockElastic],
        *,
        layouts_by_idx: dict,
        head_dim: int,
        seed: int = 42,
    ):
        self.elastics = elastics
        self.layouts_by_idx = layouts_by_idx
        self.head_dim = int(head_dim)
        self.seed = int(seed)
        self.sample_step = 0
        self.coverage: dict[str, dict[str, object]] = {}
        self.coverage_schedules = {
            elastic.layer_idx: self._axis_endpoint_schedule(elastic) for elastic in elastics
        }

    def _balanced_candidate(
        self,
        elastic: CanonicalBlockElastic,
        sample_index: int | None = None,
    ):
        """Visit every complete legal candidate once before a seeded reshuffle."""
        candidates = elastic.sampler.sizes
        count = len(candidates)
        cycle, offset = divmod(
            self.sample_step if sample_index is None else int(sample_index),
            count,
        )
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * int(elastic.layer_idx) + 97_409 * cycle
        )
        index = int(torch.randperm(count, generator=generator)[offset])
        return candidates[index]

    @staticmethod
    def _disabled_subblocks(elastic: CanonicalBlockElastic, candidate) -> tuple[str, ...]:
        """Return only removals of subblocks present in the parent architecture."""
        parent_subblocks = {
            subblock.kind: subblock
            for subblock in elastic.parent_block_config.subblock_configs
        }
        return tuple(
            sorted(
                subblock.kind
                for subblock in candidate.block_config.subblock_configs
                if subblock.no_op
                and not getattr(parent_subblocks.get(subblock.kind), "no_op", True)
            )
        )

    @staticmethod
    def _axis_endpoint_schedule(elastic: CanonicalBlockElastic) -> list[object]:
        """Select a short deterministic teacher/no-op/axis/mixed coverage schedule."""
        candidates = list(elastic.sampler.sizes)

        def changed(candidate):
            return dict(candidate.metadata.get("slice_axes") or {})

        def disabled(candidate):
            return CanonicalCandidateMasker._disabled_subblocks(elastic, candidate)

        schedule: list[object] = []
        teacher = next(
            (
                candidate
                for candidate in candidates
                if not changed(candidate) and not disabled(candidate)
            ),
            None,
        )
        if teacher is not None:
            schedule.append(teacher)

        noops: dict[tuple[str, ...], object] = {}
        for candidate in candidates:
            signature = disabled(candidate)
            if signature and not changed(candidate):
                noops.setdefault(signature, candidate)
        schedule.extend(noops[key] for key in sorted(noops, key=lambda item: (len(item), item)))

        axis_candidates: dict[str, object] = {}
        for candidate in candidates:
            edits = changed(candidate)
            if len(edits) == 1 and not disabled(candidate):
                axis_candidates.setdefault(next(iter(edits)), candidate)
        schedule.extend(axis_candidates[key] for key in sorted(axis_candidates))

        mixed = [
            candidate
            for candidate in candidates
            if len(changed(candidate)) > 1 and not disabled(candidate)
        ]
        if mixed:
            schedule.append(
                max(
                    mixed,
                    key=lambda candidate: (
                        len(changed(candidate)),
                        str(candidate.identity.value),
                    ),
                )
            )

        deduped: list[object] = []
        seen: set[str] = set()
        for candidate in schedule:
            identity = str(candidate.identity.value)
            if identity not in seen:
                deduped.append(candidate)
                seen.add(identity)
        expected_axes = {
            axis
            for candidate in candidates
            if not disabled(candidate)
            for axis in changed(candidate)
        }
        covered_axes = {
            axis
            for candidate in deduped
            for axis in changed(candidate)
        }
        if covered_axes != expected_axes or set(noops) != {
            disabled(candidate) for candidate in deduped if disabled(candidate)
        }:
            raise ValueError(
                f"Incomplete axis-endpoint schedule for layer {elastic.layer_idx}: "
                f"axes={sorted(covered_axes)}/{sorted(expected_axes)}, "
                f"noops={sorted(disabled(candidate) for candidate in deduped if disabled(candidate))}"
            )
        return deduped

    def sample_targets(
        self,
        generator=None,
        *,
        sample_index: int | None = None,
        cycle_all: bool = False,
        coverage_mode: str | None = None,
        selection: str | None = None,
    ):
        choices = {}
        for elastic in self.elastics:
            index = self.sample_step if sample_index is None else int(sample_index)
            lane_generator = generator
            if sample_index is not None:
                lane_generator = torch.Generator().manual_seed(
                    self.seed
                    + 1_000_003 * int(elastic.layer_idx)
                    + 97_409 * index
                )
            if selection == "smallest":
                candidate = elastic.sampler.sizes[
                    int(torch.argmax(elastic.sampler.probs).item())
                ]
            elif selection is not None:
                raise ValueError(
                    f"Unknown canonical candidate selection {selection!r}; "
                    "expected None or 'smallest'"
                )
            elif coverage_mode == "axis_endpoints":
                schedule = self.coverage_schedules[elastic.layer_idx]
                candidate = schedule[index % len(schedule)]
            elif coverage_mode == "coverage_then_uniform":
                schedule = self.coverage_schedules[elastic.layer_idx]
                if index < len(schedule):
                    candidate = schedule[index]
                else:
                    candidates = elastic.sampler.sizes
                    index = int(
                        torch.randint(
                            len(candidates),
                            (),
                            generator=lane_generator,
                        ).item()
                    )
                    candidate = candidates[index]
            elif cycle_all:
                candidate = self._balanced_candidate(elastic, sample_index)
            else:
                candidate = elastic.sampler.sample(lane_generator)
            choices[elastic.layer_idx] = candidate
            key = str(candidate.identity.value)
            entry = self.coverage.setdefault(
                key,
                {
                    "candidate_id": key,
                    "layer_idx": elastic.layer_idx,
                    "block_config": candidate.block_config.to_dict(),
                    "changed_axes": dict(candidate.metadata.get("slice_axes") or {}),
                    "visits": 0,
                },
            )
            entry["visits"] = int(entry["visits"]) + 1
        self.sample_step += 1
        return choices

    def coverage_schedule_manifest(self) -> dict[str, list[dict[str, object]]]:
        return {
            f"layer_{layer_idx}": [
                {
                    "candidate_id": str(candidate.identity.value),
                    "block_config": candidate.block_config.to_dict(),
                    "changed_axes": dict(candidate.metadata.get("slice_axes") or {}),
                    "disabled_subblocks": list(
                        self._disabled_subblocks(
                            next(
                                elastic
                                for elastic in self.elastics
                                if elastic.layer_idx == layer_idx
                            ),
                            candidate,
                        )
                    ),
                }
                for candidate in schedule
            ]
            for layer_idx, schedule in sorted(self.coverage_schedules.items())
        }

    def apply(self, model, choices: dict[int, object]) -> list:
        from ..pruning.attention_ffn_surgery import attention_keep_mask, ffn_keep_mask
        from ..pruning.dynamic_block_prune import register_mask_hook
        from ..pruning.runtime_candidate import apply_runtime_candidate

        handles = []
        by_layer = {elastic.layer_idx: elastic for elastic in self.elastics}
        for layer_idx, candidate in choices.items():
            decoder = _find_decoder_layer(model, layer_idx)
            if decoder is None:
                continue
            elastic = by_layer[layer_idx]
            parent = elastic.parent_block_config
            child = candidate.block_config
            handles.append(apply_runtime_candidate(decoder, parent, child))
            layout = self.layouts_by_idx[layer_idx]

            parent_ffn = parent.get_subblock("ffn")
            child_ffn = child.get_subblock("ffn")
            if (
                parent_ffn is not None
                and child_ffn is not None
                and not child_ffn.no_op
                and child_ffn.intermediate_size < parent_ffn.intermediate_size
            ):
                down = _submodule(decoder, _rel_module_name(layout.down_key))
                if down is not None:
                    keep = torch.arange(int(child_ffn.intermediate_size))
                    handles.append(
                        register_mask_hook(down, ffn_keep_mask(int(parent_ffn.intermediate_size), keep))
                    )

            parent_attn = parent.get_subblock("attention")
            child_attn = child.get_subblock("attention")
            if (
                parent_attn is not None
                and child_attn is not None
                and not child_attn.no_op
                and (
                    child_attn.num_query_heads < parent_attn.num_query_heads
                    or child_attn.num_kv_heads < parent_attn.num_kv_heads
                )
            ):
                from ..pruning.attention_ffn_surgery import sorted_attention_keep_indices

                output = _submodule(decoder, _rel_module_name(layout.o_key))
                if output is not None:
                    keep_q, _ = sorted_attention_keep_indices(
                        int(child_attn.num_kv_heads),
                        int(child_attn.num_query_heads) // int(child_attn.num_kv_heads),
                        int(parent_attn.num_query_heads) // int(parent_attn.num_kv_heads),
                    )
                    handles.append(
                        register_mask_hook(
                            output,
                            attention_keep_mask(
                                int(parent_attn.num_query_heads), keep_q, self.head_dim
                            ),
                        )
                    )
        return handles


def build_canonical_block_elastics(
    block_configs,
    *,
    search_space,
    model_config,
    descriptor,
    include_no_op: bool,
) -> list[CanonicalBlockElastic]:
    """Build complete legal per-layer candidates using the canonical builder."""

    from ..candidates import build_candidate_library
    from ..subblock_stats.calc_subblock_params_and_memory import calc_subblock_active_params

    candidates = build_candidate_library(
        block_configs,
        search_space=search_space,
        parent_checkpoint_identity="elastic_sorted_teacher",
        include_self=True,
        include_noops=include_no_op,
    )
    by_layer: dict[int, list] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.source_kind not in {"self", "sorted_slice", "no_op"}:
            continue
        identity = json.dumps(candidate.block_config.to_dict(), sort_keys=True)
        if identity in seen[candidate.layer_idx]:
            continue
        seen[candidate.layer_idx].add(identity)
        by_layer[candidate.layer_idx].append(candidate)

    cost_cache: dict[str, int] = {}
    hidden_size = int(descriptor.get_language_model_config(model_config).hidden_size)

    def active_cost(block_config) -> int:
        total = 0
        for subblock in block_config.subblock_configs:
            key = repr(subblock)
            if key not in cost_cache:
                cost_cache[key] = calc_subblock_active_params(
                    subblock,
                    model_config,
                    descriptor,
                    hidden_size,
                )
            total += cost_cache[key]
        return total

    elastics = []
    for layer_idx, parent in enumerate(block_configs):
        layer_candidates = by_layer.get(layer_idx) or []
        if len(layer_candidates) <= 1:
            continue
        elastics.append(
            CanonicalBlockElastic(
                layer_idx=layer_idx,
                parent_block_config=parent,
                sampler=ElasticSizeSampler(
                    layer_candidates,
                    [active_cost(candidate.block_config) for candidate in layer_candidates],
                ),
            )
        )
    return elastics


def make_param_fn(model_config, descriptor):
    """Canonical per-subblock param counter (builds a meta layer + counts; GQA/gated/biases exact).

    Returns ``param_fn(subblock_config) -> int`` wrapping
    ``calculate_subblock_params(model_config, subblock_config, descriptor)``, memoized by config repr
    (the count is layer-independent for a homogeneous teacher, so we avoid rebuilding a meta layer
    per layer). This is the SAME function block-stats uses, so the sampling weights match exactly.
    """
    from ..subblock_stats.calc_subblock_params_and_memory import calculate_subblock_params

    cache: dict[str, int] = {}

    def _fn(subblock_config) -> int:
        key = str(subblock_config)
        if key not in cache:
            cache[key] = calculate_subblock_params(model_config, subblock_config, descriptor)
        return cache[key]

    return _fn


def build_subblock_elastics(
    layouts,
    *,
    param_fn,
    ffn_sizes: list[int] | None = None,
    attn_targets: list[tuple[int, int]] | None = None,
    gdn_targets: list[tuple[int, int, int, int]] | None = None,
    moe_latent_sizes: list[int] | None = None,
    include_no_op: bool = True,
) -> list[SubblockElastic]:
    """Build per-subblock samplers; param counts come from ``param_fn`` (use :func:`make_param_fn`).

    ``ffn_sizes`` are intermediate sizes (each subblock's teacher intermediate is added);
    ``attn_targets`` are ``(q, kv)`` pairs (each subblock's teacher (q, kv) is added).  When
    ``include_no_op`` is true, FFN size ``0`` and attention target ``(0, 0)`` are added as explicit
    subblock NoOp endpoints. For each candidate size, ``param_fn`` is called with the matching
    ``FFNConfig`` / ``AttentionConfig`` so the ``p ~ 1/num_params`` weights use the exact
    (GQA-/gated-aware) param counts.

    ``moe_latent_sizes`` are target latent-dim sizes for layers that have a MoE latent
    projection (``fc1_latent_proj`` / ``fc2_latent_proj``).  The teacher latent dim is
    always included.  There is no no-op for the latent axis; the sampling weight is
    proportional to ``1/size`` (smaller = cheaper, sampled more).
    """
    from ..block_config import AttentionConfig, FFNConfig, MambaConfig

    elastics: list[SubblockElastic] = []
    for layout in layouts:
        if layout.down_key is not None and layout.ffn_intermediate:
            raw_sizes = {int(s) for s in (ffn_sizes or [])}
            raw_sizes.add(int(layout.ffn_intermediate))
            if include_no_op:
                raw_sizes.add(0)
            sizes = sorted(s for s in raw_sizes if 0 <= s <= layout.ffn_intermediate)
            params = [
                param_fn(FFNConfig(no_op=True)) if s == 0 else param_fn(FFNConfig(intermediate_size=s))
                for s in sizes
            ]
            elastics.append(SubblockElastic(layout.layer_idx, "ffn", ElasticSizeSampler(sizes, params)))
        if layout.o_key is not None and layout.num_kv_heads:
            full = (layout.num_q_heads, layout.num_kv_heads)
            orig_heads_in_group = full[0] // full[1]
            targets = set()
            for q_raw, kv_raw in attn_targets or []:
                q = int(q_raw)
                kv = int(kv_raw)
                if (q, kv) == ATTENTION_NO_OP_TARGET:
                    targets.add(ATTENTION_NO_OP_TARGET)
                    continue
                if q <= 0 or kv <= 0:
                    raise ValueError(
                        f"Invalid attention elastic target {(q, kv)} for layer {layout.layer_idx}; "
                        "use (0, 0) for attention NoOp or positive (num_query_heads, num_kv_heads)."
                    )
                if q > full[0] or kv > full[1] or q % kv != 0:
                    continue
                if q // kv > orig_heads_in_group:
                    raise ValueError(
                        f"Invalid attention elastic target {(q, kv)} for layer {layout.layer_idx}; "
                        "q/kv cannot exceed the teacher heads-per-KV-group "
                        f"({orig_heads_in_group}). This path does not support q-preserving KV merges."
                    )
                targets.add((q, kv))
            targets.add(full)
            if include_no_op:
                targets.add(ATTENTION_NO_OP_TARGET)
            targets = sorted(targets)
            params = [
                param_fn(AttentionConfig(no_op=True))
                if (q, kv) == ATTENTION_NO_OP_TARGET
                else param_fn(AttentionConfig(num_query_heads=q, num_kv_heads=kv))
                for q, kv in targets
            ]
            elastics.append(SubblockElastic(layout.layer_idx, "attn", ElasticSizeSampler(targets, params)))
        if layout.gated_delta_net and layout.mamba_prefix:
            full = (
                int(layout.mamba_num_groups),
                int(layout.mamba_num_heads) // int(layout.mamba_num_groups),
                int(layout.mamba_state_dim),
                int(layout.mamba_head_dim),
            )
            targets = {
                tuple(int(value) for value in target)
                for target in (gdn_targets or ())
                if all(int(value) > 0 for value in target)
            }
            targets.add(full)
            targets = {
                target
                for target in targets
                if target[0] <= full[0]
                and target[1] <= full[1]
                and target[2] <= full[2]
                and target[3] <= full[3]
            }
            if include_no_op:
                targets.add((0, 0, 0, 0))
            targets = sorted(targets)
            params = [
                param_fn(MambaConfig(no_op=True))
                if target == (0, 0, 0, 0)
                else param_fn(
                    MambaConfig(
                        num_groups=target[0],
                        num_heads=target[0] * target[1],
                        state_dim=target[2],
                        head_dim=target[3],
                    )
                )
                for target in targets
            ]
            elastics.append(
                SubblockElastic(layout.layer_idx, "gdn", ElasticSizeSampler(targets, params))
            )
        if layout.moe_latent_dim and layout.moe_fc1_latent_key:
            full_latent = int(layout.moe_latent_dim)
            raw_sizes = {int(s) for s in (moe_latent_sizes or [])}
            raw_sizes.add(full_latent)
            sizes = sorted(s for s in raw_sizes if 0 < s <= full_latent)
            # Use the size itself as the param proxy: p ~ 1/size gives smaller
            # latent dims higher sampling frequency, matching the FFN/GDN convention.
            elastics.append(
                SubblockElastic(
                    layout.layer_idx,
                    "moe_latent",
                    ElasticSizeSampler(sizes, sizes),
                )
            )
    return elastics


def gdn_targets_from_search_space(
    search_space,
    *,
    full: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Resolve the enabled semantic GDN axes into one legal coupled Cartesian grid."""
    axes = search_space.get("axes", {}) if search_space is not None else {}

    def values(axis: str, teacher: int) -> list[int]:
        spec = axes.get(axis, {}) if axes is not None else {}
        configured = list(spec.get("values", []) or []) if spec.get("enabled", False) else []
        return sorted({teacher, *(int(value) for value in configured if 0 < int(value) <= teacher)})

    groups, ratio, key_dim, value_dim = full
    return [
        tuple(target)
        for target in product(
            values("gdn_key_groups", groups),
            values("gdn_value_heads_per_group", ratio),
            values("gdn_key_head_dim", key_dim),
            values("gdn_value_head_dim", value_dim),
        )
    ]


class ElasticSupernetMasker:
    """Samples a per-step config (one size per subblock) and applies/removes the D3 masks.

    Pure sampling (:meth:`sample_targets`) is CPU-testable; :meth:`apply` resolves each subblock's
    module on the loaded model (by ``layer_idx`` + standard names) and registers the differentiable
    input mask, returning handles to remove after the step.
    """

    def __init__(self, elastics: list[SubblockElastic], *, head_dim: int):
        self.elastics = elastics
        self.head_dim = head_dim
        self.sample_step = 0
        self.coverage: dict[str, set[str]] = defaultdict(set)
        by_layer: dict[int, list[SubblockElastic]] = defaultdict(list)
        for elastic in elastics:
            by_layer[elastic.layer_idx].append(elastic)
        self.elastics_by_layer = dict(by_layer)

    def sample_targets(
        self,
        generator: torch.Generator | None = None,
        *,
        cycle_all: bool = False,
    ) -> dict[int, dict]:
        """Sample a size for every subblock -> ``{layer_idx: {"ffn": K, "attn": (q, kv)}}``."""
        out: dict[int, dict] = {}
        for layer_idx, elastics in self.elastics_by_layer.items():
            if cycle_all:
                choice = {
                    e.kind: e.sampler.sizes[(self.sample_step + layer_idx) % len(e.sampler.sizes)]
                    for e in elastics
                }
            else:
                choice = {e.kind: e.sampler.sample(generator) for e in elastics}
            out[layer_idx] = choice
            for kind, target in choice.items():
                self.coverage[f"layer_{layer_idx}.{kind}"].add(str(target))
        self.sample_step += 1
        return out

    def apply(self, model, targets: dict[int, dict], layouts_by_idx: dict) -> list:
        """Register differentiable masks for the sampled ``targets`` on the loaded model."""
        from ..pruning.attention_ffn_surgery import (
            attention_keep_mask,
            ffn_keep_mask,
            sorted_attention_keep_indices,
        )

        handles = []
        for layer_idx, choice in targets.items():
            layout = layouts_by_idx[layer_idx]
            decoder = _find_decoder_layer(model, layer_idx)
            if decoder is None:
                continue
            k = choice.get("ffn")
            if k is not None and k < (layout.ffn_intermediate or 0):
                down = _submodule(decoder, _rel_module_name(layout.down_key))
                if down is not None:
                    handles.append(register_mask_hook(down, ffn_keep_mask(layout.ffn_intermediate, torch.arange(k))))
            qkv = choice.get("attn")
            if qkv is not None and (qkv[0] < layout.num_q_heads or qkv[1] < layout.num_kv_heads):
                q, kv = qkv
                o = _submodule(decoder, _rel_module_name(layout.o_key))
                if o is not None:
                    if (q, kv) == ATTENTION_NO_OP_TARGET:
                        keep_q = torch.empty(0, dtype=torch.long)
                    else:
                        if q <= 0 or kv <= 0:
                            raise ValueError(
                                f"Invalid attention elastic target {(q, kv)} for layer {layer_idx}"
                            )
                        keep_q, _ = sorted_attention_keep_indices(
                            kv, q // kv, layout.num_q_heads // layout.num_kv_heads
                        )
                    handles.append(register_mask_hook(o, attention_keep_mask(layout.num_q_heads, keep_q, self.head_dim)))
            gdn = choice.get("gdn")
            if gdn is not None:
                handles.extend(_apply_gdn_prefix_hooks(decoder, layout, tuple(gdn)))
            moe_latent = choice.get("moe_latent")
            if (
                moe_latent is not None
                and layout.moe_latent_dim is not None
                and int(moe_latent) < int(layout.moe_latent_dim)
                and layout.moe_fc1_latent_key is not None
            ):
                latent_keep = torch.zeros(int(layout.moe_latent_dim), dtype=torch.bool)
                latent_keep[: int(moe_latent)] = True
                fc1 = _submodule(decoder, _rel_module_name(layout.moe_fc1_latent_key))
                if fc1 is not None:
                    handles.append(fc1.register_forward_hook(_output_mask_hook(latent_keep)))
                if layout.moe_fc2_latent_key is not None:
                    fc2 = _submodule(decoder, _rel_module_name(layout.moe_fc2_latent_key))
                    if fc2 is not None:
                        handles.append(
                            fc2.register_forward_pre_hook(
                                lambda module, args, km=latent_keep: (
                                    args[0]
                                    * km.to(dtype=args[0].dtype, device=args[0].device).reshape(
                                        (1,) * (args[0].ndim - 1) + (-1,)
                                    ),
                                    *args[1:],
                                )
                            )
                        )
        return handles


def _is_no_op_choice(kind: str, choice) -> bool:
    if kind == "ffn":
        return int(choice) == 0
    if kind == "attn":
        return tuple(choice) == ATTENTION_NO_OP_TARGET
    if kind == "gdn":
        return tuple(choice) == (0, 0, 0, 0)
    return False


def _output_mask_hook(keep_mask: torch.Tensor, *, scale: float = 1.0):
    from ..pruning.dynamic_block_prune import _apply_feature_mask

    def hook(module, args, output):
        return _apply_feature_mask(output, keep_mask) * scale

    return hook


def _norm_prefix_prehook(keep_mask: torch.Tensor, scale: float):
    from ..pruning.dynamic_block_prune import _apply_feature_mask

    def hook(module, args):
        x, z, *rest = args
        return (
            _apply_feature_mask(x, keep_mask) * scale,
            _apply_feature_mask(z, keep_mask),
            *rest,
        )

    return hook


def _apply_gdn_prefix_hooks(decoder, layout, target: tuple[int, int, int, int]) -> list:
    """Install an exact padded-prefix GDN child for one training step."""
    from ..pruning.dynamic_block_prune import register_mask_hook
    from ..pruning.gated_delta_net import GDNShape, gated_delta_net_prefix_indices

    module = _submodule(decoder, "linear_attn")
    if module is None:
        return []
    full = GDNShape(
        num_key_heads=int(layout.mamba_num_groups),
        num_value_heads=int(layout.mamba_num_heads),
        key_head_dim=int(layout.mamba_state_dim),
        value_head_dim=int(layout.mamba_head_dim),
    )
    if target == (0, 0, 0, 0):
        return [
            register_mask_hook(
                module.out_proj,
                torch.zeros(full.num_value_heads * full.value_head_dim, dtype=torch.bool),
            )
        ]

    groups, ratio, key_dim, value_dim = (int(value) for value in target)
    child = GDNShape(
        num_key_heads=groups,
        num_value_heads=groups * ratio,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
    )
    idx = gated_delta_net_prefix_indices(full, child)

    def mask(length: int, keep: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(length, dtype=torch.bool)
        result[keep] = True
        return result

    qkv_mask = mask(full.num_key_heads * full.key_head_dim * 2 + full.num_value_heads * full.value_head_dim, idx["cidx"])
    value_mask = mask(full.num_value_heads * full.value_head_dim, idx["vidx"])
    head_mask = mask(full.num_value_heads, idx["hidx"])
    value_dim_mask = torch.arange(full.value_head_dim) < child.value_head_dim
    norm_scale = (full.value_head_dim / child.value_head_dim) ** 0.5

    handles = [
        module.in_proj_qkv.register_forward_hook(_output_mask_hook(qkv_mask)),
        module.in_proj_z.register_forward_hook(_output_mask_hook(value_mask)),
        module.in_proj_a.register_forward_hook(_output_mask_hook(head_mask)),
        module.in_proj_b.register_forward_hook(_output_mask_hook(head_mask)),
        register_mask_hook(module.out_proj, value_mask),
    ]
    if child.value_head_dim < full.value_head_dim:
        handles.extend(
            [
                module.norm.register_forward_pre_hook(
                    _norm_prefix_prehook(value_dim_mask, norm_scale)
                ),
                module.norm.register_forward_hook(
                    _output_mask_hook(value_dim_mask, scale=1.0 / norm_scale)
                ),
            ]
        )
    return handles


def _is_all_no_op_block_choice(choice: dict) -> bool:
    """Reject only true whole-block NoOp samples when both sides are prunable."""
    has_ffn = "ffn" in choice
    has_attn = "attn" in choice
    return has_ffn and has_attn and _is_no_op_choice("ffn", choice["ffn"]) and _is_no_op_choice("attn", choice["attn"])


def _force_one_active_endpoint(choice: dict, elastics: list[SubblockElastic]) -> dict:
    """Deterministic escape hatch if random resampling somehow keeps drawing all-NoOp."""
    fixed = dict(choice)
    for elastic in elastics:
        for candidate in reversed(elastic.sampler.sizes):
            if not _is_no_op_choice(elastic.kind, candidate):
                fixed[elastic.kind] = candidate
                return fixed
    return fixed


def _rel_module_name(weight_key: str) -> str:
    """Submodule path relative to its decoder layer from a layout weight key.

    Layout keys are descriptor-built (e.g. ``model...layers.5.mlp.down_proj.weight`` or, for other
    families, ``...mixer.down_proj.weight``); the module relative to the decoder layer is the last
    two dotted components without ``.weight`` — so this works without hardcoding ``mlp``/``self_attn``.
    """
    return ".".join(weight_key[: -len(".weight")].split(".")[-2:])


def _find_decoder_layer(model, layer_idx):
    from torch import nn

    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "layers":
            continue
        if isinstance(module, nn.ModuleDict) and str(layer_idx) in module:
            return module[str(layer_idx)]
        if isinstance(module, nn.ModuleList) and layer_idx < len(module):
            return module[layer_idx]
    return None


def _submodule(layer, name):
    try:
        return layer.get_submodule(name)
    except AttributeError:
        return None
