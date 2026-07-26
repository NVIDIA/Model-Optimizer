# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped attention-head scoring hooks for AutoModel.

Puzzletron only prunes full attention in two default prefix-sortable ways:

* remove whole KV groups, along with their corresponding query heads;
* remove the same number of query heads from every surviving KV group.

The hook attaches to ``o_proj`` and observes its input.  This is the generic
post-attention, post-gate tensor for both gated and ungated attention modules, so
Qwen-style attention gates are included in the measured contribution without any
model-specific logic.

The scorer emits explicit fields instead of an ambiguous generic ``score``:

``kv_group_scores``:
    ``[num_kv_heads]`` group importances used to sort/drop whole KV groups.
``query_head_scores``:
    ``[num_kv_heads, heads_per_group]`` importances used to sort query heads
    inside each group.
For whole KV groups the scorer uses the same greedy removal idea as the
iterative FFN scorer: at each calibration step it asks which group would cause
the smallest output perturbation if removed from the currently-pruned state.
Query heads use that same deletion-damage objective.  This is intentionally the
only attention ranking method: it is shared by Qwen and Nemotron models.

Parallelism contract: the hook point may shard the feature axis across TP, while
DP/FSDP/CP shard the token axis.  For the iterative head scores we gather the
feature/head axis over TP inside each forward, leave token shards local, then
SUM-reduce per-iteration token sums over the token group in ``step_iteration``.
That keeps the greedy state identical across ranks without double-counting TP.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as _tdist
from torch.distributed.tensor import DTensor

from modelopt.torch.prune.importance_hooks.base_hooks import clear_gpu_memory, get_pruning_schedule

from ..reduction import (
    full_weight,
    gather_scored_axis,
    reduce_token_sum,
    to_local_with_feature_group,
)
from .base import ScoringHook

_DBG_ITERS = int(os.environ.get("ACTIVATION_SCORE_DEBUG", "0"))

__all__ = ["GroupedAttentionScorer"]


def _normalize_axes(scored_axes) -> set[str]:
    if scored_axes is None:
        return {"kv_groups", "q_heads_per_group"}
    aliases = {
        "kv_heads": "kv_groups",
        "num_kv_heads": "kv_groups",
        "query_heads": "q_heads_per_group",
        "num_query_heads": "q_heads_per_group",
        "query_heads_per_group": "q_heads_per_group",
    }
    axes = set()
    for axis in scored_axes:
        axes.add(aliases.get(str(axis), str(axis)))
    supported = {"kv_groups", "q_heads_per_group"}
    unknown = axes - supported
    if unknown:
        raise ValueError(f"Unknown grouped attention scored_axes: {sorted(unknown)}")
    return axes


class GroupedAttentionScorer(ScoringHook):
    """The canonical iterative KV-group and query-head scorer."""

    target_type = "attn"
    method = "grouped_attention_contribution"

    def __init__(
        self,
        module,
        groups,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        optimize_for: str = "memory",
        validation_full_iters: int | None = None,
        calibration_method: str | None = None,
        clear_gpu_memory: bool = False,
        scored_axes=None,
        token_chunk_size: int | None = None,
        block_idx=None,
        name=None,
    ):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        assert num_q_heads % num_kv_heads == 0, (num_q_heads, num_kv_heads)
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.optimize_for = optimize_for
        self.heads_per_group = self.num_q_heads // self.num_kv_heads
        self.scored_axes = _normalize_axes(scored_axes)
        self._score_kv = "kv_groups" in self.scored_axes
        self._score_q = "q_heads_per_group" in self.scored_axes
        self._score_head_axes = self._score_kv or self._score_q
        if self._score_head_axes and validation_full_iters is None:
            raise ValueError("grouped attention iterative scoring requires validation_full_iters.")
        self.pruning_iters = int(validation_full_iters or 0)
        self.calibration_method = calibration_method
        self._clear = clear_gpu_memory
        self.epsilon = 1e-8
        self.token_chunk_size = int(
            token_chunk_size or os.environ.get("ATTENTION_SCORE_CHUNK_SIZE", 8192)
        )

        self._num_q_local: int | None = None
        self._weight_heads: torch.Tensor | None = None  # [num_q_heads, hidden, head_dim]
        self._hidden: int | None = None
        self._count: int = 0
        self._feature_group = None
        self._debug_call_count = 0

        self.curr_iter = 0
        self._kv_schedule: list[int] | None = None
        self._head_schedule: list[int] | None = None
        self._pruned_kv: list[int] = []
        self._pruned_heads_per_group: list[list[int]] = [
            [] for _ in range(self.num_kv_heads)
        ]
        self._kv_agg: torch.Tensor | None = None
        self._head_agg: torch.Tensor | None = None
        self._pending_kv_sum: torch.Tensor | None = None
        self._pending_head_sum: torch.Tensor | None = None
        self._pending_count: int = 0
        self._resume_kv_agg: torch.Tensor | None = None
        self._resume_head_agg: torch.Tensor | None = None

    def _ensure_weight_stats(self, feature_group) -> None:
        if self._weight_heads is not None:
            return
        weight = full_weight(self.module.weight).float()  # [hidden, num_q_heads * head_dim]
        hidden, in_features = weight.shape
        if in_features != self.num_q_heads * self.head_dim:
            raise RuntimeError(
                f"Attention scorer expected o_proj input features {self.num_q_heads * self.head_dim}, "
                f"got {in_features} for module={self.name}"
            )
        self._hidden = hidden
        self._weight_heads = weight.view(hidden, self.num_q_heads, self.head_dim).permute(1, 0, 2).contiguous()
        if self._score_kv:
            self._kv_schedule = get_pruning_schedule(self.num_kv_heads, self.pruning_iters)
            self._kv_agg = torch.zeros(self.num_kv_heads, dtype=torch.float32, device=weight.device)
        if self._score_q:
            self._head_schedule = get_pruning_schedule(self.heads_per_group, self.pruning_iters)
            self._head_agg = torch.zeros(
                self.num_kv_heads,
                self.heads_per_group,
                dtype=torch.float32,
                device=weight.device,
            )
        if self._resume_kv_agg is not None and self._kv_agg is not None:
            self._kv_agg.copy_(
                self._resume_kv_agg.to(device=weight.device, dtype=self._kv_agg.dtype)
            )
            self._resume_kv_agg = None
        if self._resume_head_agg is not None and self._head_agg is not None:
            self._head_agg.copy_(
                self._resume_head_agg.to(device=weight.device, dtype=self._head_agg.dtype)
            )
            self._resume_head_agg = None

    def _gather_full_heads(self, attn_out: torch.Tensor) -> torch.Tensor:
        local, feature_group = to_local_with_feature_group(attn_out, feature_dim=-1)
        self._feature_group = feature_group
        local_in = local.shape[-1]
        if local_in % self.head_dim != 0:
            raise RuntimeError(
                f"Attention scorer expected local o_proj input features divisible by head_dim: "
                f"local_in={local_in}, head_dim={self.head_dim}, module={self.name}"
            )
        self._num_q_local = local_in // self.head_dim
        flat_local = local.reshape(-1, self._num_q_local, self.head_dim).float()
        flat = gather_scored_axis(flat_local, feature_group, dim=1)
        if flat.shape[1] != self.num_q_heads:
            raise RuntimeError(
                f"Attention scorer reconstructed {flat.shape[1]} query heads, "
                f"expected {self.num_q_heads} for module={self.name}"
            )
        return flat

    def _gather_structured_heads(self, attn_out: torch.Tensor) -> torch.Tensor:
        local, feature_group = to_local_with_feature_group(attn_out, feature_dim=-1)
        self._feature_group = feature_group
        local_in = local.shape[-1]
        if local_in % self.head_dim != 0:
            raise RuntimeError(
                f"Attention scorer expected local o_proj input features divisible by head_dim: "
                f"local_in={local_in}, head_dim={self.head_dim}, module={self.name}"
            )
        self._num_q_local = local_in // self.head_dim
        local_heads = local.reshape(*local.shape[:-1], self._num_q_local, self.head_dim).float()
        full_heads = gather_scored_axis(local_heads, feature_group, dim=-2)
        if full_heads.shape[-2] != self.num_q_heads:
            raise RuntimeError(
                f"Attention scorer reconstructed {full_heads.shape[-2]} query heads, "
                f"expected {self.num_q_heads} for module={self.name}"
            )
        return full_heads

    def _current_kv_keep_mask(self, device) -> torch.Tensor:
        keep = torch.ones(self.num_kv_heads, dtype=torch.bool, device=device)
        if self._pruned_kv:
            keep[self._pruned_kv] = False
        return keep

    def _current_head_keep_mask(self, device) -> torch.Tensor:
        keep = torch.ones(
            self.num_kv_heads,
            self.heads_per_group,
            dtype=torch.bool,
            device=device,
        )
        for group_idx, pruned in enumerate(self._pruned_heads_per_group):
            if pruned:
                keep[group_idx, pruned] = False
        return keep

    def _accumulate_head_axis_scores(self, flat: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._score_head_axes:
            return None, None
        weight = self._weight_heads.float()
        kv_keep = self._current_kv_keep_mask(flat.device) if self._score_kv else None
        head_keep = self._current_head_keep_mask(flat.device) if self._score_q else None
        kv_sum = (
            torch.zeros(self.num_kv_heads, dtype=torch.float64, device=flat.device)
            if self._score_kv
            else None
        )
        head_sum = (
            torch.zeros(
                self.num_kv_heads,
                self.heads_per_group,
                dtype=torch.float64,
                device=flat.device,
            )
            if self._score_q
            else None
        )

        for start in range(0, flat.shape[0], self.token_chunk_size):
            act = flat[start : start + self.token_chunk_size]
            # [tokens, query_heads, hidden]. This is the per-head contribution to o_proj output.
            head_contrib = torch.einsum("nqd,qhd->nqh", act, weight)

            if self._score_kv:
                group_contrib = head_contrib.view(
                    act.shape[0], self.num_kv_heads, self.heads_per_group, self._hidden
                ).sum(dim=2)
                output_full = group_contrib.sum(dim=1)
                output_curr = (group_contrib * kv_keep.view(1, -1, 1)).sum(dim=1)
                if self.calibration_method is None:
                    residual = output_full - output_curr
                elif self.calibration_method == "scale_by_magnitude":
                    scale = torch.linalg.vector_norm(output_curr, dim=-1) / (
                        torch.linalg.vector_norm(output_full, dim=-1) + self.epsilon
                    )
                    residual = scale.unsqueeze(-1) * output_full - output_curr
                else:
                    raise NotImplementedError(self.calibration_method)
                # The downstream diagnosis and KD objectives are mean-squared
                # projected-output error.  Accumulate that same squared deletion
                # damage rather than a per-token L2 norm; summing unsquared norms
                # overweights small token residuals and can reverse a close
                # two-group ordering.
                damage = (residual.unsqueeze(1) + group_contrib).square().sum(dim=-1)
                kv_sum += damage.double().sum(dim=0)
                del group_contrib, output_full, output_curr, residual, damage

            if self._score_q:
                grouped_heads = head_contrib.view(
                    act.shape[0],
                    self.num_kv_heads,
                    self.heads_per_group,
                    self._hidden,
                )
                group_full = grouped_heads.sum(dim=2)
                group_curr = (grouped_heads * head_keep.view(1, self.num_kv_heads, self.heads_per_group, 1)).sum(dim=2)
                if self.calibration_method is None:
                    residual = group_full - group_curr
                elif self.calibration_method == "scale_by_magnitude":
                    scale = torch.linalg.vector_norm(group_curr, dim=-1) / (
                        torch.linalg.vector_norm(group_full, dim=-1) + self.epsilon
                    )
                    residual = scale.unsqueeze(-1) * group_full - group_curr
                else:
                    raise NotImplementedError(self.calibration_method)
                damage = (residual.unsqueeze(2) + grouped_heads).square().sum(dim=-1)
                head_sum += damage.double().sum(dim=0)
                del grouped_heads, group_full, group_curr, residual, damage

            del head_contrib, act
            clear_gpu_memory(clear=self._clear)

        return kv_sum, head_sum

    def __call__(self, module, args, output):
        attn_out = args[0]
        heads = self._gather_structured_heads(attn_out)
        flat = self._flatten_valid_tokens(heads, trailing_dims=2)
        if flat.shape[0] == 0:
            return
        self._ensure_weight_stats(self._feature_group)

        if _DBG_ITERS > 0 and self._debug_call_count < _DBG_ITERS:
            gr = _tdist.get_rank() if _tdist.is_initialized() else 0
            is_dt = isinstance(attn_out, DTensor)
            placements = str(attn_out.placements) if is_dt else "plain"
            local_shape = tuple(attn_out.to_local().shape) if is_dt else tuple(attn_out.shape)
            print(
                f"[DBG:attn_grouped rank={gr} pp={self.groups.pp_rank} tp={self.groups.tp_rank} "
                f"tok={self.groups.token_rank}] {self.name} call={self._debug_call_count} "
                f"input: dtensor={is_dt} placements={placements} local={local_shape} "
                f"full_heads={tuple(flat.shape)} |attn_out|.mean={flat.abs().float().mean().item():.5f}",
                flush=True,
            )

        if self._score_head_axes:
            kv_sum, head_sum = self._accumulate_head_axis_scores(flat)
            if kv_sum is not None:
                self._pending_kv_sum = (
                    kv_sum if self._pending_kv_sum is None else self._pending_kv_sum + kv_sum
                )
            if head_sum is not None:
                self._pending_head_sum = (
                    head_sum if self._pending_head_sum is None else self._pending_head_sum + head_sum
                )

        self._pending_count += flat.shape[0]
        self._count += flat.shape[0]

        if _DBG_ITERS > 0 and self._debug_call_count < _DBG_ITERS:
            gr = _tdist.get_rank() if _tdist.is_initialized() else 0
            if self._pending_kv_sum is not None:
                top = self._pending_kv_sum.float().topk(min(3, self._pending_kv_sum.numel())).indices.tolist()
            else:
                top = []
            print(
                f"[DBG:attn_grouped rank={gr} pp={self.groups.pp_rank} tp={self.groups.tp_rank} "
                f"tok={self.groups.token_rank}] {self.name} n_tokens={flat.shape[0]} "
                f"iter={self.curr_iter} pending_kv_top={top}",
                flush=True,
            )
        self._debug_call_count += 1

    def step_iteration(self) -> None:
        if not self._score_head_axes:
            return
        assert self._pending_count > 0, "step_iteration() called before any attention forward."
        device_tensor = self._pending_kv_sum if self._pending_kv_sum is not None else self._pending_head_sum
        count = torch.tensor(
            float(self._pending_count),
            dtype=torch.float64,
            device=device_tensor.device,
        )
        reduce_token_sum(count, self.groups.token_group)

        if self._score_kv:
            assert self._pending_kv_sum is not None
            kv_sum = self._pending_kv_sum
            reduce_token_sum(kv_sum, self.groups.token_group)
            kv_mean = (kv_sum / count.to(kv_sum.device)).float()
            kv_mean[self._pruned_kv] = torch.inf
            self._kv_agg += kv_mean
            n_to_prune = self._kv_schedule[self.curr_iter]
            if n_to_prune > 0:
                _, worst = torch.topk(self._kv_agg, n_to_prune, largest=False)
                worst_list = worst.tolist()
                assert not set(self._pruned_kv).intersection(worst_list)
                self._pruned_kv.extend(worst_list)
                self._kv_agg.zero_()

        if self._score_q:
            assert self._pending_head_sum is not None
            head_sum = self._pending_head_sum
            reduce_token_sum(head_sum, self.groups.token_group)
            head_mean = (head_sum / count.to(head_sum.device)).float()
            for group_idx, pruned in enumerate(self._pruned_heads_per_group):
                if pruned:
                    head_mean[group_idx, pruned] = torch.inf
            self._head_agg += head_mean
            n_to_prune = self._head_schedule[self.curr_iter]
            if n_to_prune > 0:
                for group_idx in range(self.num_kv_heads):
                    _, worst = torch.topk(self._head_agg[group_idx], n_to_prune, largest=False)
                    worst_list = worst.tolist()
                    assert not set(self._pruned_heads_per_group[group_idx]).intersection(worst_list)
                    self._pruned_heads_per_group[group_idx].extend(worst_list)
                self._head_agg.zero_()

        self.curr_iter += 1
        self._pending_kv_sum = None
        self._pending_head_sum = None
        self._pending_count = 0

    def finalize(self) -> dict:
        device = self.module.weight.device
        # Exact resume may finalize without another forward, so the lazily
        # allocated aggregate buffers have not been rebuilt by
        # ``_ensure_weight_stats``. Rankings are already complete; only a device
        # anchor is needed to materialize their ordinal score tensors.
        if self._score_kv and self._kv_agg is None:
            self._kv_agg = (
                self._resume_kv_agg.to(device=device, dtype=torch.float32)
                if self._resume_kv_agg is not None
                else torch.zeros(self.num_kv_heads, dtype=torch.float32, device=device)
            )
        if self._score_q and self._head_agg is None:
            self._head_agg = (
                self._resume_head_agg.to(device=device, dtype=torch.float32)
                if self._resume_head_agg is not None
                else torch.zeros(
                    self.num_kv_heads,
                    self.heads_per_group,
                    dtype=torch.float32,
                    device=device,
                )
            )
        out = {}
        if self._score_head_axes:
            assert self.curr_iter == self.pruning_iters, (
                f"grouped attention scoring ran {self.curr_iter}/{self.pruning_iters} iterations; "
                "eval_samples // micro_batch_size must match validation_full_iters."
            )
        if self._score_kv:
            assert len(self._pruned_kv) == self.num_kv_heads
            kv_ascending = torch.tensor(self._pruned_kv, dtype=torch.long, device=self._kv_agg.device)
            kv_score = torch.empty(self.num_kv_heads, dtype=torch.float32, device=self._kv_agg.device)
            kv_score[kv_ascending] = torch.arange(
                self.num_kv_heads, dtype=torch.float32, device=self._kv_agg.device
            )
            out["kv_group_scores"] = kv_score
            out["kv_groups_importance_ascending"] = kv_ascending
        if self._score_q:
            head_score = torch.empty(
                self.num_kv_heads,
                self.heads_per_group,
                dtype=torch.float32,
                device=self._head_agg.device,
            )
            ascending = []
            for group_idx, pruned in enumerate(self._pruned_heads_per_group):
                assert len(pruned) == self.heads_per_group
                asc = torch.tensor(pruned, dtype=torch.long, device=self._head_agg.device)
                head_score[group_idx, asc] = torch.arange(
                    self.heads_per_group, dtype=torch.float32, device=self._head_agg.device
                )
                ascending.append(asc)
            out["query_head_scores"] = head_score
            out["query_heads_importance_ascending_per_group"] = ascending
        return out

    def checkpoint_state(self) -> dict:
        return {
            "curr_iter": self.curr_iter,
            "pruned_kv": list(self._pruned_kv),
            "pruned_heads_per_group": [list(values) for values in self._pruned_heads_per_group],
            "kv_agg": None if self._kv_agg is None else self._kv_agg.detach().cpu(),
            "head_agg": None if self._head_agg is None else self._head_agg.detach().cpu(),
            "count": self._count,
        }

    def load_checkpoint_state(self, state: dict) -> None:
        device = self.module.weight.device
        self.curr_iter = int(state["curr_iter"])
        self._pruned_kv = [int(value) for value in state["pruned_kv"]]
        self._pruned_heads_per_group = [
            [int(value) for value in values]
            for values in state["pruned_heads_per_group"]
        ]
        self._resume_kv_agg = state.get("kv_agg")
        self._resume_head_agg = state.get("head_agg")
        self._count = int(state.get("count", 0))
        self._pending_kv_sum = None
        self._pending_head_sum = None
        self._pending_count = 0
