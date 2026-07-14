# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba activation scorers for Nemotron3-style Mamba2 mixers."""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from ..reduction import (
    finalize_additive,
    gather_scored_axis,
    reduce_token_sum,
    to_local_with_feature_group,
)
from .base import ScoringHook
from .samplewise import flatten_sample_tokens

__all__ = ["MambaInProjContributionScorer"]


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _split_in_proj(
    projected: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    num_groups: int,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Nemotron3 Mamba in_proj output into ``gate, x, B, C, dt``."""
    inner = int(num_heads) * int(head_dim)
    state = int(num_groups) * int(state_dim)
    gate, hidden_bc, dt = projected.split([inner, inner + 2 * state, int(num_heads)], dim=-1)
    x, b, c = hidden_bc.split([inner, state, state], dim=-1)
    return gate, x, b, c, dt


class MambaInProjContributionScorer(ScoringHook):
    """Reproduce the ModelOpt MiniTron Mamba ``x`` ranking exactly.

    MiniTron first averages ``abs(x)`` over the sequence for each sample,
    squares those per-sample means, and sums over samples.  The square root is
    the per-feature score.  Head dimensions use one global order shared by all
    heads; heads use a group-local order.  The older tokenwise ``E[x^2]`` and
    per-head dimension orders were evaluated here and diagnosed poorly, so they
    intentionally are not retained as selectable alternatives.
    """

    target_type = "mamba"
    method = "mamba_head_and_dim"
    checkpoint_tensor_names = ("_sum_sq",)

    def __init__(
        self,
        module,
        groups,
        *,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        state_dim: int,
        block_idx=None,
        name=None,
    ):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_groups = int(num_groups)
        self.state_dim = int(state_dim)
        self._sum_sq: torch.Tensor | None = None
        self._feature_group = None
        self._sequence_ids: torch.Tensor | None = None
        self._sequence_cursor = 0
        self._num_samples = 0

    def set_batch_metadata(self, *, sequence_ids: torch.Tensor, num_samples: int) -> None:
        if sequence_ids.ndim != 2:
            raise ValueError("Mamba sequence_ids must be [batch, sequence]")
        if int(num_samples) < 1:
            raise ValueError("Mamba num_samples must be positive")
        self._sequence_ids = sequence_ids
        self._sequence_cursor = 0
        self._num_samples = int(num_samples)

    def __call__(self, module, args, output):
        projected, feature_group = to_local_with_feature_group(output, feature_dim=-1)
        if feature_group is not None:
            projected = gather_scored_axis(projected, feature_group, dim=-1)
        self._feature_group = None
        _, x, _, _, _ = _split_in_proj(
            projected,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_groups=self.num_groups,
            state_dim=self.state_dim,
        )
        structured = x.reshape(*x.shape[:-1], self.num_heads, self.head_dim).float()
        if self._sequence_ids is None:
            raise RuntimeError("Mamba scorer batch metadata was not set before forward")
        per_token, ids, self._sequence_cursor = flatten_sample_tokens(
            structured.abs(),
            scored_dim=structured.ndim - 1,
            sequence_ids=self._sequence_ids,
            sequence_cursor=self._sequence_cursor,
        )
        per_token = per_token.reshape(per_token.shape[0], self.num_heads, self.head_dim).double()
        ids = ids.to(device=structured.device, dtype=torch.long)
        valid = (ids >= 0) & (ids < self._num_samples)
        sums = torch.zeros(
            self._num_samples,
            self.num_heads,
            self.head_dim,
            dtype=torch.float64,
            device=structured.device,
        )
        counts = torch.zeros(self._num_samples, dtype=torch.float64, device=structured.device)
        if bool(valid.any()):
            sums.index_add_(0, ids[valid], per_token[valid])
            counts.index_add_(0, ids[valid], torch.ones_like(ids[valid], dtype=torch.float64))
        reduce_token_sum(sums, self.groups.cp_group)
        reduce_token_sum(counts, self.groups.cp_group)
        present = counts > 0
        partial = (sums[present] / counts[present, None, None]).square().sum(dim=0)
        if self.groups.cp_rank != 0:
            partial.zero_()
        self._sum_sq = partial if self._sum_sq is None else self._sum_sq + partial

    def finalize(self) -> dict:
        if self._sum_sq is None:
            raise RuntimeError("No Mamba in_proj activations captured before finalize().")
        full = finalize_additive(
            self._sum_sq,
            feature_group=self._feature_group,
            groups=self.groups,
            scored_dim=0,
        )
        feature_scores = full.clamp_min(0).sqrt()
        # MiniTron uses one dimension order for every head.
        head_dim_scores = torch.linalg.vector_norm(feature_scores, ord=2, dim=0).float()
        head_scores = torch.linalg.vector_norm(feature_scores, ord=2, dim=-1).float()
        heads_per_group = self.num_heads // max(self.num_groups, 1)
        grouped = head_scores.view(self.num_groups, heads_per_group)
        return {
            "score": head_scores,
            "mamba_head_scores": grouped,
            "mamba_head_dim_scores": head_dim_scores,
            "x_scores": feature_scores.float().reshape(-1),
            "heads_importance_ascending_per_group": torch.argsort(grouped, dim=-1),
            "head_dims_importance_ascending": torch.argsort(head_dim_scores),
        }
