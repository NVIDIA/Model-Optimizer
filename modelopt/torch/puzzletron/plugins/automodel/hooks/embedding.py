# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact packed MiniTron residual-width scoring at one normalization site."""

from __future__ import annotations

import torch

from ..reduction import gather_scored_axis, reduce_token_sum, to_local_with_feature_group
from .base import ScoringHook

__all__ = ["HiddenWidthSiteScorer"]


def _tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class HiddenWidthSiteScorer(ScoringHook):
    """Accumulate sqrt(sum_samples(mean_tokens(abs(x)))^2) for one norm site."""

    target_type = "embedding"
    method = "minitron_hidden_width"
    checkpoint_tensor_names = ("_squared_sum",)
    checkpoint_scalar_names = (
        "_sample_count",
        "_local_hidden_size",
        "_feature_sharded",
    )

    def __init__(self, module, groups, *, hidden_size: int, block_idx=None, name=None):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        self.hidden_size = int(hidden_size)
        self._squared_sum: torch.Tensor | None = None
        self._sample_count = 0
        self._sequence_ids: torch.Tensor | None = None
        self._sequence_cursor = 0
        self._num_samples = 0
        self._feature_group = None
        self._local_hidden_size: int | None = None
        self._feature_sharded = False
        self._restored_from_checkpoint = False
        self._feature_layout_restored = False

    def checkpoint_state(self) -> dict:
        return {
            "_squared_sum": (
                None if self._squared_sum is None else self._squared_sum.detach().cpu()
            ),
            "_sample_count": self._sample_count,
            "_local_hidden_size": self._local_hidden_size,
            "_feature_sharded": self._feature_sharded,
        }

    def load_checkpoint_state(self, state: dict) -> None:
        # Checkpoints written before activation-layout metadata existed contain
        # only the accumulator and count. A full-width accumulator proves the
        # residual axis was replicated; this is also the safe legacy default for
        # a zero-owning CP peer from that format.
        old_keys = {"_squared_sum", "_sample_count"}
        new_keys = {*old_keys, "_local_hidden_size", "_feature_sharded"}
        if set(state) not in (old_keys, new_keys):
            raise RuntimeError(
                f"HiddenWidthSiteScorer resume keys={sorted(state)}, "
                f"expected={sorted(new_keys)} (or legacy {sorted(old_keys)})"
            )
        saved = state["_squared_sum"]
        try:
            device = next(self.module.parameters()).device
        except StopIteration:
            device = None
        self._squared_sum = None if saved is None else saved.to(device=device)
        self._sample_count = int(state["_sample_count"])
        if set(state) == new_keys:
            local_size = state["_local_hidden_size"]
            self._local_hidden_size = None if local_size is None else int(local_size)
            self._feature_sharded = bool(state["_feature_sharded"])
        else:
            self._local_hidden_size = (
                int(saved.numel()) if saved is not None else self.hidden_size
            )
            self._feature_sharded = self._local_hidden_size != self.hidden_size
        self._restored_from_checkpoint = True

    def _restore_feature_layout(self) -> None:
        """Recover the non-serializable activation group after an exact restart."""
        if not self._restored_from_checkpoint or self._feature_layout_restored:
            return
        local_size = int(self._local_hidden_size or self.hidden_size)
        self._feature_group = self.groups.tp_group if self._feature_sharded else None
        if self._squared_sum is None:
            self._squared_sum = torch.zeros(
                local_size,
                dtype=torch.float32,
                device=next(self.module.parameters()).device,
            )
        self._feature_layout_restored = True

    def set_batch_metadata(self, *, sequence_ids: torch.Tensor, num_samples: int) -> None:
        if sequence_ids.ndim != 2:
            raise ValueError("hidden-width sequence_ids must be [batch, sequence]")
        if num_samples <= 0:
            raise ValueError("hidden-width num_samples must be positive")
        self._sequence_ids = sequence_ids
        self._sequence_cursor = 0
        self._num_samples = int(num_samples)

    def __call__(self, module, args, output):
        del module, args
        if self._sequence_ids is None:
            raise RuntimeError("hidden-width scorer batch metadata was not set before forward")
        raw_output = _tensor(output)
        activations, feature_group = to_local_with_feature_group(raw_output, feature_dim=-1)
        self._feature_group = feature_group
        self._local_hidden_size = int(activations.shape[-1])
        self._feature_sharded = feature_group is not None
        ids = self._sequence_ids
        if activations.ndim == 2:
            sequence_length = int(ids.shape[1])
            token_count = int(activations.shape[0])
            if sequence_length <= 0 or token_count % sequence_length:
                raise ValueError(
                    "packed hidden-width norm output cannot be aligned with canonical "
                    f"sequence IDs: activations={tuple(activations.shape)} ids={tuple(ids.shape)}"
                )
            activation_batch = token_count // sequence_length
            activations = activations.reshape(
                activation_batch,
                sequence_length,
                int(activations.shape[-1]),
            )
        if activations.ndim != 3:
            raise ValueError(
                "hidden-width norm output must be [batch, sequence, hidden] or "
                f"packed [tokens, hidden], got {tuple(activations.shape)}"
            )
        activation_batch, activation_length = map(int, activations.shape[:2])
        if int(ids.shape[1]) != activation_length:
            placements = getattr(raw_output, "placements", None)
            raise ValueError(
                f"hidden-width sequence IDs {tuple(ids.shape)} do not match activations "
                f"{tuple(activations.shape[:2])}; raw_type={type(raw_output).__name__} "
                f"raw_shape={tuple(raw_output.shape)} local_shape={tuple(activations.shape)} "
                f"placements={placements}"
            )
        stop = self._sequence_cursor + activation_batch
        if stop > int(ids.shape[0]):
            raise ValueError(
                "hidden-width PP microbatches consumed more rows than the canonical batch: "
                f"cursor={self._sequence_cursor} microbatch={activation_batch} "
                f"available={int(ids.shape[0])}"
            )
        ids = ids[self._sequence_cursor:stop]
        self._sequence_cursor = stop
        ids = ids.to(device=activations.device, dtype=torch.long)
        values = activations.detach().float().abs()
        local_hidden = int(values.shape[-1])
        sums = torch.zeros(
            self._num_samples,
            local_hidden,
            dtype=torch.float64,
            device=values.device,
        )
        # Every CP peer must enter finalize's token-group collectives. CP rank 0
        # owns the de-duplicated per-sample contribution; the remaining peers
        # contribute explicit zeros rather than having no accumulator at all.
        if self._squared_sum is None:
            self._squared_sum = torch.zeros(
                local_hidden,
                dtype=torch.float32,
                device=values.device,
            )
        counts = torch.zeros(self._num_samples, dtype=torch.float64, device=values.device)
        flat_ids = ids.reshape(-1)
        flat_values = values.reshape(-1, local_hidden).double()
        valid = (flat_ids >= 0) & (flat_ids < self._num_samples)
        if bool(valid.any()):
            valid_ids = flat_ids[valid]
            sums.index_add_(0, valid_ids, flat_values[valid])
            counts.index_add_(0, valid_ids, torch.ones_like(valid_ids, dtype=torch.float64))
        reduce_token_sum(sums, self.groups.cp_group)
        reduce_token_sum(counts, self.groups.cp_group)

        # CP peers reconstructed the same per-sample means. Keep one copy; the final
        # dp_cp reduction then sums disjoint DP samples without CP duplication.
        if self.groups.cp_rank == 0:
            present = counts > 0
            if bool(present.any()):
                means = sums[present] / counts[present].unsqueeze(1)
                squared = means.square().sum(dim=0).float()
                self._squared_sum += squared
                self._sample_count += int(present.sum().item())

    def finalize(self) -> dict:
        self._restore_feature_layout()
        if self._squared_sum is None:
            raise RuntimeError(f"hidden-width scorer {self.name!r} collected no samples")
        squared = reduce_token_sum(self._squared_sum.clone(), self.groups.token_group)
        score = gather_scored_axis(squared.clamp_min(0).sqrt(), self._feature_group, dim=0)
        count = torch.tensor(
            self._sample_count,
            dtype=torch.int64,
            device=self._squared_sum.device,
        )
        reduce_token_sum(count, self.groups.token_group)
        if score.numel() != self.hidden_size:
            raise RuntimeError(
                f"hidden-width scorer reconstructed {score.numel()} channels, "
                f"expected {self.hidden_size}"
            )
        return {"score": score, "sample_count": int(count.item())}
