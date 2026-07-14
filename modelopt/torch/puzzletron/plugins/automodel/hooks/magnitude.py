"""Descriptor-owned mean-|activation| fallback for otherwise unscored axes."""

from __future__ import annotations

from typing import Any

import torch

from ..reduction import gather_scored_axis, reduce_token_sum, to_local_with_feature_group
from .base import ScoringHook
from .samplewise import flatten_sample_tokens

__all__ = ["ActivationMagnitudeScorer"]


def _select_tensor(args: tuple[Any, ...], output: Any, selector: str) -> torch.Tensor:
    parts = selector.split(".")
    root = parts.pop(0)
    value = args if root == "input" else output
    for part in parts:
        if isinstance(value, (tuple, list)):
            try:
                value = value[int(part)]
            except (ValueError, IndexError) as error:
                raise ValueError(f"tensor selector {selector!r} has invalid tuple index {part!r}") from error
        elif isinstance(value, dict):
            if part not in value:
                raise ValueError(f"tensor selector {selector!r} is missing mapping key {part!r}")
            value = value[part]
        else:
            if not hasattr(value, part):
                raise ValueError(f"tensor selector {selector!r} is missing attribute {part!r}")
            value = getattr(value, part)
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            value = value[0]
        else:
            raise ValueError(
                f"tensor selector {selector!r} is ambiguous for a {type(value).__name__} "
                f"with {len(value)} values; select an explicit index"
            )
    if not torch.is_tensor(value):
        raise ValueError(
            f"tensor selector {selector!r} resolved to {type(value).__name__}, not a tensor"
        )
    return value


class ActivationMagnitudeScorer(ScoringHook):
    """Sum per-original-sample mean absolute activation along one declared axis."""

    method = "magnitude_fallback"
    checkpoint_tensor_names = ("_score_sum",)
    checkpoint_scalar_names = (
        "_sample_count",
        "_token_count",
        "_local_scored_size",
        "_feature_sharded",
    )

    def __init__(
        self,
        module,
        groups,
        *,
        tensor_selector: str,
        scored_dim: int,
        output_field: str,
        expected_size: int,
        target_type: str,
        block_idx: int | None = None,
        name: str | None = None,
    ):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        if tensor_selector.split(".", 1)[0] not in {"input", "output"}:
            raise ValueError("magnitude tensor_selector must start with input or output")
        if not output_field:
            raise ValueError("magnitude output_field must not be empty")
        if int(expected_size) < 1:
            raise ValueError("magnitude expected_size must be positive")
        self.tensor_selector = tensor_selector
        self.scored_dim = int(scored_dim)
        self.output_field = output_field
        self.expected_size = int(expected_size)
        self.target_type = target_type
        self._score_sum: torch.Tensor | None = None
        self._sample_count = 0
        self._token_count = 0
        self._local_scored_size: int | None = None
        self._feature_sharded = False
        self._feature_group = None
        self._sequence_ids: torch.Tensor | None = None
        self._sequence_cursor = 0
        self._num_samples = 0
        self._restored_from_checkpoint = False

    def load_checkpoint_state(self, state: dict) -> None:
        super().load_checkpoint_state(state)
        self._restored_from_checkpoint = True

    def _restore_feature_layout(self) -> None:
        if not self._restored_from_checkpoint:
            return
        self._feature_group = self.groups.tp_group if self._feature_sharded else None
        if self._score_sum is None:
            local_size = int(self._local_scored_size or self.expected_size)
            try:
                device = next(self.module.parameters()).device
            except StopIteration:
                device = None
            self._score_sum = torch.zeros(local_size, dtype=torch.float32, device=device)
        self._restored_from_checkpoint = False

    def set_batch_metadata(self, *, sequence_ids: torch.Tensor, num_samples: int) -> None:
        if sequence_ids.ndim != 2:
            raise ValueError("magnitude sequence_ids must be [batch, sequence]")
        if int(num_samples) < 1:
            raise ValueError("magnitude num_samples must be positive")
        self._sequence_ids = sequence_ids
        self._sequence_cursor = 0
        self._num_samples = int(num_samples)

    def __call__(self, module, args, output):
        del module
        if self._sequence_ids is None:
            raise RuntimeError("magnitude scorer batch metadata was not set before forward")
        selected = _select_tensor(args, output, self.tensor_selector)
        scored_dim = self.scored_dim % selected.ndim
        activations, feature_group = to_local_with_feature_group(
            selected, feature_dim=scored_dim
        )
        self._feature_group = feature_group
        self._feature_sharded = feature_group is not None
        local_scored_size = int(activations.shape[scored_dim])
        self._local_scored_size = local_scored_size

        per_token, flat_ids, self._sequence_cursor = flatten_sample_tokens(
            activations.detach().float().abs(),
            scored_dim=scored_dim,
            sequence_ids=self._sequence_ids,
            sequence_cursor=self._sequence_cursor,
        )
        other_features = int(per_token.shape[1])
        per_token = per_token.double().sum(dim=1)
        flat_ids = flat_ids.to(device=activations.device, dtype=torch.long)
        valid = (flat_ids >= 0) & (flat_ids < self._num_samples)
        sample_sums = torch.zeros(
            self._num_samples,
            local_scored_size,
            dtype=torch.float64,
            device=activations.device,
        )
        element_counts = torch.zeros(
            self._num_samples, dtype=torch.float64, device=activations.device
        )
        token_counts = torch.zeros(
            self._num_samples, dtype=torch.int64, device=activations.device
        )
        if bool(valid.any()):
            valid_ids = flat_ids[valid]
            sample_sums.index_add_(0, valid_ids, per_token[valid])
            element_counts.index_add_(
                0,
                valid_ids,
                torch.full_like(valid_ids, other_features, dtype=torch.float64),
            )
            token_counts.index_add_(
                0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64)
            )
        reduce_token_sum(sample_sums, self.groups.cp_group)
        reduce_token_sum(element_counts, self.groups.cp_group)
        reduce_token_sum(token_counts, self.groups.cp_group)

        if self._score_sum is None:
            self._score_sum = torch.zeros(
                local_scored_size, dtype=torch.float32, device=activations.device
            )
        if self.groups.cp_rank == 0:
            present = element_counts > 0
            if bool(present.any()):
                sample_means = sample_sums[present] / element_counts[present].unsqueeze(1)
                self._score_sum += sample_means.sum(dim=0).float()
                self._sample_count += int(present.sum().item())
                self._token_count += int(token_counts[present].sum().item())

    def finalize(self) -> dict:
        self._restore_feature_layout()
        if self._score_sum is None:
            raise RuntimeError(f"magnitude scorer {self.name!r} collected no samples")
        local = reduce_token_sum(self._score_sum.clone(), self.groups.token_group)
        score = gather_scored_axis(local, self._feature_group, dim=0)
        counters = torch.tensor(
            [self._sample_count, self._token_count],
            dtype=torch.int64,
            device=self._score_sum.device,
        )
        reduce_token_sum(counters, self.groups.token_group)
        if score.numel() != self.expected_size:
            raise RuntimeError(
                f"magnitude scorer reconstructed {score.numel()} values, "
                f"expected {self.expected_size}"
            )
        return {
            self.output_field: score,
            "magnitude_metadata": {
                self.output_field: {
                    "metric_kind": "magnitude_fallback",
                    "target_path": self.name,
                    "tensor_selector": self.tensor_selector,
                    "scored_dim": self.scored_dim,
                    "sample_count": int(counters[0].item()),
                    "token_count": int(counters[1].item()),
                    "expected_size": self.expected_size,
                    "topology": {
                        "tp": self.groups.tp_size,
                        "cp": self.groups.cp_size,
                        "ep": self.groups.ep_size,
                        "pp": self.groups.pp_size,
                        "token": self.groups.token_size,
                    },
                }
            },
        }
