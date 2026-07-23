# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-owned residual hidden-width ranking, sorting, and slicing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import torch

__all__ = ["EmbeddingPruningSpec", "PackedMinitronImportance", "TensorAxisRule"]


@dataclass(frozen=True)
class TensorAxisRule:
    pattern: str
    axes: tuple[int, ...]
    description: str
    chunked_axes: tuple[tuple[int, int], ...] = ()
    grouped_axes: tuple[tuple[int, int], ...] = ()

    def matches(self, key: str) -> bool:
        return re.search(self.pattern, key) is not None


@dataclass(frozen=True)
class EmbeddingPruningSpec:
    """Complete model-family contract for the language residual width."""

    hidden_size: int
    legal_widths: tuple[int, ...]
    alignment: int
    tensor_rules: tuple[TensorAxisRule, ...]
    exempt_patterns: tuple[str, ...] = ()
    tie_groups: tuple[tuple[str, ...], ...] = ()
    config_paths: tuple[tuple[str, ...], ...] = (("hidden_size",),)
    residual_norm_patterns: tuple[str, ...] = ()
    permutation_group_size: int = 1

    def validate_width(self, width: int, *, tp_size: int = 1) -> int:
        width = int(width)
        if width not in self.legal_widths:
            raise ValueError(
                f"hidden width {width} is not legal; expected one of {self.legal_widths}"
            )
        if self.alignment > 1 and width % self.alignment:
            raise ValueError(
                f"hidden width {width} is not aligned to {self.alignment}"
            )
        if width % int(self.permutation_group_size):
            raise ValueError(
                f"hidden width {width} is not divisible by permutation_group_size="
                f"{self.permutation_group_size}"
            )
        if tp_size < 1 or width % int(tp_size):
            raise ValueError(f"hidden width {width} is not divisible by tp_size={tp_size}")
        return width

    def _rule(self, key: str) -> TensorAxisRule | None:
        matches = [rule for rule in self.tensor_rules if rule.matches(key)]
        if len(matches) > 1:
            raise ValueError(
                f"embedding pruning tensor {key!r} matches multiple axis rules: "
                + ", ".join(rule.description for rule in matches)
            )
        return matches[0] if matches else None

    def rule_for(self, key: str) -> TensorAxisRule | None:
        """Return the unique descriptor rule for a canonical tensor key."""
        return self._rule(key)

    def _is_exempt(self, key: str) -> bool:
        return any(re.search(pattern, key) is not None for pattern in self.exempt_patterns)

    def audit_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        handled: dict[str, dict[str, Any]] = {}
        exempt: list[str] = []
        unhandled: list[str] = []
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            rule = self._rule(key)
            if rule is not None:
                normalized_axes = []
                for raw_axis in rule.axes:
                    axis = raw_axis if raw_axis >= 0 else tensor.ndim + raw_axis
                    if not 0 <= axis < tensor.ndim:
                        raise ValueError(
                            f"axis {raw_axis} for {key!r} is invalid for shape {tuple(tensor.shape)}"
                        )
                    if tensor.shape[axis] != self.hidden_size:
                        raise ValueError(
                            f"axis {axis} for {key!r} has size {tensor.shape[axis]}, "
                            f"expected hidden_size={self.hidden_size}"
                        )
                    normalized_axes.append(axis)
                normalized_chunked_axes = []
                for raw_axis, chunks in rule.chunked_axes:
                    axis = raw_axis if raw_axis >= 0 else tensor.ndim + raw_axis
                    if not 0 <= axis < tensor.ndim:
                        raise ValueError(
                            f"chunked axis {raw_axis} for {key!r} is invalid for "
                            f"shape {tuple(tensor.shape)}"
                        )
                    expected = self.hidden_size * int(chunks)
                    if tensor.shape[axis] != expected:
                        raise ValueError(
                            f"chunked axis {axis} for {key!r} has size {tensor.shape[axis]}, "
                            f"expected {chunks} * hidden_size = {expected}"
                        )
                    normalized_chunked_axes.append((axis, int(chunks)))
                normalized_grouped_axes = []
                for raw_axis, group_size in rule.grouped_axes:
                    axis = raw_axis if raw_axis >= 0 else tensor.ndim + raw_axis
                    if not 0 <= axis < tensor.ndim:
                        raise ValueError(
                            f"grouped axis {raw_axis} for {key!r} is invalid for "
                            f"shape {tuple(tensor.shape)}"
                        )
                    group_size = int(group_size)
                    if group_size < 1 or self.hidden_size % group_size:
                        raise ValueError(
                            f"grouped axis for {key!r} has invalid group size {group_size}"
                        )
                    expected = self.hidden_size // group_size
                    if tensor.shape[axis] != expected:
                        raise ValueError(
                            f"grouped axis {axis} for {key!r} has size {tensor.shape[axis]}, "
                            f"expected hidden_size / {group_size} = {expected}"
                        )
                    normalized_grouped_axes.append((axis, group_size))
                handled[key] = {
                    "axes": normalized_axes,
                    "chunked_axes": normalized_chunked_axes,
                    "grouped_axes": normalized_grouped_axes,
                    "description": rule.description,
                    "shape": list(tensor.shape),
                }
                continue
            if self._is_exempt(key):
                exempt.append(key)
                continue
            if self.hidden_size in tensor.shape:
                unhandled.append(key)
        if unhandled:
            raise ValueError(
                "embedding pruning found hidden-sensitive tensors without descriptor rules: "
                + ", ".join(sorted(unhandled))
            )
        return {
            "hidden_size": self.hidden_size,
            "handled": handled,
            "exempt": sorted(exempt),
        }

    def _validate_order(self, order: torch.Tensor) -> torch.Tensor:
        order = order.detach().to(dtype=torch.long).reshape(-1)
        if order.numel() != self.hidden_size:
            raise ValueError(
                f"embedding permutation has {order.numel()} channels, expected {self.hidden_size}"
            )
        if not torch.equal(torch.sort(order.cpu()).values, torch.arange(self.hidden_size)):
            raise ValueError("embedding permutation must contain every hidden channel exactly once")
        return order

    def order_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Return a legal residual permutation for the descriptor's storage granularity.

        Block-quantized inputs cannot move individual channels across quantization
        blocks without dequantizing and requantizing.  Such descriptors rank whole
        blocks by summed channel importance and preserve channel order within a block.
        """
        scores = scores.detach().float().reshape(-1)
        if scores.numel() != self.hidden_size:
            raise ValueError(
                f"embedding scores have {scores.numel()} channels, expected {self.hidden_size}"
            )
        group_size = int(self.permutation_group_size)
        if group_size < 1 or self.hidden_size % group_size:
            raise ValueError(
                f"invalid permutation_group_size={group_size} for hidden_size={self.hidden_size}"
            )
        if group_size == 1:
            return torch.argsort(scores, descending=True)
        group_scores = scores.reshape(-1, group_size).sum(dim=1)
        group_order = torch.argsort(group_scores, descending=True)
        group_count = self.hidden_size // group_size
        cutoffs = sorted(
            {
                int(width) // group_size
                for width in self.legal_widths
                if 0 < int(width) < self.hidden_size
            }
        )
        if cutoffs:
            # Only membership at each legal prefix affects nested pruning.  Keep
            # original storage order inside each newly admitted tier so grouped
            # quantized models do not acquire avoidable full-width numerical
            # drift from an otherwise meaningless within-tier permutation.
            importance_rank = torch.empty_like(group_order)
            importance_rank[group_order] = torch.arange(
                group_count, device=group_order.device
            )
            tier_order = []
            previous = 0
            original_groups = torch.arange(group_count, device=group_order.device)
            for cutoff in (*cutoffs, group_count):
                tier = original_groups[
                    (importance_rank >= previous) & (importance_rank < cutoff)
                ]
                tier_order.append(tier)
                previous = cutoff
            group_order = torch.cat(tier_order)
        offsets = torch.arange(group_size, device=group_order.device)
        return (group_order[:, None] * group_size + offsets[None, :]).reshape(-1)

    def _group_order(self, order: torch.Tensor, group_size: int) -> torch.Tensor:
        group_size = int(group_size)
        if group_size != int(self.permutation_group_size):
            raise ValueError(
                f"tensor group size {group_size} does not match descriptor permutation "
                f"group size {self.permutation_group_size}"
            )
        rows = order.reshape(-1, group_size)
        groups = torch.div(rows[:, 0], group_size, rounding_mode="floor")
        expected = groups[:, None] * group_size + torch.arange(
            group_size, device=order.device
        )[None, :]
        if not torch.equal(rows, expected):
            raise ValueError("embedding permutation does not preserve quantization groups")
        return groups

    def _retie(self, state_dict: dict[str, torch.Tensor]) -> None:
        for group in self.tie_groups:
            present = [key for key in group if key in state_dict]
            if len(present) < 2:
                continue
            reference = state_dict[present[0]]
            for key in present[1:]:
                if tuple(state_dict[key].shape) != tuple(reference.shape):
                    raise ValueError(
                        f"cannot tie embedding tensors with different shapes: {present[0]} and {key}"
                    )
                if not torch.equal(state_dict[key], reference):
                    raise ValueError(
                        f"declared tie group contains different tensor values: {present[0]} and {key}"
                    )
                state_dict[key] = reference

    def permute_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        order: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.audit_state_dict(state_dict)
        order = self._validate_order(order)
        result: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            rule = self._rule(key)
            value = tensor
            if rule is not None:
                for raw_axis in rule.axes:
                    axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
                    value = value.index_select(axis, order.to(value.device))
                for raw_axis, chunks in rule.chunked_axes:
                    axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
                    chunk_order = torch.cat(
                        [order + chunk * self.hidden_size for chunk in range(int(chunks))]
                    )
                    value = value.index_select(axis, chunk_order.to(value.device))
                for raw_axis, group_size in rule.grouped_axes:
                    axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
                    group_order = self._group_order(order, group_size)
                    value = value.index_select(axis, group_order.to(value.device))
            result[key] = value
        self._retie(result)
        return result

    def slice_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        width: int,
        *,
        tp_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        width = self.validate_width(width, tp_size=tp_size)
        self.audit_state_dict(state_dict)
        result: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            rule = self._rule(key)
            value = tensor
            if rule is not None:
                for raw_axis in rule.axes:
                    axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
                    value = value.narrow(axis, 0, width)
                for raw_axis, chunks in rule.chunked_axes:
                    axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
                    value = torch.cat(
                        [
                            value.narrow(axis, chunk * self.hidden_size, width)
                            for chunk in range(int(chunks))
                        ],
                        dim=axis,
                    )
                for raw_axis, group_size in rule.grouped_axes:
                    axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
                    value = value.narrow(axis, 0, width // int(group_size))
            result[key] = value
        self._retie(result)
        return result

    def sliced_shape(self, key: str, shape: Sequence[int], width: int) -> tuple[int, ...]:
        """Return a tensor's exact shape after residual-width slicing."""

        width = self.validate_width(width)
        source_shape = tuple(int(dim) for dim in shape)
        rule = self._rule(key)
        if rule is None:
            if not self._is_exempt(key) and self.hidden_size in source_shape:
                raise ValueError(
                    "embedding pruning found hidden-sensitive tensor without a descriptor "
                    f"rule: {key!r}"
                )
            return source_shape

        target_shape = list(source_shape)
        target_sizes: dict[int, int] = {}

        def set_target(raw_axis: int, expected: int, target: int, description: str) -> None:
            axis = raw_axis if raw_axis >= 0 else len(source_shape) + raw_axis
            if not 0 <= axis < len(source_shape):
                raise ValueError(
                    f"{description} axis {raw_axis} for {key!r} is invalid for "
                    f"shape {source_shape}"
                )
            if source_shape[axis] != expected:
                raise ValueError(
                    f"{description} axis {axis} for {key!r} has size "
                    f"{source_shape[axis]}, expected {expected}"
                )
            if axis in target_sizes and target_sizes[axis] != target:
                raise ValueError(
                    f"embedding pruning tensor {key!r} defines conflicting target sizes "
                    f"for axis {axis}: {target_sizes[axis]} and {target}"
                )
            target_sizes[axis] = target

        for raw_axis in rule.axes:
            set_target(raw_axis, self.hidden_size, width, "residual")
        for raw_axis, chunks in rule.chunked_axes:
            chunks = int(chunks)
            set_target(
                raw_axis,
                self.hidden_size * chunks,
                width * chunks,
                "chunked residual",
            )
        for raw_axis, group_size in rule.grouped_axes:
            group_size = int(group_size)
            if group_size < 1 or self.hidden_size % group_size or width % group_size:
                raise ValueError(
                    f"grouped axis for {key!r} has invalid group size {group_size}"
                )
            set_target(
                raw_axis,
                self.hidden_size // group_size,
                width // group_size,
                "grouped residual",
            )
        for axis, target in target_sizes.items():
            target_shape[axis] = target
        return tuple(target_shape)

    def parameter_count(self, state_dict: Mapping[str, torch.Tensor]) -> int:
        tied_aliases: set[str] = set()
        for group in self.tie_groups:
            present = [key for key in group if key in state_dict]
            tied_aliases.update(present[1:])
        return sum(
            int(tensor.numel())
            for key, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor) and key not in tied_aliases
        )

    def update_config(self, config: Mapping[str, Any], width: int) -> dict[str, Any]:
        width = self.validate_width(width)
        updated = deepcopy(dict(config))
        for path in self.config_paths:
            if not path:
                continue
            node: dict[str, Any] = updated
            for key in path[:-1]:
                child = node.get(key)
                if not isinstance(child, dict):
                    raise ValueError(
                        f"embedding config path {'.'.join(path)} is missing mapping {key!r}"
                    )
                node = child
            if path[-1] not in node:
                raise ValueError(f"embedding config path {'.'.join(path)} is missing")
            node[path[-1]] = width
        return updated

    def update_config_object(self, config: Any, width: int) -> Any:
        """Return a deep-copied config with every descriptor-owned width updated.

        AutoModel/HF configuration trees mix ``PretrainedConfig`` objects and
        dictionaries.  Physical width realization must update both the nested
        language config and any top-level mirror without round-tripping through
        ``to_dict()``, which would lose the concrete config classes needed by
        model construction.
        """
        width = self.validate_width(width)
        updated = deepcopy(config)
        for path in self.config_paths:
            if not path:
                continue
            node = updated
            for key in path[:-1]:
                if isinstance(node, Mapping):
                    if key not in node:
                        raise ValueError(f"embedding config path {'.'.join(path)} is missing")
                    node = node[key]
                else:
                    if not hasattr(node, key):
                        raise ValueError(f"embedding config path {'.'.join(path)} is missing")
                    node = getattr(node, key)
            leaf = path[-1]
            if isinstance(node, Mapping):
                if leaf not in node:
                    raise ValueError(f"embedding config path {'.'.join(path)} is missing")
                node[leaf] = width
            else:
                if not hasattr(node, leaf):
                    raise ValueError(f"embedding config path {'.'.join(path)} is missing")
                setattr(node, leaf, width)
        return updated


class PackedMinitronImportance:
    """Exact Minitron hidden metric with original packed-sample boundaries."""

    def __init__(self, hidden_size: int):
        self.hidden_size = int(hidden_size)
        self._site_squared_sums: dict[str, torch.Tensor] = {}
        self._site_sample_counts: dict[str, int] = {}

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(self._site_squared_sums)

    @property
    def sample_count(self) -> int:
        return sum(self._site_sample_counts.values())

    def _sample_means(
        self,
        activations: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        token_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if activations.ndim != 3 or activations.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Minitron activations must be [batch, sequence, {self.hidden_size}], "
                f"got {tuple(activations.shape)}"
            )
        values = activations.detach().float().abs()
        mask = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if token_mask is None
            else token_mask.to(device=values.device, dtype=torch.bool)
        )
        if tuple(mask.shape) != tuple(values.shape[:2]):
            raise ValueError("Minitron token_mask must match activation batch/sequence dimensions")
        samples: list[torch.Tensor] = []
        if cu_seqlens is not None:
            if values.shape[0] != 1:
                raise ValueError("packed cu_seqlens require one packed row")
            cu = cu_seqlens.detach().cpu().to(torch.long).reshape(-1)
            for start, stop in zip(cu[:-1], cu[1:]):
                segment = values[0, int(start) : int(stop)]
                segment_mask = mask[0, int(start) : int(stop)]
                if bool(segment_mask.any()):
                    samples.append(segment[segment_mask].mean(dim=0))
        else:
            for row, row_mask in zip(values, mask):
                if bool(row_mask.any()):
                    samples.append(row[row_mask].mean(dim=0))
        return samples

    def update(
        self,
        site_name: str,
        activations: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        samples = self._sample_means(
            activations,
            cu_seqlens=cu_seqlens,
            token_mask=token_mask,
        )
        if not samples:
            return
        squared_sum = torch.stack(samples).square().sum(dim=0)
        if site_name in self._site_squared_sums:
            self._site_squared_sums[site_name] = (
                self._site_squared_sums[site_name] + squared_sum
            )
        else:
            self._site_squared_sums[site_name] = squared_sum
        self._site_sample_counts[site_name] = self._site_sample_counts.get(site_name, 0) + len(samples)

    def squared_sums(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._site_squared_sums.items()}

    def scores(self) -> torch.Tensor:
        if not self._site_squared_sums:
            raise RuntimeError("no Minitron embedding activations were collected")
        return torch.stack(
            [value.clamp_min(0).sqrt() for value in self._site_squared_sums.values()]
        ).sum(dim=0)
