# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic packed-tensor surgery for Mamba2-style mixers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

__all__ = [
    "Mamba2TensorLayout",
    "mamba2_projected_prefix_mask",
    "slice_mamba2_state_dict",
    "sort_mamba2_state_dict",
]


@dataclass(frozen=True)
class Mamba2TensorLayout:
    """Logical parameter names and geometry for the common Mamba2 layout."""

    in_proj_key: str
    out_proj_key: str | None
    conv_weight_key: str | None
    conv_bias_key: str | None
    norm_key: str | None
    a_log_key: str | None
    d_key: str | None
    dt_bias_key: str | None
    num_heads: int
    head_dim: int
    num_groups: int
    state_dim: int
    in_proj_bias_key: str | None = None

    def __post_init__(self) -> None:
        for name in ("num_heads", "head_dim", "num_groups", "state_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.num_heads % self.num_groups:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by num_groups={self.num_groups}"
            )

    @property
    def inner_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def grouped_state_size(self) -> int:
        return self.num_groups * self.state_dim

    @property
    def in_proj_size(self) -> int:
        return 2 * self.inner_size + 2 * self.grouped_state_size + self.num_heads

    @property
    def conv_size(self) -> int:
        return self.inner_size + 2 * self.grouped_state_size

    @property
    def bias_key(self) -> str:
        if self.in_proj_bias_key is not None:
            return self.in_proj_bias_key
        if self.in_proj_key.endswith(".weight"):
            return self.in_proj_key[: -len(".weight")] + ".bias"
        return self.in_proj_key + ".bias"

    def validate(self, state_dict: Mapping[str, torch.Tensor], *, require_in_proj: bool = True) -> None:
        if require_in_proj and self.in_proj_key not in state_dict:
            raise KeyError(f"missing Mamba in projection: {self.in_proj_key}")
        in_proj = state_dict.get(self.in_proj_key)
        if in_proj is not None and in_proj.shape[0] != self.in_proj_size:
            raise ValueError(
                f"{self.in_proj_key} rows={in_proj.shape[0]} but geometry requires {self.in_proj_size}"
            )
        expected = {
            self.out_proj_key: (None, self.inner_size),
            self.conv_bias_key: (self.conv_size,),
            self.norm_key: (self.inner_size,),
            self.a_log_key: (self.num_heads,),
            self.d_key: (self.num_heads,),
            self.dt_bias_key: (self.num_heads,),
        }
        for key, shape in expected.items():
            if key is None or key not in state_dict:
                continue
            actual = state_dict[key].shape
            if len(shape) != len(actual):
                raise ValueError(f"{key} has shape {tuple(actual)}, expected rank {len(shape)}")
            if any(want is not None and got != want for got, want in zip(actual, shape)):
                raise ValueError(f"{key} has shape {tuple(actual)}, expected {shape}")
        if self.conv_weight_key in state_dict:
            actual = state_dict[self.conv_weight_key].shape
            if not actual or actual[0] != self.conv_size:
                raise ValueError(
                    f"{self.conv_weight_key} has shape {tuple(actual)}, expected first dimension "
                    f"{self.conv_size}"
                )


def _groupwise_prefix_indices(num_groups: int, width: int, target_width: int, *, device=None):
    if target_width <= 0 or target_width > width:
        raise ValueError(f"target group width must be in [1, {width}], got {target_width}")
    base = torch.arange(num_groups, device=device).unsqueeze(-1) * width
    return (base + torch.arange(target_width, device=device).unsqueeze(0)).reshape(-1)


def _keep_indices(
    layout: Mamba2TensorLayout,
    target_heads: int,
    target_head_dim: int,
    target_state_dim: int,
    *,
    device=None,
):
    if target_heads > layout.num_heads or target_heads % layout.num_groups:
        raise ValueError(
            f"target_heads={target_heads} must be <= {layout.num_heads} and divisible by "
            f"num_groups={layout.num_groups}"
        )
    if not 0 < target_head_dim <= layout.head_dim:
        raise ValueError(f"target_head_dim must be in [1, {layout.head_dim}], got {target_head_dim}")
    if not 0 < target_state_dim <= layout.state_dim:
        raise ValueError(f"target_state_dim must be in [1, {layout.state_dim}], got {target_state_dim}")
    heads_per_group = layout.num_heads // layout.num_groups
    target_heads_per_group = target_heads // layout.num_groups
    keep_heads = _groupwise_prefix_indices(
        layout.num_groups, heads_per_group, target_heads_per_group, device=device
    )
    keep_inner = (
        keep_heads[:, None] * layout.head_dim
        + torch.arange(target_head_dim, device=device)[None, :]
    ).reshape(-1)
    keep_state = _groupwise_prefix_indices(
        layout.num_groups, layout.state_dim, target_state_dim, device=device
    )
    return keep_heads, keep_inner, keep_state


def mamba2_projected_prefix_mask(
    layout: Mamba2TensorLayout,
    *,
    target_heads: int | None = None,
    target_head_dim: int | None = None,
    target_state_dim: int | None = None,
    device=None,
) -> torch.Tensor:
    """Mask for `[gate, x, B, C, dt]` emitted by a Mamba2 in projection."""

    target_heads = int(target_heads or layout.num_heads)
    target_head_dim = int(target_head_dim or layout.head_dim)
    target_state_dim = int(target_state_dim or layout.state_dim)
    keep_heads, keep_inner, keep_state = _keep_indices(
        layout, target_heads, target_head_dim, target_state_dim, device=device
    )
    mask = torch.zeros(layout.in_proj_size, dtype=torch.bool, device=device)
    inner, state = layout.inner_size, layout.grouped_state_size
    mask[keep_inner] = True
    mask[inner + keep_inner] = True
    mask[2 * inner + keep_state] = True
    mask[2 * inner + state + keep_state] = True
    mask[2 * inner + 2 * state + keep_heads] = True
    return mask


def _index_first(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return tensor[index.to(tensor.device)]


def _index_last(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return tensor[..., index.to(tensor.device)]


def slice_mamba2_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    layout: Mamba2TensorLayout,
    *,
    target_heads: int | None = None,
    target_head_dim: int | None = None,
    target_state_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return a shallow state-dict copy with Mamba logical prefixes sliced."""

    layout.validate(state_dict, require_in_proj=False)
    target_heads = int(target_heads or layout.num_heads)
    target_head_dim = int(target_head_dim or layout.head_dim)
    target_state_dim = int(target_state_dim or layout.state_dim)
    keep_heads, keep_inner, keep_state = _keep_indices(
        layout, target_heads, target_head_dim, target_state_dim
    )
    out = dict(state_dict)
    inner, state = layout.inner_size, layout.grouped_state_size
    packed_keep = torch.cat(
        [
            keep_inner,
            inner + keep_inner,
            2 * inner + keep_state,
            2 * inner + state + keep_state,
            2 * inner + 2 * state + keep_heads,
        ]
    )
    if layout.in_proj_key in state_dict:
        out[layout.in_proj_key] = _index_first(state_dict[layout.in_proj_key], packed_keep)
    if layout.bias_key in state_dict:
        out[layout.bias_key] = _index_first(state_dict[layout.bias_key], packed_keep)
    if layout.out_proj_key in state_dict:
        out[layout.out_proj_key] = _index_last(state_dict[layout.out_proj_key], keep_inner)
    for key in (layout.a_log_key, layout.d_key, layout.dt_bias_key):
        if key in state_dict:
            out[key] = _index_first(state_dict[key], keep_heads)
    if layout.norm_key in state_dict:
        out[layout.norm_key] = _index_first(state_dict[layout.norm_key], keep_inner)
    conv_keep = torch.cat([keep_inner, inner + keep_state, inner + state + keep_state])
    for key in (layout.conv_weight_key, layout.conv_bias_key):
        if key in state_dict:
            out[key] = _index_first(state_dict[key], conv_keep)
    return out


def _groupwise_order(scores: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    grouped = scores if scores.ndim == 2 else scores.reshape(groups, width)
    if tuple(grouped.shape) != (groups, width):
        raise ValueError(f"score shape {tuple(scores.shape)} is incompatible with {(groups, width)}")
    within = torch.argsort(grouped, dim=-1, descending=True)
    base = torch.arange(groups, device=within.device).unsqueeze(-1) * width
    return (base + within).reshape(-1)


def sort_mamba2_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    layout: Mamba2TensorLayout,
    *,
    head_scores: torch.Tensor | None = None,
    head_dim_scores: torch.Tensor | None = None,
    state_scores: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Sort Mamba heads, per-head dimensions, and grouped state channels."""

    layout.validate(state_dict, require_in_proj=False)
    out = dict(state_dict)
    orders: dict[str, torch.Tensor] = {}
    inner, state = layout.inner_size, layout.grouped_state_size

    head_order = None
    if head_scores is not None:
        head_order = _groupwise_order(
            head_scores, layout.num_groups, layout.num_heads // layout.num_groups
        )
        dim_grid = torch.arange(layout.head_dim, device=head_order.device)
        inner_order = (head_order[:, None] * layout.head_dim + dim_grid[None, :]).reshape(-1)
        packed_order = torch.cat(
            [
                inner_order,
                inner + inner_order,
                torch.arange(2 * state, device=head_order.device) + 2 * inner,
                2 * inner + 2 * state + head_order,
            ]
        )
        if layout.in_proj_key in out:
            out[layout.in_proj_key] = _index_first(out[layout.in_proj_key], packed_order)
        if layout.bias_key in out:
            out[layout.bias_key] = _index_first(out[layout.bias_key], packed_order)
        if layout.out_proj_key in out:
            out[layout.out_proj_key] = _index_last(out[layout.out_proj_key], inner_order)
        for key in (layout.a_log_key, layout.d_key, layout.dt_bias_key):
            if key in out:
                out[key] = _index_first(out[key], head_order)
        if layout.norm_key in out:
            out[layout.norm_key] = _index_first(out[layout.norm_key], inner_order)
        conv_order = torch.cat(
            [inner_order, inner + torch.arange(2 * state, device=head_order.device)]
        )
        for key in (layout.conv_weight_key, layout.conv_bias_key):
            if key in out:
                out[key] = _index_first(out[key], conv_order)
        orders["heads"] = head_order

    if head_dim_scores is not None:
        scores = head_dim_scores
        if tuple(scores.shape) != (layout.num_heads, layout.head_dim):
            raise ValueError(
                f"head_dim_scores must have shape {(layout.num_heads, layout.head_dim)}, "
                f"got {tuple(scores.shape)}"
            )
        if head_order is not None:
            scores = scores[head_order]
        within = torch.argsort(scores, dim=-1, descending=True)
        base = torch.arange(layout.num_heads, device=within.device).unsqueeze(-1) * layout.head_dim
        dim_order = (base + within).reshape(-1)
        packed_order = torch.cat(
            [
                dim_order,
                inner + dim_order,
                torch.arange(2 * state + layout.num_heads, device=within.device) + 2 * inner,
            ]
        )
        if layout.in_proj_key in out:
            out[layout.in_proj_key] = _index_first(out[layout.in_proj_key], packed_order)
        if layout.bias_key in out:
            out[layout.bias_key] = _index_first(out[layout.bias_key], packed_order)
        if layout.out_proj_key in out:
            out[layout.out_proj_key] = _index_last(out[layout.out_proj_key], dim_order)
        if layout.norm_key in out:
            out[layout.norm_key] = _index_first(out[layout.norm_key], dim_order)
        conv_order = torch.cat(
            [dim_order, inner + torch.arange(2 * state, device=within.device)]
        )
        for key in (layout.conv_weight_key, layout.conv_bias_key):
            if key in out:
                out[key] = _index_first(out[key], conv_order)
        orders["head_dim"] = dim_order

    if state_scores is not None:
        state_order = _groupwise_order(state_scores, layout.num_groups, layout.state_dim)
        packed_order = torch.cat(
            [
                torch.arange(2 * inner, device=state_order.device),
                2 * inner + state_order,
                2 * inner + state + state_order,
                torch.arange(layout.num_heads, device=state_order.device) + 2 * inner + 2 * state,
            ]
        )
        if layout.in_proj_key in out:
            out[layout.in_proj_key] = _index_first(out[layout.in_proj_key], packed_order)
        if layout.bias_key in out:
            out[layout.bias_key] = _index_first(out[layout.bias_key], packed_order)
        conv_order = torch.cat(
            [
                torch.arange(inner, device=state_order.device),
                inner + state_order,
                inner + state + state_order,
            ]
        )
        for key in (layout.conv_weight_key, layout.conv_bias_key):
            if key in out:
                out[key] = _index_first(out[key], conv_order)
        orders["state_dim"] = state_order
    return out, orders
