# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen GatedDeltaNet geometry and coupled permutation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "GDNPermutation",
    "GDNShape",
    "gated_delta_net_prefix_indices",
    "permute_gated_delta_net_state_dict",
    "slice_gated_delta_net_state_dict",
]


@dataclass(frozen=True)
class GDNShape:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int

    def __post_init__(self):
        values = (
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError(f"GatedDeltaNet dimensions must be positive, got {values}")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError(
                "GatedDeltaNet num_value_heads must be divisible by num_key_heads, got "
                f"{self.num_value_heads} and {self.num_key_heads}"
            )

    @property
    def value_heads_per_group(self) -> int:
        return self.num_value_heads // self.num_key_heads

    @classmethod
    def from_module(cls, module) -> "GDNShape":
        return cls(
            num_key_heads=int(module.num_k_heads),
            num_value_heads=int(module.num_v_heads),
            key_head_dim=int(module.head_k_dim),
            value_head_dim=int(module.head_v_dim),
        )

    @classmethod
    def from_mamba_config(cls, config) -> "GDNShape":
        return cls(
            num_key_heads=int(config.num_groups),
            num_value_heads=int(config.num_heads),
            key_head_dim=int(config.state_dim),
            value_head_dim=int(config.head_dim),
        )


@dataclass(frozen=True)
class GDNPermutation:
    key_groups: torch.Tensor
    value_lanes: torch.Tensor
    key_dim: torch.Tensor
    value_dim: torch.Tensor

    @classmethod
    def from_score_payload(cls, payload: dict, shape: GDNShape) -> "GDNPermutation":
        device = None

        def tensor(name, default):
            value = payload.get(name, default)
            return torch.as_tensor(value, dtype=torch.long, device=device)

        groups = tensor("key_group_order_most_important_first", torch.arange(shape.num_key_heads))
        lanes = tensor(
            "value_lane_order_most_important_first",
            torch.arange(shape.value_heads_per_group).expand(shape.num_key_heads, -1),
        )
        key_dim = tensor(
            "key_dim_order_most_important_first",
            torch.arange(shape.key_head_dim).expand(shape.num_key_heads, -1),
        )
        value_dim = tensor("value_dim_order_most_important_first", torch.arange(shape.value_head_dim))
        expected = {
            "key_groups": (shape.num_key_heads,),
            "value_lanes": (shape.num_key_heads, shape.value_heads_per_group),
            "key_dim": (shape.num_key_heads, shape.key_head_dim),
            "value_dim": (shape.value_head_dim,),
        }
        actual = {
            "key_groups": groups,
            "value_lanes": lanes,
            "key_dim": key_dim,
            "value_dim": value_dim,
        }
        for name, wanted in expected.items():
            if tuple(actual[name].shape) != wanted:
                raise ValueError(f"GDN {name} order shape {tuple(actual[name].shape)} != {wanted}")
        _require_permutation(groups, shape.num_key_heads, "key groups")
        _require_permutation(value_dim, shape.value_head_dim, "value dimensions")
        for group in range(shape.num_key_heads):
            _require_permutation(lanes[group], shape.value_heads_per_group, f"value lanes group {group}")
            _require_permutation(key_dim[group], shape.key_head_dim, f"key dimensions group {group}")
        return cls(groups, lanes, key_dim, value_dim)

    def indices(self, shape: GDNShape) -> dict[str, torch.Tensor]:
        key_rows = []
        value_heads = []
        for old_group_t in self.key_groups:
            old_group = int(old_group_t)
            key_rows.append(old_group * shape.key_head_dim + self.key_dim[old_group])
            value_heads.append(
                old_group * shape.value_heads_per_group + self.value_lanes[old_group]
            )
        qidx = torch.cat(key_rows)
        hidx = torch.cat(value_heads)
        vidx = (
            hidx[:, None] * shape.value_head_dim + self.value_dim[None, :]
        ).reshape(-1)
        key_width = shape.num_key_heads * shape.key_head_dim
        cidx = torch.cat((qidx, key_width + qidx, 2 * key_width + vidx))
        return {"qidx": qidx, "hidx": hidx, "vidx": vidx, "cidx": cidx}


def _require_permutation(order: torch.Tensor, size: int, name: str) -> None:
    if not torch.equal(torch.sort(order.cpu()).values, torch.arange(size)):
        raise ValueError(f"GDN {name} order is not a permutation of range({size}): {order.tolist()}")


def permute_gated_delta_net_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    shape: GDNShape,
    permutation: GDNPermutation,
) -> dict[str, torch.Tensor]:
    """Apply one function-preserving full-width Qwen GDN permutation in-place."""
    idx = permutation.indices(shape)
    row_bindings = {
        f"{prefix}.in_proj_qkv.weight": idx["cidx"],
        f"{prefix}.conv1d.weight": idx["cidx"],
        f"{prefix}.in_proj_z.weight": idx["vidx"],
        f"{prefix}.in_proj_a.weight": idx["hidx"],
        f"{prefix}.in_proj_b.weight": idx["hidx"],
        f"{prefix}.A_log": idx["hidx"],
        f"{prefix}.dt_bias": idx["hidx"],
        f"{prefix}.norm.weight": permutation.value_dim,
    }
    for key, order in row_bindings.items():
        if key in state_dict:
            state_dict[key] = state_dict[key].index_select(0, order.to(state_dict[key].device))
    out_key = f"{prefix}.out_proj.weight"
    if out_key in state_dict:
        state_dict[out_key] = state_dict[out_key].index_select(
            1, idx["vidx"].to(state_dict[out_key].device)
        )
    return idx


def gated_delta_net_prefix_indices(
    shape: GDNShape,
    target: GDNShape,
) -> dict[str, torch.Tensor]:
    """Return coupled full-width indices for one legal nested GDN prefix."""
    if target.num_key_heads > shape.num_key_heads:
        raise ValueError(
            f"target key groups {target.num_key_heads} exceed teacher {shape.num_key_heads}"
        )
    if target.value_heads_per_group > shape.value_heads_per_group:
        raise ValueError(
            "target value heads per group exceed the teacher: "
            f"{target.value_heads_per_group} > {shape.value_heads_per_group}"
        )
    if target.key_head_dim > shape.key_head_dim:
        raise ValueError(
            f"target key dim {target.key_head_dim} exceeds teacher {shape.key_head_dim}"
        )
    if target.value_head_dim > shape.value_head_dim:
        raise ValueError(
            f"target value dim {target.value_head_dim} exceeds teacher {shape.value_head_dim}"
        )

    qidx = _groupwise_prefix_indices(
        target.num_key_heads,
        shape.key_head_dim,
        target.key_head_dim,
    )
    full_ratio = shape.value_heads_per_group
    target_ratio = target.value_heads_per_group
    hidx = (
        torch.arange(target.num_key_heads).unsqueeze(-1) * full_ratio
        + torch.arange(target_ratio).unsqueeze(0)
    ).reshape(-1)
    vidx = (
        hidx[:, None] * shape.value_head_dim
        + torch.arange(target.value_head_dim)[None, :]
    ).reshape(-1)
    full_key_width = shape.num_key_heads * shape.key_head_dim
    cidx = torch.cat((qidx, full_key_width + qidx, 2 * full_key_width + vidx))
    return {"qidx": qidx, "hidx": hidx, "vidx": vidx, "cidx": cidx}


def _groupwise_prefix_indices(num_groups: int, width: int, target_width: int) -> torch.Tensor:
    return (
        torch.arange(num_groups).unsqueeze(-1) * width
        + torch.arange(target_width).unsqueeze(0)
    ).reshape(-1)


def slice_gated_delta_net_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    shape: GDNShape,
    target: GDNShape,
) -> dict[str, torch.Tensor]:
    """Physically slice every coupled Qwen GDN tensor to a nested target shape."""
    idx = gated_delta_net_prefix_indices(shape, target)
    row_bindings = {
        f"{prefix}.in_proj_qkv.weight": idx["cidx"],
        f"{prefix}.conv1d.weight": idx["cidx"],
        f"{prefix}.in_proj_z.weight": idx["vidx"],
        f"{prefix}.in_proj_a.weight": idx["hidx"],
        f"{prefix}.in_proj_b.weight": idx["hidx"],
        f"{prefix}.A_log": idx["hidx"],
        f"{prefix}.dt_bias": idx["hidx"],
        f"{prefix}.norm.weight": torch.arange(target.value_head_dim),
    }
    for key, keep in row_bindings.items():
        if key in state_dict:
            state_dict[key] = state_dict[key].index_select(0, keep.to(state_dict[key].device))
        bias_key = key[: -len(".weight")] + ".bias" if key.endswith(".weight") else None
        if bias_key is not None and bias_key in state_dict:
            state_dict[bias_key] = state_dict[bias_key].index_select(
                0, keep.to(state_dict[bias_key].device)
            )
    out_key = f"{prefix}.out_proj.weight"
    if out_key in state_dict:
        state_dict[out_key] = state_dict[out_key].index_select(
            1, idx["vidx"].to(state_dict[out_key].device)
        )
    return idx
