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

"""Build a "sorted teacher": the teacher with each layer's FFN channels and attention heads
reordered by importance (from activation scoring).

Because the reorder is a consistent permutation of a contracted dim (see
:mod:`.attention_ffn_surgery`), the sorted teacher computes the **same outputs** as the teacher —
it is the same model, reindexed so that "most important first". Pruning any variant then becomes a
**prefix slice** (FFN ``[:K]``; attention = first ``kv`` groups x first ``m`` query heads), which is
what lets the block library, replace-1-block scoring, bypass, and final realization all operate on
this single artifact with no realized intermediate checkpoints.

This module is backend-agnostic (a weight-only transform): the scores come from
the unified AutoModel activation pass, and the per-layer key layout comes from
the descriptor.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from ..block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MLAConfig,
    MoEConfig,
)
from .attention_ffn_surgery import (
    ffn_permutation,
    grouped_attention_permutations,
    permute_query_rows_by_head,
)
from .gated_delta_net import GDNPermutation, GDNShape, permute_gated_delta_net_state_dict
from .latent_moe_surgery import (
    LatentMoETensorLayout,
    LatentMoETransform,
    apply_latent_moe_sort,
    apply_latent_moe_transform,
    compute_latent_moe_transform,
    reverse_latent_moe_transform,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LayerLayout",
    "iter_safetensor_weight_files",
    "sort_state_dict",
    "build_layer_layouts",
    "build_sorted_teacher",
]

_WEIGHT_SUFFIX = ".weight"


@dataclass
class LayerLayout:
    """Per-layer weight keys + head geometry the sorter needs (descriptor-derived)."""

    layer_idx: int
    head_dim: int
    # FFN (None when the layer has no prunable FFN — mamba / no-op).
    gate_key: str | None = None
    up_key: str | None = None
    down_key: str | None = None
    ffn_intermediate: int | None = None
    # Attention (None when the layer has no prunable attention — mamba / no-op).
    q_key: str | None = None
    k_key: str | None = None
    v_key: str | None = None
    o_key: str | None = None
    num_q_heads: int | None = None
    num_kv_heads: int | None = None
    q_gate_row_group: int | None = None
    q_head_aux_keys: tuple[str, ...] = ()
    # Multi-head latent attention (MLA).
    mla_prefix: str | None = None
    mla_q_a_key: str | None = None
    mla_q_norm_key: str | None = None
    mla_q_b_key: str | None = None
    mla_kv_a_key: str | None = None
    mla_kv_norm_key: str | None = None
    mla_kv_b_key: str | None = None
    mla_o_key: str | None = None
    mla_num_heads: int | None = None
    mla_q_lora_rank: int | None = None
    mla_kv_lora_rank: int | None = None
    # MoE.
    moe_prefix: str | None = None
    moe_gate_prefix: str | None = None
    moe_gate_key: str | None = None
    moe_gate_bias_key: str | None = None
    moe_router_aux_keys: tuple[str, ...] = ()
    moe_experts_prefix: str | None = None
    moe_expert_up_keys: list[str] | None = None
    moe_expert_down_keys: list[str] | None = None
    moe_fused_expert_keys: tuple[str, ...] = ()
    moe_fused_gate_up_keys: tuple[str, ...] = ()
    moe_fused_down_keys: tuple[str, ...] = ()
    moe_expert_intermediate_group_size: int = 1
    moe_expert_order_mode: str = "physical"
    moe_fused_gate_layout: str = "concatenated"
    moe_shared_gate_key: str | None = None
    moe_shared_up_key: str | None = None
    moe_shared_down_key: str | None = None
    moe_fc1_latent_key: str | None = None
    moe_fc2_latent_key: str | None = None
    moe_num_experts: int | None = None
    moe_expert_intermediate: int | None = None
    moe_shared_intermediate: int | None = None
    moe_latent_dim: int | None = None
    # Mamba.
    mamba_prefix: str | None = None
    mamba_in_key: str | None = None
    mamba_out_key: str | None = None
    mamba_conv_key: str | None = None
    mamba_conv_bias_key: str | None = None
    mamba_a_key: str | None = None
    mamba_d_key: str | None = None
    mamba_dt_bias_key: str | None = None
    mamba_norm_key: str | None = None
    mamba_num_heads: int | None = None
    mamba_head_dim: int | None = None
    mamba_num_groups: int | None = None
    mamba_state_dim: int | None = None
    gated_delta_net: bool = False


def _permute_rows_by_head(t: torch.Tensor, perm: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Permute a ``[num_heads*head_dim, ...]`` tensor (weight rows or bias) by head order."""
    lead = t.shape[0] // head_dim
    return t.view(lead, head_dim, *t.shape[1:])[perm].reshape(t.shape)


def _permute_cols_by_head(t: torch.Tensor, perm: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Permute a ``[..., num_heads*head_dim]`` tensor by head order along columns."""
    trailing = t.shape[-1] // head_dim
    return t.view(*t.shape[:-1], trailing, head_dim)[..., perm, :].reshape(t.shape)


def _bias_key(weight_key: str | None) -> str | None:
    return None if weight_key is None else weight_key.rsplit(".weight", 1)[0] + ".bias"


def _score_from_log(logs: dict[str, dict], key: str | None) -> torch.Tensor | None:
    if not key:
        return None
    log = logs.get(key)
    if isinstance(log, dict) and "score" in log:
        return log["score"]
    return None


def _identity_descending_scores(
    length: int,
    *,
    device: torch.device | None = None,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Scores whose descending argsort keeps the existing prefix order."""
    values = torch.arange(length, 0, -1, dtype=torch.float32, device=device)
    if shape is None:
        return values
    return values.reshape((1,) * (len(shape) - 1) + (length,)).expand(shape).clone()


def _permute_maybe_bias(sd, weight_key: str | None, perm: torch.Tensor, *, rows: bool) -> None:
    if weight_key not in sd:
        return
    sd[weight_key] = sd[weight_key][perm] if rows else sd[weight_key][:, perm]
    bkey = _bias_key(weight_key)
    if rows and bkey in sd:
        sd[bkey] = sd[bkey][perm]


def _clone_for_save(t: torch.Tensor) -> torch.Tensor:
    """Break storage aliases introduced by cross-key tensor moves before safetensors save."""
    return t.clone().contiguous()


def _gated_up_perm(up: torch.Tensor, perm: torch.Tensor, intermediate: int) -> torch.Tensor:
    if up.shape[0] == 2 * intermediate:
        return up[torch.cat([perm, perm + intermediate])]
    return up[perm]


def _grouped_ffn_permutation(score: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rank quantization groups while preserving channel order inside each group."""
    if group_size <= 1:
        return ffn_permutation(score)
    if score.numel() % group_size:
        raise ValueError(
            f"expert intermediate score length {score.numel()} is not divisible by "
            f"storage group size {group_size}"
        )
    group_scores = score.reshape(-1, group_size).sum(dim=1)
    group_order = torch.argsort(group_scores, descending=True)
    offsets = torch.arange(group_size, device=score.device)
    return (group_order[:, None] * group_size + offsets[None, :]).reshape(-1)


def _fused_gated_up_perm(
    up: torch.Tensor,
    perm: torch.Tensor,
    intermediate: int,
    layout: str,
) -> torch.Tensor:
    if layout == "concatenated":
        return _gated_up_perm(up, perm, intermediate)
    if layout == "interleaved":
        if up.shape[0] != 2 * intermediate:
            raise ValueError(
                f"interleaved gate/up tensor must have {2 * intermediate} rows, "
                f"got {tuple(up.shape)}"
            )
        pairs = torch.stack((2 * perm, 2 * perm + 1), dim=1).reshape(-1)
        return up[pairs]
    raise ValueError(f"unsupported fused gate layout {layout!r}")


def _slice_gated_up(up: torch.Tensor, keep: torch.Tensor, intermediate: int) -> torch.Tensor:
    if up.shape[0] == 2 * intermediate:
        return up[torch.cat([keep, keep + intermediate])]
    return up[keep]


def _head_dim_perm(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim == 1:
        return torch.argsort(scores, descending=True)
    per_head = torch.argsort(scores, dim=-1, descending=True)
    base = torch.arange(scores.shape[0], device=scores.device).unsqueeze(-1) * scores.shape[1]
    return (base + per_head).reshape(-1)


def _mamba_groupwise_head_perm(scores: torch.Tensor, num_heads: int, num_groups: int) -> torch.Tensor:
    if scores.ndim == 2:
        grouped = scores
    else:
        grouped = scores.view(num_groups, num_heads // num_groups)
    within = torch.argsort(grouped, dim=-1, descending=True)
    base = torch.arange(num_groups, device=within.device).unsqueeze(-1) * grouped.shape[1]
    return (base + within).reshape(-1)


def _groupwise_axis_perm(scores: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Sort a grouped axis independently inside every group and flatten it."""
    grouped = scores if scores.ndim == 2 else scores.view(num_groups, -1)
    within = torch.argsort(grouped, dim=-1, descending=True)
    base = torch.arange(num_groups, device=within.device).unsqueeze(-1) * grouped.shape[1]
    return (base + within).reshape(-1)


def _permute_mamba_inner_rows(t: torch.Tensor, perm: torch.Tensor, head_dim: int) -> torch.Tensor:
    return t.view(-1, head_dim, *t.shape[1:])[perm].reshape(t.shape)


def _permute_mamba_inner_cols(t: torch.Tensor, perm: torch.Tensor, head_dim: int) -> torch.Tensor:
    return t.view(*t.shape[:-1], -1, head_dim)[..., perm, :].reshape(t.shape)


def sort_state_dict(
    state_dict: dict[str, torch.Tensor],
    layouts: list[LayerLayout],
    ffn_scores: dict[int, torch.Tensor],
    attention_logs: dict[int, dict],
    score_logs: dict[str, dict] | None = None,
    original_tensor_loader: Callable[[str], torch.Tensor | None] | None = None,
    latent_transform_cache: dict[int, LatentMoETransform] | None = None,
    deferred_axes: frozenset[str] = frozenset(),
    mamba_state_score_key: str = "ssm_channel_contrib",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ``(sorted_state_dict, permutations)`` — the keystone, pure transform.

    ``ffn_scores[i]`` is the per-intermediate-channel importance ``[I]``.
    ``attention_logs[i]`` carries explicit grouped-attention scores:
    ``kv_group_scores [num_kv]`` and
    ``query_head_scores [num_kv, heads_per_group]``. Only keys present in
    ``state_dict`` are touched, so this is safe for heterogeneous (mamba / no-op)
    layers. ``permutations`` records the applied orders.
    """
    sd = dict(state_dict)
    perms: dict[str, torch.Tensor] = {}
    score_logs = score_logs or {}
    latent_transform_cache = latent_transform_cache if latent_transform_cache is not None else {}

    def original_tensor(key: str | None) -> torch.Tensor | None:
        if not key:
            return None
        if key in state_dict:
            return state_dict[key]
        if original_tensor_loader is None:
            return None
        return original_tensor_loader(key)

    for layout in layouts:
        # ---- FFN: permute intermediate channels (gate/up rows, down cols) ----
        if "ffn_intermediate" not in deferred_axes and layout.layer_idx in ffn_scores:
            perm = ffn_permutation(ffn_scores[layout.layer_idx])
            if layout.gate_key in sd:
                sd[layout.gate_key] = sd[layout.gate_key][perm]
            if layout.up_key in sd:
                sd[layout.up_key] = sd[layout.up_key][perm]
            if layout.down_key in sd:
                sd[layout.down_key] = sd[layout.down_key][:, perm]
            for wkey in (layout.gate_key, layout.up_key):
                bkey = _bias_key(wkey)
                if bkey in sd:
                    sd[bkey] = sd[bkey][perm]
            perms[f"ffn.{layout.layer_idx}"] = perm

        # ---- Attention: permute KV groups, then query heads within each group ----
        attn_log = attention_logs.get(layout.layer_idx)
        if (
            isinstance(attn_log, dict)
            and layout.num_kv_heads
            and layout.num_q_heads
            and not {"kv_groups", "q_heads_per_group"}.issubset(deferred_axes)
        ):
            kv_scores = attn_log.get("kv_group_scores")
            query_scores = attn_log.get("query_head_scores")
            if "kv_groups" in deferred_axes:
                kv_scores = None
            if "q_heads_per_group" in deferred_axes:
                query_scores = None
            if torch.is_tensor(kv_scores) or torch.is_tensor(query_scores):
                heads_per_group = layout.num_q_heads // layout.num_kv_heads
                device = None
                if torch.is_tensor(kv_scores):
                    device = kv_scores.device
                elif torch.is_tensor(query_scores):
                    device = query_scores.device
                if not torch.is_tensor(kv_scores):
                    kv_scores = _identity_descending_scores(
                        layout.num_kv_heads,
                        device=device,
                    )
                if not torch.is_tensor(query_scores):
                    query_scores = _identity_descending_scores(
                        heads_per_group,
                        device=device,
                        shape=(layout.num_kv_heads, heads_per_group),
                    )
                q_perm, kv_perm = grouped_attention_permutations(
                    kv_scores,
                    query_scores,
                    layout.num_kv_heads,
                    heads_per_group,
                )
            else:
                q_perm = kv_perm = None
        else:
            q_perm = kv_perm = None

        if q_perm is not None and kv_perm is not None:
            if layout.q_key in sd:
                sd[layout.q_key] = permute_query_rows_by_head(
                    sd[layout.q_key], q_perm, layout.head_dim, layout.num_q_heads
                )
            if layout.k_key in sd:
                sd[layout.k_key] = _permute_rows_by_head(sd[layout.k_key], kv_perm, layout.head_dim)
            if layout.v_key in sd:
                sd[layout.v_key] = _permute_rows_by_head(sd[layout.v_key], kv_perm, layout.head_dim)
            if layout.o_key in sd:
                sd[layout.o_key] = _permute_cols_by_head(sd[layout.o_key], q_perm, layout.head_dim)
            for key in layout.q_head_aux_keys:
                if key not in sd:
                    continue
                tensor = sd[key]
                if tensor.ndim < 1 or tensor.shape[0] != layout.num_q_heads:
                    raise ValueError(
                        f"query-head auxiliary tensor {key!r} must have first dimension "
                        f"num_q_heads={layout.num_q_heads}, got {tuple(tensor.shape)}"
                    )
                sd[key] = tensor.index_select(0, q_perm.to(tensor.device))
            for wkey, perm in (
                (layout.q_key, q_perm),
                (layout.k_key, kv_perm),
                (layout.v_key, kv_perm),
            ):
                bkey = _bias_key(wkey)
                if bkey in sd:
                    if wkey == layout.q_key:
                        sd[bkey] = permute_query_rows_by_head(
                            sd[bkey], perm, layout.head_dim, layout.num_q_heads
                        )
                    else:
                        sd[bkey] = _permute_rows_by_head(sd[bkey], perm, layout.head_dim)
            perms[f"attn.q.{layout.layer_idx}"] = q_perm
            perms[f"attn.kv.{layout.layer_idx}"] = kv_perm

        # ---- MLA: globally permute the two latent bases independently ----
        if layout.mla_prefix:
            head_log = score_logs.get(f"{layout.mla_prefix}.o_proj")
            head_score = (
                head_log.get("mla_head_scores", head_log.get("kv_group_scores"))
                if isinstance(head_log, dict)
                else None
            )
            if "mla_heads" not in deferred_axes and torch.is_tensor(head_score):
                num_heads = int(layout.mla_num_heads or 0)
                if head_score.numel() != num_heads:
                    raise ValueError(
                        f"MLA head score for layer {layout.layer_idx} has "
                        f"{head_score.numel()} entries, expected {num_heads}"
                    )
                head_perm = torch.argsort(head_score.reshape(-1), descending=True)
                for key in (layout.mla_q_b_key, layout.mla_kv_b_key):
                    if key in sd:
                        tensor = sd[key]
                        if tensor.shape[0] % num_heads:
                            raise ValueError(
                                f"{key} rows={tensor.shape[0]} are not divisible by "
                                f"MLA heads={num_heads}"
                            )
                        rows_per_head = tensor.shape[0] // num_heads
                        sd[key] = tensor.view(num_heads, rows_per_head, *tensor.shape[1:])[
                            head_perm
                        ].reshape_as(tensor)
                    bias_key = _bias_key(key)
                    if bias_key in sd:
                        bias = sd[bias_key]
                        rows_per_head = bias.shape[0] // num_heads
                        sd[bias_key] = bias.view(num_heads, rows_per_head)[head_perm].reshape_as(
                            bias
                        )
                if layout.mla_o_key in sd:
                    tensor = sd[layout.mla_o_key]
                    if tensor.shape[1] % num_heads:
                        raise ValueError(
                            f"{layout.mla_o_key} columns={tensor.shape[1]} are not divisible "
                            f"by MLA heads={num_heads}"
                        )
                    cols_per_head = tensor.shape[1] // num_heads
                    sd[layout.mla_o_key] = tensor.view(
                        tensor.shape[0], num_heads, cols_per_head
                    )[:, head_perm].reshape_as(tensor)
                perms[f"mla.heads.{layout.layer_idx}"] = head_perm

            q_log = score_logs.get(f"{layout.mla_prefix}.q_a_layernorm")
            q_score = q_log.get("q_lora_rank_score") if isinstance(q_log, dict) else None
            if "mla_q_lora_rank" not in deferred_axes and torch.is_tensor(q_score):
                q_perm = torch.argsort(q_score, descending=True)
                if layout.mla_q_a_key in sd:
                    sd[layout.mla_q_a_key] = sd[layout.mla_q_a_key][q_perm]
                q_a_bias = _bias_key(layout.mla_q_a_key)
                if q_a_bias in sd:
                    sd[q_a_bias] = sd[q_a_bias][q_perm]
                if layout.mla_q_norm_key in sd:
                    sd[layout.mla_q_norm_key] = sd[layout.mla_q_norm_key][q_perm]
                if layout.mla_q_b_key in sd:
                    sd[layout.mla_q_b_key] = sd[layout.mla_q_b_key][:, q_perm]
                perms[f"mla.q_lora.{layout.layer_idx}"] = q_perm

            kv_log = score_logs.get(f"{layout.mla_prefix}.kv_a_layernorm")
            kv_score = kv_log.get("kv_lora_rank_score") if isinstance(kv_log, dict) else None
            if "mla_kv_lora_rank" not in deferred_axes and torch.is_tensor(kv_score):
                kv_perm = torch.argsort(kv_score, descending=True)
                rank = int(layout.mla_kv_lora_rank or 0)
                if layout.mla_kv_a_key in sd:
                    tensor = sd[layout.mla_kv_a_key]
                    sd[layout.mla_kv_a_key] = torch.cat((tensor[:rank][kv_perm], tensor[rank:]), dim=0)
                kv_a_bias = _bias_key(layout.mla_kv_a_key)
                if kv_a_bias in sd:
                    tensor = sd[kv_a_bias]
                    sd[kv_a_bias] = torch.cat((tensor[:rank][kv_perm], tensor[rank:]), dim=0)
                if layout.mla_kv_norm_key in sd:
                    sd[layout.mla_kv_norm_key] = sd[layout.mla_kv_norm_key][kv_perm]
                if layout.mla_kv_b_key in sd:
                    sd[layout.mla_kv_b_key] = sd[layout.mla_kv_b_key][:, kv_perm]
                perms[f"mla.kv_lora.{layout.layer_idx}"] = kv_perm

        # ---- MoE intermediate channels, shared expert, expert order, latent basis ----
        if layout.moe_prefix:
            expert_perms: dict[int, torch.Tensor] = {}
            grouped_log = score_logs.get(layout.moe_experts_prefix or "")
            if (
                "moe_expert_intermediate" not in deferred_axes
                and isinstance(grouped_log, dict)
                and isinstance(grouped_log.get("expert_stats_dict"), dict)
            ):
                for expert, stat in grouped_log["expert_stats_dict"].items():
                    if isinstance(stat, dict) and "score" in stat:
                        expert_perms[int(expert)] = _grouped_ffn_permutation(
                            stat["score"], layout.moe_expert_intermediate_group_size
                        )
            if "moe_expert_intermediate" not in deferred_axes and layout.moe_expert_down_keys:
                for expert, down_key in enumerate(layout.moe_expert_down_keys):
                    score = _score_from_log(score_logs, down_key[: -len(_WEIGHT_SUFFIX)])
                    if score is not None:
                        expert_perms[expert] = _grouped_ffn_permutation(
                            score, layout.moe_expert_intermediate_group_size
                        )
            if (
                "moe_expert_intermediate" not in deferred_axes
                and layout.moe_fused_gate_up_keys
                and layout.moe_fused_down_keys
            ):
                intermediate = int(layout.moe_expert_intermediate or 0)
                group_size = int(layout.moe_expert_intermediate_group_size)
                for key in layout.moe_fused_gate_up_keys:
                    if key not in sd:
                        continue
                    tensor = sd[key]
                    rows = []
                    for expert in range(tensor.shape[0]):
                        perm = expert_perms.get(expert)
                        rows.append(
                            tensor[expert]
                            if perm is None
                            else _fused_gated_up_perm(
                                tensor[expert],
                                perm.to(tensor.device),
                                intermediate,
                                layout.moe_fused_gate_layout,
                            )
                        )
                    sd[key] = torch.stack(rows)
                for key in layout.moe_fused_down_keys:
                    if key not in sd:
                        continue
                    tensor = sd[key]
                    rows = []
                    for expert in range(tensor.shape[0]):
                        perm = expert_perms.get(expert)
                        if perm is None:
                            rows.append(tensor[expert])
                            continue
                        grouped = (
                            perm.reshape(-1, group_size)[:, 0] // group_size
                        ).to(tensor.device)
                        rows.append(tensor[expert].index_select(1, grouped))
                    sd[key] = torch.stack(rows)
                for expert, perm in expert_perms.items():
                    perms[f"moe.expert_intermediate.{layout.layer_idx}.{expert}"] = perm
            if (
                "moe_expert_intermediate" not in deferred_axes
                and layout.moe_expert_up_keys
                and layout.moe_expert_down_keys
            ):
                for expert, perm in expert_perms.items():
                    if expert >= len(layout.moe_expert_up_keys):
                        continue
                    up_key = layout.moe_expert_up_keys[expert]
                    down_key = layout.moe_expert_down_keys[expert]
                    if up_key in sd and down_key in sd:
                        sd[up_key] = _gated_up_perm(sd[up_key], perm, sd[down_key].shape[1])
                        sd[down_key] = sd[down_key][:, perm]
                        bkey = _bias_key(up_key)
                        if bkey in sd:
                            sd[bkey] = _gated_up_perm(sd[bkey], perm, sd[down_key].shape[1])
                        perms[f"moe.expert_intermediate.{layout.layer_idx}.{expert}"] = perm

            shared_score = _score_from_log(
                score_logs,
                layout.moe_shared_down_key[: -len(_WEIGHT_SUFFIX)] if layout.moe_shared_down_key else None,
            )
            if (
                "moe_shared_expert_intermediate" not in deferred_axes
                and shared_score is not None
                and layout.moe_shared_up_key in sd
                and layout.moe_shared_down_key in sd
            ):
                perm = ffn_permutation(shared_score)
                intermediate = int(sd[layout.moe_shared_down_key].shape[1])
                if layout.moe_shared_gate_key and layout.moe_shared_gate_key in sd:
                    sd[layout.moe_shared_gate_key] = sd[layout.moe_shared_gate_key][perm]
                    gate_bias = _bias_key(layout.moe_shared_gate_key)
                    if gate_bias in sd:
                        sd[gate_bias] = sd[gate_bias][perm]
                sd[layout.moe_shared_up_key] = _gated_up_perm(
                    sd[layout.moe_shared_up_key], perm, intermediate
                )
                up_bias = _bias_key(layout.moe_shared_up_key)
                if up_bias in sd:
                    sd[up_bias] = _gated_up_perm(sd[up_bias], perm, intermediate)
                sd[layout.moe_shared_down_key] = sd[layout.moe_shared_down_key][:, perm]
                perms[f"moe.shared_intermediate.{layout.layer_idx}"] = perm

            expert_score = _score_from_log(score_logs, layout.moe_gate_prefix)
            if expert_score is None:
                expert_score = _score_from_log(score_logs, layout.moe_prefix)
            if (
                "moe_experts" not in deferred_axes
                and
                expert_score is not None
                and layout.moe_num_experts
                and layout.moe_expert_order_mode == "physical"
                and (
                    layout.moe_fused_expert_keys
                    or (layout.moe_expert_up_keys and layout.moe_expert_down_keys)
                )
            ):
                perm = torch.argsort(expert_score, descending=True)
                if layout.moe_fused_expert_keys:
                    for key in layout.moe_fused_expert_keys:
                        if key not in sd:
                            continue
                        tensor = sd[key]
                        if tensor.ndim < 1 or tensor.shape[0] != layout.moe_num_experts:
                            raise ValueError(
                                f"fused expert tensor {key!r} must have first dimension "
                                f"num_experts={layout.moe_num_experts}, got {tuple(tensor.shape)}"
                            )
                        sd[key] = tensor.index_select(0, perm.to(tensor.device))
                else:
                    for new_idx, old_idx_t in enumerate(perm.tolist()):
                        if old_idx_t >= len(layout.moe_expert_up_keys):
                            continue
                        src_up_key = layout.moe_expert_up_keys[old_idx_t]
                        src_down_key = layout.moe_expert_down_keys[old_idx_t]
                        dst_up_key = layout.moe_expert_up_keys[new_idx]
                        dst_down_key = layout.moe_expert_down_keys[new_idx]
                        src_bias_key = _bias_key(src_up_key)
                        dst_bias_key = _bias_key(dst_up_key)
                        if (
                            dst_up_key not in sd
                            and dst_down_key not in sd
                            and dst_bias_key not in sd
                        ):
                            continue
                        src_up = original_tensor(src_up_key)
                        src_down = original_tensor(src_down_key)
                        src_perm = expert_perms.get(old_idx_t)
                        if dst_up_key in sd and src_up is not None:
                            if src_perm is not None and src_down is not None:
                                src_up = _gated_up_perm(src_up, src_perm, src_down.shape[1])
                            sd[dst_up_key] = _clone_for_save(src_up)
                        if dst_down_key in sd and src_down is not None:
                            if src_perm is not None:
                                src_down = src_down[:, src_perm]
                            sd[dst_down_key] = _clone_for_save(src_down)
                        if dst_bias_key in sd and src_bias_key:
                            src_bias = original_tensor(src_bias_key)
                            if src_bias is not None:
                                if src_perm is not None and src_down is not None:
                                    src_bias = _gated_up_perm(src_bias, src_perm, src_down.shape[1])
                                sd[dst_bias_key] = _clone_for_save(src_bias)
                if layout.moe_gate_key in sd:
                    sd[layout.moe_gate_key] = sd[layout.moe_gate_key][perm]
                if layout.moe_gate_bias_key in sd:
                    sd[layout.moe_gate_bias_key] = sd[layout.moe_gate_bias_key][perm]
                for key in layout.moe_router_aux_keys:
                    if key not in sd:
                        continue
                    tensor = sd[key]
                    if tensor.ndim < 1 or tensor.shape[0] != layout.moe_num_experts:
                        raise ValueError(
                            f"router auxiliary tensor {key!r} must have first dimension "
                            f"num_experts={layout.moe_num_experts}, got {tuple(tensor.shape)}"
                        )
                    sd[key] = tensor.index_select(0, perm.to(tensor.device))
                perms[f"moe.experts.{layout.layer_idx}"] = perm

            if (
                "moe_experts" not in deferred_axes
                and expert_score is not None
                and layout.moe_num_experts
                and layout.moe_expert_order_mode == "metadata_only"
            ):
                perms[f"moe.experts.{layout.layer_idx}"] = torch.argsort(
                    expert_score, descending=True
                )
            elif layout.moe_expert_order_mode not in {"physical", "metadata_only"}:
                raise ValueError(
                    f"unsupported moe_expert_order_mode={layout.moe_expert_order_mode!r}"
                )

            latent_log = score_logs.get(layout.moe_prefix or "")
            latent_keys = {
                key
                for key in (
                    layout.moe_fc1_latent_key,
                    layout.moe_fc2_latent_key,
                    *(layout.moe_expert_up_keys or ()),
                    *(layout.moe_expert_down_keys or ()),
                )
                if key
            }
            if (
                "moe_latent_dim" not in deferred_axes
                and layout.moe_latent_dim is not None
                and isinstance(latent_log, dict)
                and latent_keys.intersection(sd)
            ):
                if int(latent_log.get("format_version", 0)) != 5:
                    raise RuntimeError(
                        f"layer {layout.layer_idx} latent pruning requires the exact v5 "
                        "activation-aware sufficient statistics; obsolete covariance proxies "
                        "cannot be used"
                    )
                required_stats = ("latent_cov_in", "expert_weights_sum", "latent_cov_out")
                missing_stats = [name for name in required_stats if not torch.is_tensor(latent_log.get(name))]
                if missing_stats:
                    raise RuntimeError(
                        f"layer {layout.layer_idx} latent pruning requires proper-rotation statistics "
                        f"{required_stats}; missing {missing_stats}"
                    )
                if not layout.moe_fc1_latent_key or not layout.moe_fc2_latent_key:
                    raise RuntimeError(f"layer {layout.layer_idx} latent projections are not described")
                logical_layout = LatentMoETensorLayout(
                    fc1_key=layout.moe_fc1_latent_key,
                    fc2_key=layout.moe_fc2_latent_key,
                    expert_up_keys=tuple(layout.moe_expert_up_keys or ()),
                    expert_down_keys=tuple(layout.moe_expert_down_keys or ()),
                )
                transform = latent_transform_cache.get(layout.layer_idx)
                if transform is None:
                    transform = apply_latent_moe_sort(
                        sd,
                        logical_layout,
                        latent_cov_in=latent_log["latent_cov_in"],
                        expert_weights=latent_log["expert_weights_sum"],
                        latent_cov_out=latent_log["latent_cov_out"],
                        reverse=bool(latent_log.get("reverse_ranking", False)),
                        tensor_loader=original_tensor_loader,
                    )
                    latent_transform_cache[layout.layer_idx] = transform
                else:
                    apply_latent_moe_transform(sd, logical_layout, transform)
                perms[f"moe.latent_input_rotation.{layout.layer_idx}"] = (
                    transform.input_basis.to(torch.float32).cpu()
                )
                perms[f"moe.latent_output_basis.{layout.layer_idx}"] = (
                    transform.output_basis.to(torch.float32).cpu()
                )
                perms[f"moe.latent_output_compressor.{layout.layer_idx}"] = (
                    transform.output_compressor.to(torch.float32).cpu()
                )

        # ---- Qwen GatedDeltaNet: one coupled four-axis permutation ----
        if (
            layout.gated_delta_net
            and layout.mamba_prefix
            and {
                "gdn_key_groups",
                "gdn_value_heads_per_group",
                "gdn_key_head_dim",
                "gdn_value_head_dim",
            }.isdisjoint(deferred_axes)
        ):
            gdn_log = score_logs.get(layout.mamba_prefix)
            if isinstance(gdn_log, dict) and "key_group_order_most_important_first" in gdn_log:
                shape = GDNShape(
                    num_key_heads=layout.mamba_num_groups,
                    num_value_heads=layout.mamba_num_heads,
                    key_head_dim=layout.mamba_state_dim,
                    value_head_dim=layout.mamba_head_dim,
                )
                permutation = GDNPermutation.from_score_payload(gdn_log, shape)
                indices = permute_gated_delta_net_state_dict(
                    sd,
                    prefix=layout.mamba_prefix,
                    shape=shape,
                    permutation=permutation,
                )
                perms[f"gdn.key_groups.{layout.layer_idx}"] = permutation.key_groups
                perms[f"gdn.value_lanes.{layout.layer_idx}"] = permutation.value_lanes
                perms[f"gdn.key_dim.{layout.layer_idx}"] = permutation.key_dim
                perms[f"gdn.value_dim.{layout.layer_idx}"] = permutation.value_dim
                perms[f"gdn.qkv_rows.{layout.layer_idx}"] = indices["cidx"]

        # ---- Mamba: sort heads, head dims, and SSM state channels using in_proj scores ----
        if layout.mamba_prefix and layout.mamba_in_key and not layout.gated_delta_net:
            mamba_log = score_logs.get(layout.mamba_in_key[: -len(_WEIGHT_SUFFIX)])
            hperm_for_dim = None
            if (
                "mamba_heads" not in deferred_axes
                and isinstance(mamba_log, dict)
                and "mamba_head_scores" in mamba_log
            ):
                hperm = _mamba_groupwise_head_perm(
                    mamba_log["mamba_head_scores"],
                    layout.mamba_num_heads,
                    layout.mamba_num_groups,
                )
                hperm_for_dim = hperm
                hd = layout.mamba_head_dim
                inner = layout.mamba_num_heads * hd
                state = layout.mamba_num_groups * layout.mamba_state_dim
                dt_start = 2 * inner + 2 * state
                dt_end = dt_start + layout.mamba_num_heads
                if layout.mamba_in_key in sd:
                    w = sd[layout.mamba_in_key]
                    gate = _permute_mamba_inner_rows(w[:inner], hperm, hd)
                    x = _permute_mamba_inner_rows(w[inner : 2 * inner], hperm, hd)
                    # Nemotron Mamba packs in_proj as [gate, x, B, C, dt].
                    # The learned dt row is head-indexed just like A/D/dt_bias.
                    bc = w[2 * inner : dt_start]
                    dt = w[dt_start:dt_end][hperm]
                    sd[layout.mamba_in_key] = torch.cat([gate, x, bc, dt, w[dt_end:]], dim=0)
                    bkey = _bias_key(layout.mamba_in_key)
                    if bkey in sd:
                        b = sd[bkey]
                        sd[bkey] = torch.cat(
                            [
                                _permute_mamba_inner_rows(b[:inner], hperm, hd),
                                _permute_mamba_inner_rows(b[inner : 2 * inner], hperm, hd),
                                b[2 * inner : dt_start],
                                b[dt_start:dt_end][hperm],
                                b[dt_end:],
                            ],
                            dim=0,
                        )
                if layout.mamba_out_key in sd:
                    sd[layout.mamba_out_key] = _permute_mamba_inner_cols(sd[layout.mamba_out_key], hperm, hd)
                for key in (layout.mamba_a_key, layout.mamba_d_key, layout.mamba_dt_bias_key):
                    if key in sd:
                        sd[key] = sd[key][hperm]
                if layout.mamba_norm_key in sd:
                    sd[layout.mamba_norm_key] = _permute_mamba_inner_rows(sd[layout.mamba_norm_key], hperm, hd)
                if layout.mamba_conv_key in sd:
                    head_channels = (
                        hperm[:, None] * hd + torch.arange(hd, device=hperm.device)[None, :]
                    ).reshape(-1)
                    idx = torch.cat([head_channels, torch.arange(2 * state, device=hperm.device) + inner])
                    sd[layout.mamba_conv_key] = sd[layout.mamba_conv_key][idx]
                    if layout.mamba_conv_bias_key in sd:
                        sd[layout.mamba_conv_bias_key] = sd[layout.mamba_conv_bias_key][idx]
                perms[f"mamba.heads.{layout.layer_idx}"] = hperm

            if (
                "mamba_head_dim" not in deferred_axes
                and isinstance(mamba_log, dict)
                and "mamba_head_dim_scores" in mamba_log
            ):
                dim_scores = mamba_log["mamba_head_dim_scores"]
                if dim_scores.ndim == 1 and dim_scores.numel() == layout.mamba_head_dim:
                    # MiniTron defines one global within-head dimension order shared by every
                    # Mamba head.  Expand that D-wide order to the flattened H*D projection
                    # axis before permuting checkpoint tensors.  A legacy 1-D H*D payload is
                    # still handled by _head_dim_perm below.
                    within_head = _head_dim_perm(dim_scores)
                    head_base = (
                        torch.arange(layout.mamba_num_heads, device=within_head.device)
                        * layout.mamba_head_dim
                    )
                    dperm = (head_base[:, None] + within_head[None, :]).reshape(-1)
                else:
                    if hperm_for_dim is not None and dim_scores.ndim == 2:
                        dim_scores = dim_scores[hperm_for_dim]
                    dperm = _head_dim_perm(dim_scores)
                if dperm.numel() == layout.mamba_num_heads * layout.mamba_head_dim:
                    if layout.mamba_in_key in sd:
                        w = sd[layout.mamba_in_key]
                        inner = dperm.numel()
                        sd[layout.mamba_in_key] = torch.cat(
                            [w[:inner][dperm], w[inner : 2 * inner][dperm], w[2 * inner :]],
                            dim=0,
                        )
                        bkey = _bias_key(layout.mamba_in_key)
                        if bkey in sd:
                            b = sd[bkey]
                            sd[bkey] = torch.cat(
                                [b[:inner][dperm], b[inner : 2 * inner][dperm], b[2 * inner :]],
                                dim=0,
                            )
                    if layout.mamba_out_key in sd:
                        sd[layout.mamba_out_key] = sd[layout.mamba_out_key][:, dperm]
                    if layout.mamba_norm_key in sd:
                        sd[layout.mamba_norm_key] = sd[layout.mamba_norm_key][dperm]
                    if layout.mamba_conv_key in sd:
                        state = layout.mamba_num_groups * layout.mamba_state_dim
                        conv_idx = torch.cat(
                            [dperm, torch.arange(2 * state, device=dperm.device) + inner]
                        )
                        sd[layout.mamba_conv_key] = sd[layout.mamba_conv_key][conv_idx]
                        if layout.mamba_conv_bias_key in sd:
                            sd[layout.mamba_conv_bias_key] = sd[layout.mamba_conv_bias_key][conv_idx]
                    perms[f"mamba.head_dim.{layout.layer_idx}"] = dperm

            state_scores = None
            state_log = score_logs.get(
                layout.mamba_conv_key[: -len(_WEIGHT_SUFFIX)]
                if layout.mamba_conv_key
                else ""
            )
            if not isinstance(state_log, dict):
                # Backward compatibility for activation logs captured at in_proj.
                state_log = mamba_log
            if isinstance(state_log, dict):
                state_scores = state_log.get(mamba_state_score_key)
                if (
                    mamba_state_score_key != "ssm_channel_contrib"
                    and state_scores is None
                    and "mamba_state_dim" not in deferred_axes
                ):
                    raise KeyError(
                        "configured Mamba state score is missing from activation payload: "
                        f"layer={layout.layer_idx} key={mamba_state_score_key!r} "
                        f"available={sorted(state_log)}"
                    )
            if (
                "mamba_state_dim" not in deferred_axes
                and
                state_scores is not None
                and layout.mamba_num_groups
                and layout.mamba_state_dim
            ):
                sperm = _groupwise_axis_perm(state_scores, layout.mamba_num_groups)
                state = layout.mamba_num_groups * layout.mamba_state_dim
                inner = layout.mamba_num_heads * layout.mamba_head_dim
                if sperm.numel() == state:
                    if layout.mamba_in_key in sd:
                        w = sd[layout.mamba_in_key]
                        b = w[2 * inner : 2 * inner + state][sperm]
                        c = w[2 * inner + state : 2 * inner + 2 * state][sperm]
                        sd[layout.mamba_in_key] = torch.cat(
                            [w[: 2 * inner], b, c, w[2 * inner + 2 * state :]],
                            dim=0,
                        )
                        bkey = _bias_key(layout.mamba_in_key)
                        if bkey in sd:
                            bias = sd[bkey]
                            bb = bias[2 * inner : 2 * inner + state][sperm]
                            cb = bias[2 * inner + state : 2 * inner + 2 * state][sperm]
                            sd[bkey] = torch.cat(
                                [bias[: 2 * inner], bb, cb, bias[2 * inner + 2 * state :]],
                                dim=0,
                            )
                    if layout.mamba_conv_key in sd:
                        conv_idx = torch.cat(
                            [
                                torch.arange(inner, device=sperm.device),
                                inner + sperm,
                                inner + state + sperm,
                            ]
                        )
                        sd[layout.mamba_conv_key] = sd[layout.mamba_conv_key][conv_idx]
                        if layout.mamba_conv_bias_key in sd:
                            sd[layout.mamba_conv_bias_key] = sd[layout.mamba_conv_bias_key][conv_idx]
                    perms[f"mamba.state_dim.{layout.layer_idx}"] = sperm

    return sd, perms


def build_layer_layouts(
    block_configs,
    *,
    layer_prefix_tmpl: str,
    num_attention_heads: int,
    head_dim: int,
    ffn_subnames: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
    attn_subnames: tuple[str, str, str, str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    mlp_module: str = "mlp",
    attn_module: str = "self_attn",
    moe_module: str = "mlp",
    moe_router_subname: str = "gate",
    moe_router_aux_subnames: tuple[str, ...] = (),
    moe_fused_expert_subnames: tuple[str, ...] = (),
    moe_fused_gate_up_subnames: tuple[str, ...] = (),
    moe_fused_down_subnames: tuple[str, ...] = (),
    moe_expert_intermediate_group_size: int = 1,
    moe_expert_order_mode: str = "physical",
    moe_fused_gate_layout: str = "concatenated",
    moe_shared_expert_subname: str = "shared_experts",
    moe_shared_gate_subname: str | None = None,
    moe_shared_up_subname: str = "up_proj",
    moe_shared_down_subname: str = "down_proj",
    mamba_module: str = "self_attn",
    q_gate_row_group: int | None = None,
    attention_q_head_subnames: tuple[str, ...] = (),
    gated_delta_net: bool = False,
) -> list[LayerLayout]:
    """Build per-layer layouts from the (cast) block configs + a layer-prefix template.

    ``layer_prefix_tmpl`` is e.g. ``"model.language_model.layers.{i}"``. FFN/attention keys use the
    standard HF sub-names (overridable). A layer contributes FFN keys only if its FFN is prunable
    (not no-op) and attention keys only if its attention is a real GQA block (``num_kv_heads``
    set, i.e. not mamba / no-op).
    """
    gate, up, down = ffn_subnames
    qn, kn, vn, on = attn_subnames
    layouts = []
    for i, bc in enumerate(block_configs):
        if isinstance(bc, dict):
            bc = BlockConfig(**bc)
        prefix = layer_prefix_tmpl.format(i=i)
        layout = LayerLayout(layer_idx=i, head_dim=head_dim)
        ffn = bc.get_subblock("ffn")
        if ffn is not None and not isinstance(ffn, FFNConfig):
            raise TypeError(f"Expected FFNConfig for 'ffn', got {type(ffn).__name__}")
        if ffn is not None and not ffn.no_op:
            layout.gate_key = f"{prefix}.{mlp_module}.{gate}.weight"
            layout.up_key = f"{prefix}.{mlp_module}.{up}.weight"
            layout.down_key = f"{prefix}.{mlp_module}.{down}.weight"
            layout.ffn_intermediate = ffn.intermediate_size
        mamba = bc.get_subblock("mamba")
        if mamba is not None and not isinstance(mamba, MambaConfig):
            raise TypeError(f"Expected MambaConfig for 'mamba', got {type(mamba).__name__}")
        attn = bc.get_subblock("attention")
        if attn is not None and not isinstance(attn, AttentionConfig):
            raise TypeError(f"Expected AttentionConfig for 'attention', got {type(attn).__name__}")
        kv = attn.num_kv_heads if attn is not None else None
        if attn is not None and not attn.no_op and mamba is None and kv:
            layout.head_dim = attn.qk_head_dim or head_dim
            layout.q_key = f"{prefix}.{attn_module}.{qn}.weight"
            layout.k_key = f"{prefix}.{attn_module}.{kn}.weight"
            layout.v_key = f"{prefix}.{attn_module}.{vn}.weight"
            layout.o_key = f"{prefix}.{attn_module}.{on}.weight"
            layout.num_q_heads = attn.num_query_heads or num_attention_heads
            layout.num_kv_heads = kv
            layout.q_gate_row_group = q_gate_row_group
            layout.q_head_aux_keys = tuple(
                f"{prefix}.{attn_module}.{subname}"
                for subname in attention_q_head_subnames
            )
        mla = bc.get_subblock("mla")
        if mla is not None and not isinstance(mla, MLAConfig):
            raise TypeError(f"Expected MLAConfig for 'mla', got {type(mla).__name__}")
        if mla is not None and not mla.no_op:
            aprefix = f"{prefix}.{attn_module}"
            layout.mla_prefix = aprefix
            layout.mla_q_a_key = f"{aprefix}.q_a_proj.weight"
            layout.mla_q_norm_key = f"{aprefix}.q_a_layernorm.weight"
            layout.mla_q_b_key = f"{aprefix}.q_b_proj.weight"
            layout.mla_kv_a_key = f"{aprefix}.kv_a_proj_with_mqa.weight"
            layout.mla_kv_norm_key = f"{aprefix}.kv_a_layernorm.weight"
            layout.mla_kv_b_key = f"{aprefix}.kv_b_proj.weight"
            layout.mla_o_key = f"{aprefix}.o_proj.weight"
            layout.mla_num_heads = mla.num_heads or num_attention_heads
            layout.mla_q_lora_rank = mla.q_lora_rank
            layout.mla_kv_lora_rank = mla.kv_lora_rank
        moe = bc.get_subblock("moe")
        if moe is not None and not isinstance(moe, MoEConfig):
            raise TypeError(f"Expected MoEConfig for 'moe', got {type(moe).__name__}")
        if moe is not None and not moe.no_op:
            mprefix = f"{prefix}.{moe_module}"
            layout.moe_prefix = mprefix
            layout.moe_gate_prefix = f"{mprefix}.{moe_router_subname}"
            layout.moe_gate_key = f"{layout.moe_gate_prefix}.weight"
            layout.moe_gate_bias_key = f"{layout.moe_gate_prefix}.e_score_correction_bias"
            layout.moe_router_aux_keys = tuple(
                f"{mprefix}.{subname}" for subname in moe_router_aux_subnames
            )
            layout.moe_experts_prefix = f"{mprefix}.experts"
            layout.moe_fused_expert_keys = tuple(
                f"{mprefix}.{subname}" for subname in moe_fused_expert_subnames
            )
            layout.moe_fused_gate_up_keys = tuple(
                f"{mprefix}.{subname}" for subname in moe_fused_gate_up_subnames
            )
            layout.moe_fused_down_keys = tuple(
                f"{mprefix}.{subname}" for subname in moe_fused_down_subnames
            )
            layout.moe_expert_intermediate_group_size = int(
                moe_expert_intermediate_group_size
            )
            layout.moe_expert_order_mode = str(moe_expert_order_mode)
            layout.moe_fused_gate_layout = str(moe_fused_gate_layout)
            layout.moe_num_experts = moe.num_experts
            layout.moe_expert_intermediate = moe.expert_intermediate_size
            layout.moe_shared_intermediate = moe.shared_expert_intermediate_size
            layout.moe_latent_dim = moe.latent_dim
            if moe.num_experts:
                layout.moe_expert_up_keys = [
                    f"{mprefix}.experts.{e}.up_proj.weight" for e in range(moe.num_experts)
                ]
                layout.moe_expert_down_keys = [
                    f"{mprefix}.experts.{e}.down_proj.weight" for e in range(moe.num_experts)
                ]
            shared_prefix = f"{mprefix}.{moe_shared_expert_subname}"
            if moe_shared_gate_subname:
                layout.moe_shared_gate_key = (
                    f"{shared_prefix}.{moe_shared_gate_subname}.weight"
                )
            layout.moe_shared_up_key = f"{shared_prefix}.{moe_shared_up_subname}.weight"
            layout.moe_shared_down_key = (
                f"{shared_prefix}.{moe_shared_down_subname}.weight"
            )
            if moe.latent_dim is not None:
                layout.moe_fc1_latent_key = f"{mprefix}.fc1_latent_proj.weight"
                layout.moe_fc2_latent_key = f"{mprefix}.fc2_latent_proj.weight"
        if mamba is not None and not mamba.no_op:
            mprefix = f"{prefix}.{mamba_module}"
            layout.mamba_prefix = mprefix
            layout.mamba_in_key = f"{mprefix}.in_proj.weight"
            layout.mamba_out_key = f"{mprefix}.out_proj.weight"
            layout.mamba_conv_key = f"{mprefix}.conv1d.weight"
            layout.mamba_conv_bias_key = f"{mprefix}.conv1d.bias"
            layout.mamba_a_key = f"{mprefix}.A_log"
            layout.mamba_d_key = f"{mprefix}.D"
            layout.mamba_dt_bias_key = f"{mprefix}.dt_bias"
            layout.mamba_norm_key = f"{mprefix}.norm.weight"
            layout.mamba_num_heads = mamba.num_heads
            layout.mamba_head_dim = mamba.head_dim
            layout.mamba_num_groups = mamba.num_groups
            layout.mamba_state_dim = mamba.state_dim
            layout.gated_delta_net = bool(gated_delta_net)
            if layout.gated_delta_net:
                layout.mamba_in_key = f"{mprefix}.in_proj_qkv.weight"
                layout.mamba_d_key = None
        layouts.append(layout)
    return layouts


def _check_activation_passes_manifest(activations_log_dir: Path) -> set[str] | None:
    """Warn when the manifest written by the scoring stage is inconsistent with what rglob finds.

    A manifest records the exact pass names at completion time.  If a user later re-runs
    activation scoring with a *different* set of passes (e.g. adds an attention pass after an
    FFN-only run but forgets to clean the old activations dir), the manifest will mismatch the
    actual subdirs — the old pass files are stale but still present, so rglob silently merges
    them into the sorted teacher.  The warning surfaces this before it becomes a silent quality
    regression.
    """
    manifest_path = activations_log_dir / "activation_passes_manifest.json"
    if not manifest_path.exists():
        return None  # single-pass run or pre-manifest activations dir; nothing to check
    recorded_passes = json.loads(manifest_path.read_text()).get("passes", [])
    # Check that every recorded pass subdir still has rank files, and that there are no
    # extra subdirs with rank files not listed in the manifest.
    found_subdirs = {
        p.parent.name
        for p in activations_log_dir.rglob("rank_*.pth")
        if p.parent != activations_log_dir  # exclude flat single-pass files
    }
    recorded_set = set(recorded_passes)
    missing = recorded_set - found_subdirs
    extra = found_subdirs - recorded_set
    if missing or extra:
        logger.warning(
            "activation_passes_manifest inconsistency in %s: manifest lists %s, "
            "but subdirs with rank files are %s (missing=%s, extra=%s).  "
            "Only manifest-listed pass dirs will be loaded; re-run activation if "
            "a required pass is missing.",
            activations_log_dir,
            sorted(recorded_passes),
            sorted(found_subdirs),
            sorted(missing),
            sorted(extra),
        )
    return recorded_set


def _score_log_aliases(module_name: str) -> tuple[str, ...]:
    """Return equivalent module names across native AutoModel and checkpoint layouts."""
    aliases = [module_name]
    prefix_pairs = (
        ("model.layers.", "backbone.layers."),
        ("backbone.layers.", "model.layers."),
    )
    for src, dst in prefix_pairs:
        if module_name.startswith(src):
            aliases.append(dst + module_name[len(src) :])
    return tuple(dict.fromkeys(aliases))


def _merge_score_log(logs: dict[str, dict], module_name: str, log: dict, pass_name: str) -> None:
    if module_name in logs:
        existing = logs[module_name]
        if "cov_out_stats_dict" in log:
            existing.setdefault("cov_out_stats_dict", {}).update(
                log.get("cov_out_stats_dict", {})
            )
        # Multiple activation passes can legitimately target the same
        # canonical module (for example Mamba head/head-dim and SSM-state
        # scoring both hook ``in_proj``). Preserve method-specific fields
        # and keep the first generic ``score`` unless a later shard only
        # carries that field.
        for key, value in log.items():
            if key == "cov_out_stats_dict":
                continue
            if key == "score" and key in existing:
                existing.setdefault("score_aliases", {})[pass_name] = value
            else:
                existing[key] = value
        return
    logs[module_name] = dict(log)


def _load_score_logs(activations_log_dir: str | Path) -> dict[str, dict]:
    """Merge the per-module ``{"score": ...}`` entries from every ``rank_*.pth`` shard.

    Uses ``rglob`` so that a multi-pass scoring run — which writes each pass (e.g. FFN
    ``down_proj`` scores and attention ``o_proj`` scores) to its own subdir under a common parent
    ``activations_log_dir`` — is merged into one score map. The keys are distinct module names
    across passes (FFN vs attention), so there is no collision; the single-dir case is unchanged.
    """
    activations_log_dir = Path(activations_log_dir)
    manifest_passes = _check_activation_passes_manifest(activations_log_dir)
    logs: dict[str, dict] = {}
    for p in sorted(activations_log_dir.rglob("rank_*.pth")):
        if manifest_passes is not None and p.parent != activations_log_dir and p.parent.name not in manifest_passes:
            continue
        pass_name = p.parent.name
        for module_name, log in torch.load(p, map_location="cpu").items():
            if not isinstance(log, dict):
                continue
            for alias in _score_log_aliases(module_name):
                _merge_score_log(logs, alias, log, pass_name)
    return logs


def _load_scores(activations_log_dir: str | Path) -> dict[str, torch.Tensor]:
    return {
        module_name: log["score"]
        for module_name, log in _load_score_logs(activations_log_dir).items()
        if isinstance(log, dict) and "score" in log
    }


def iter_safetensor_weight_files(checkpoint_dir: str | Path) -> tuple[Path, ...]:
    """Return model safetensor files relative to a standard HF checkpoint directory.

    Puzzletron canonical checkpoints are standard HuggingFace artifacts.
    """
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return tuple(sorted({Path(filename) for filename in index["weight_map"].values()}))

    single_file = checkpoint_dir / "model.safetensors"
    if single_file.exists():
        return (Path(single_file.name),)

    raise FileNotFoundError(
        f"No safetensors checkpoint found under {checkpoint_dir}. "
        "Puzzletron requires standard HuggingFace safetensors artifacts."
    )


def _build_original_tensor_loader(checkpoint_dir: Path):
    """Return a key-based tensor loader for cross-shard checkpoint surgery."""
    from safetensors import safe_open
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        contexts: dict[str, object] = {}
        handles: dict[str, object] = {}

        def load_from_index(key: str) -> torch.Tensor | None:
            shard = weight_map.get(key)
            if shard is None:
                return None
            handle = handles.get(shard)
            if handle is None:
                context = safe_open(
                    str(checkpoint_dir / shard),
                    framework="pt",
                    device="cpu",
                )
                handle = context.__enter__()
                contexts[shard] = context
                handles[shard] = handle
            if key not in handle.keys():
                return None
            return handle.get_tensor(key)

        def close_index() -> None:
            for shard, context in list(contexts.items()):
                context.__exit__(None, None, None)
                contexts.pop(shard, None)
                handles.pop(shard, None)

        load_from_index.close = close_index  # type: ignore[attr-defined]

        return load_from_index

    single_file = checkpoint_dir / "model.safetensors"
    if single_file.exists():
        context = safe_open(str(single_file), framework="pt", device="cpu")
        handle = context.__enter__()

        def load_from_single(key: str) -> torch.Tensor | None:
            if key not in handle.keys():
                return None
            return handle.get_tensor(key)

        def close_single() -> None:
            context.__exit__(None, None, None)

        load_from_single.close = close_single  # type: ignore[attr-defined]

        return load_from_single

    return None


def _copy_checkpoint_without_weight_files(
    source_dir: Path,
    output_dir: Path,
    weight_files: tuple[Path, ...],
) -> None:
    """Copy checkpoint metadata/tokenizer files while leaving model weights to be rewritten."""
    weight_file_set = set(weight_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.rglob("*"):
        rel = item.relative_to(source_dir)
        dst = output_dir / rel
        if item.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if rel in weight_file_set:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)


def _sort_distributed_info() -> tuple[int, int, int]:
    import torch.distributed as torch_dist

    if torch_dist.is_available() and torch_dist.is_initialized():
        return (
            torch_dist.get_rank(),
            torch_dist.get_world_size(),
            int(os.environ.get("LOCAL_RANK", torch_dist.get_rank())),
        )
    return 0, 1, 0


def _latent_transform_payload(transform: LatentMoETransform) -> dict[str, torch.Tensor]:
    return {
        "input_basis": transform.input_basis.cpu(),
        "output_basis": transform.output_basis.cpu(),
        "output_compressor": transform.output_compressor.cpu(),
        "transformed_fc2": transform.transformed_fc2.cpu(),
    }


def _latent_transform_from_payload(
    payload: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> LatentMoETransform:
    return LatentMoETransform(
        input_basis=payload["input_basis"].to(device=device),
        output_basis=payload["output_basis"].to(device=device),
        output_compressor=payload["output_compressor"].to(device=device),
        transformed_fc2=payload["transformed_fc2"].to(device=device),
    )


def _checkpoint_tensor_categories(key: str) -> tuple[str, ...]:
    lowered = key.lower()
    categories: list[str] = []
    if any(token in lowered for token in ("embed_tokens", "embeddings", "embedding.weight")):
        categories.append("embedding")
    if "lm_head" in lowered:
        categories.append("lm_head")
    if any(token in lowered for token in ("mtp", "multi_token")):
        categories.append("mtp")
    if any(token in lowered for token in ("vision", "visual", "vit")):
        categories.append("vision")
    if any(
        token in lowered
        for token in (".experts.", ".shared_experts.", ".shared_expert.", ".gate.")
    ):
        categories.append("moe")
    if "latent" in lowered:
        categories.append("latent_moe")
    return tuple(categories)


def _precompute_distributed_latent_transforms(
    *,
    layouts: list[LayerLayout],
    score_logs: dict[str, dict],
    original_tensor_loader: Callable[[str], torch.Tensor | None],
    work_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    deferred_axes: frozenset[str],
) -> dict[int, LatentMoETransform]:
    """Compute each latent transform once, distributed by layer, then share it."""

    import torch.distributed as torch_dist

    if "moe_latent_dim" in deferred_axes:
        return {}
    rank_path = work_dir / f"latent_transforms_rank_{rank}.pt"
    if not rank_path.is_file():
        local_payload: dict[int, dict[str, torch.Tensor]] = {}
        latent_layouts = [
            layout
            for layout in layouts
            if layout.moe_prefix
            and layout.moe_fc1_latent_key
            and layout.moe_fc2_latent_key
            and layout.moe_expert_up_keys
        ]
        for latent_index, layout in enumerate(latent_layouts):
            if latent_index % world_size != rank:
                continue
            latent_log = score_logs.get(layout.moe_prefix or "")
            if not isinstance(latent_log, dict):
                continue
            if int(latent_log.get("format_version", 0)) != 5:
                continue
            required_stats = ("latent_cov_in", "expert_weights_sum", "latent_cov_out")
            if not all(torch.is_tensor(latent_log.get(name)) for name in required_stats):
                continue
            fc1 = original_tensor_loader(layout.moe_fc1_latent_key)
            fc2 = original_tensor_loader(layout.moe_fc2_latent_key)
            expert_ups = [
                original_tensor_loader(key) for key in (layout.moe_expert_up_keys or ())
            ]
            if fc1 is None or fc2 is None or any(value is None for value in expert_ups):
                raise RuntimeError(
                    f"rank {rank} could not load latent tensors for layer {layout.layer_idx}"
                )
            print(
                "[sorted_teacher] latent transform start "
                f"rank={rank} layer={layout.layer_idx} device={device}",
                flush=True,
            )
            transform = compute_latent_moe_transform(
                fc1,
                fc2,
                expert_ups,  # type: ignore[arg-type]
                latent_log["expert_weights_sum"],
                latent_log["latent_cov_in"],
                latent_log["latent_cov_out"],
                compute_device=device,
            )
            if bool(latent_log.get("reverse_ranking", False)):
                transform = reverse_latent_moe_transform(transform)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            local_payload[int(layout.layer_idx)] = _latent_transform_payload(transform)
            print(
                "[sorted_teacher] latent transform complete "
                f"rank={rank} layer={layout.layer_idx}",
                flush=True,
            )
            del transform, expert_ups, fc1, fc2
            if device.type == "cuda":
                torch.cuda.empty_cache()
        tmp_path = rank_path.with_suffix(".tmp")
        torch.save(local_payload, tmp_path)
        os.replace(tmp_path, rank_path)

    if world_size > 1:
        torch_dist.barrier()
    merged: dict[int, LatentMoETransform] = {}
    for owner_rank in range(world_size):
        owner_path = work_dir / f"latent_transforms_rank_{owner_rank}.pt"
        payload = torch.load(owner_path, map_location="cpu", weights_only=True)
        for layer_idx, transform_payload in payload.items():
            layer_idx = int(layer_idx)
            if layer_idx in merged:
                raise RuntimeError(f"duplicate latent transform for layer {layer_idx}")
            merged[layer_idx] = _latent_transform_from_payload(
                transform_payload,
                device=device,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return merged


def build_sorted_teacher(
    teacher_dir: str | Path,
    activations_log_dir: str | Path,
    output_dir: str | Path,
    descriptor,
    *,
    ffn_subnames: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
    attn_subnames: tuple[str, str, str, str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    mlp_module: str = "mlp",
    attn_module: str = "self_attn",
    deferred_axes: tuple[str, ...] = (),
    mamba_state_score_key: str = "ssm_channel_contrib",
    embedding_widths: Sequence[int] | None = None,
) -> Path:
    """Write the sorted teacher to ``output_dir`` and return it.

    Copies every non-weight file of the AnyModel checkpoint, then rewrites the
    standard HuggingFace safetensors shards with FFN/attention tensors permuted
    by importance (keys/shapes/index unchanged, so the index stays valid and the
    model is functionally identical). Also writes ``sorted_permutations.json``.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from ..block_config import maybe_cast_block_configs
    from ..tools.checkpoint_utils import load_model_config

    teacher_dir, output_dir = Path(teacher_dir), Path(output_dir)
    deferred_axes_set = frozenset(str(axis) for axis in deferred_axes)
    config = load_model_config(
        teacher_dir, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    block_configs = maybe_cast_block_configs(config.block_configs)
    lm = descriptor.get_language_model_config(config)
    num_q = lm.num_attention_heads
    head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // num_q)
    # Per-layer prefix template, e.g. "model.language_model.layers.{i}".
    layer_prefix_tmpl = descriptor.layer_block_name(0).rsplit(".", 1)[0] + ".{i}"

    layout_kwargs = {
        "ffn_subnames": ffn_subnames,
        "attn_subnames": attn_subnames,
        "mlp_module": mlp_module,
        "attn_module": attn_module,
    }
    if hasattr(descriptor, "sorted_teacher_layout_kwargs"):
        layout_kwargs.update(descriptor.sorted_teacher_layout_kwargs(lm))
    layouts = build_layer_layouts(
        block_configs,
        layer_prefix_tmpl=layer_prefix_tmpl,
        num_attention_heads=num_q,
        head_dim=head_dim,
        **layout_kwargs,
    )

    score_logs = _load_score_logs(activations_log_dir)
    scores = {k: v["score"] for k, v in score_logs.items() if isinstance(v, dict) and "score" in v}
    embedding_site_scores = [
        value["score"].float()
        for value in score_logs.values()
        if isinstance(value, dict)
        and "sample_count" in value
        and torch.is_tensor(value.get("score"))
        and value["score"].numel() == int(lm.hidden_size)
    ]
    embedding_order = None
    embedding_spec = None
    if embedding_site_scores:
        embedding_scores = torch.stack(embedding_site_scores).sum(dim=0)
        embedding_spec = descriptor.embedding_pruning_spec(
            config,
            widths=tuple(
                int(width)
                for width in (embedding_widths or (int(lm.hidden_size),))
            ),
            alignment=1,
        )
        embedding_order = embedding_spec.order_from_scores(embedding_scores)
        logger.info(
            "Sorted teacher: global hidden-width ranking from %d residual norm sites",
            len(embedding_site_scores),
        )
    ple_spec = descriptor.ple_pruning_spec(config)
    ple_order = None
    if ple_spec is not None and "ple_width" not in deferred_axes_set:
        ple_score_keys = [
            ple_spec.layer_score_key(layer_idx)
            for layer_idx in range(ple_spec.num_layers)
        ]
        if any(key in score_logs for key in ple_score_keys):
            ple_order = ple_spec.order_from_score_logs(score_logs)
            logger.info(
                "Sorted teacher: global PLE ranking aggregated from %d layers",
                ple_spec.num_layers,
            )
    ffn_scores, attention_logs = {}, {}
    for layout in layouts:
        if layout.down_key and layout.down_key[: -len(_WEIGHT_SUFFIX)] in scores:
            ffn_scores[layout.layer_idx] = scores[layout.down_key[: -len(_WEIGHT_SUFFIX)]]
        if layout.o_key:
            attn_log = score_logs.get(layout.o_key[: -len(_WEIGHT_SUFFIX)])
            if (
                isinstance(attn_log, dict)
                and (
                    torch.is_tensor(attn_log.get("kv_group_scores"))
                    or torch.is_tensor(attn_log.get("query_head_scores"))
                )
            ):
                attention_logs[layout.layer_idx] = attn_log
    logger.info(
        "Sorted teacher: %d layers, FFN sorted for %d, attention sorted for %d",
        len(layouts),
        len(ffn_scores),
        len(attention_logs),
    )

    rank, world_size, local_rank = _sort_distributed_info()
    use_cuda = torch.cuda.is_available() and (
        world_size > 1 or os.environ.get("PUZZLETRON_SORT_USE_CUDA", "0") == "1"
    )
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    import torch.distributed as torch_dist

    weight_files = iter_safetensor_weight_files(teacher_dir)
    work_dir = output_dir / ".parallel_sort_work"
    if rank == 0:
        _copy_checkpoint_without_weight_files(teacher_dir, output_dir, weight_files)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "plan.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "world_size": world_size,
                    "weight_files": [str(path) for path in weight_files],
                    "use_cuda": use_cuda,
                    "deferred_axes": sorted(deferred_axes_set),
                    "mamba_state_score_key": mamba_state_score_key,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if world_size > 1:
        torch_dist.barrier()

    original_tensor_loader = _build_original_tensor_loader(teacher_dir)
    if original_tensor_loader is None:
        raise RuntimeError(f"could not construct original tensor loader for {teacher_dir}")
    latent_transform_cache = _precompute_distributed_latent_transforms(
        layouts=layouts,
        score_logs=score_logs,
        original_tensor_loader=original_tensor_loader,
        work_dir=work_dir,
        rank=rank,
        world_size=world_size,
        device=device,
        deferred_axes=deferred_axes_set,
    )

    def original_tensor_on_device(key: str) -> torch.Tensor | None:
        value = original_tensor_loader(key)
        return None if value is None else value.to(device=device)

    marker_dir = work_dir / "shard_markers"
    metadata_dir = work_dir / "shard_permutations"
    marker_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assigned_files = list(weight_files[rank::world_size])
    for shard_index, rel_path in enumerate(assigned_files, start=1):
        safe_name = str(rel_path).replace("/", "__")
        marker_path = marker_dir / f"{safe_name}.done"
        shard_perms_path = metadata_dir / f"{safe_name}.pt"
        src_shard = teacher_dir / rel_path
        dst_shard = output_dir / rel_path
        if marker_path.is_file() and shard_perms_path.is_file() and dst_shard.is_file():
            print(
                "[sorted_teacher] resume shard "
                f"rank={rank} shard={rel_path} ({shard_index}/{len(assigned_files)})",
                flush=True,
            )
            continue

        print(
            "[sorted_teacher] shard start "
            f"rank={rank} device={device} shard={rel_path} "
            f"({shard_index}/{len(assigned_files)})",
            flush=True,
        )
        cpu_tensors = load_file(str(src_shard), device="cpu")
        tensors = {
            key: value.to(device=device, non_blocking=False)
            for key, value in cpu_tensors.items()
        }
        del cpu_tensors
        with safe_open(str(src_shard), framework="pt") as f:
            metadata = f.metadata()
        sorted_tensors, perms = sort_state_dict(
            tensors,
            layouts,
            ffn_scores,
            attention_logs,
            score_logs,
            original_tensor_loader=original_tensor_on_device,
            latent_transform_cache=latent_transform_cache,
            deferred_axes=deferred_axes_set,
            mamba_state_score_key=mamba_state_score_key,
        )
        embedding_handled = set()
        if embedding_order is not None:
            embedding_audit = embedding_spec.audit_state_dict(tensors)
            embedding_handled = set(embedding_audit["handled"])
            sorted_tensors = embedding_spec.permute_state_dict(
                sorted_tensors,
                embedding_order.to(device=device),
            )
            perms["embedding.hidden_order"] = embedding_order.to(device=device)
        if ple_order is not None:
            sorted_tensors, ple_handled = ple_spec.permute_state_dict(
                sorted_tensors,
                ple_order.to(device=device),
            )
            embedding_handled.update(ple_handled)
            perms["ple.global_order"] = ple_order.to(device=device)
        input_signature = {
            key: (tuple(value.shape), str(value.dtype)) for key, value in tensors.items()
        }
        output_signature = {
            key: (tuple(value.shape), str(value.dtype)) for key, value in sorted_tensors.items()
        }
        if output_signature != input_signature:
            raise RuntimeError(
                f"sorted shard changed tensor structure: rank={rank} shard={rel_path}"
            )
        inventory: dict[str, int] = {
            "all": len(input_signature),
            "embedding": 0,
            "lm_head": 0,
            "mtp": 0,
            "vision": 0,
            "moe": 0,
            "latent_moe": 0,
        }
        protected_categories = {"embedding", "lm_head", "mtp", "vision"}
        if "moe_latent_dim" in deferred_axes_set:
            protected_categories.add("latent_moe")
        for key, source_tensor in tensors.items():
            categories = _checkpoint_tensor_categories(key)
            for category in categories:
                inventory[category] += 1
            if (
                protected_categories.intersection(categories)
                and key not in embedding_handled
                and not torch.equal(
                source_tensor,
                sorted_tensors[key],
                )
            ):
                raise RuntimeError(
                    "sorted teacher modified a protected tensor: "
                    f"rank={rank} shard={rel_path} key={key} categories={categories}"
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        save_tensors = {
            key: value.detach().to(device="cpu").contiguous()
            for key, value in sorted_tensors.items()
        }
        dst_shard.parent.mkdir(parents=True, exist_ok=True)
        tmp_shard = dst_shard.with_suffix(dst_shard.suffix + f".rank{rank}.tmp")
        save_file(save_tensors, str(tmp_shard), metadata=metadata)
        os.replace(tmp_shard, dst_shard)

        lightweight_perms = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in perms.items()
            if not key.startswith("moe.latent_")
        }
        tmp_perms = shard_perms_path.with_suffix(".tmp")
        torch.save(
            {
                "permutations": lightweight_perms,
                "inventory": inventory,
                "structure_verified": True,
            },
            tmp_perms,
        )
        os.replace(tmp_perms, shard_perms_path)
        marker_path.write_text(json.dumps({"rank": rank, "shard": str(rel_path)}) + "\n")
        print(
            "[sorted_teacher] shard complete "
            f"rank={rank} shard={rel_path} ({shard_index}/{len(assigned_files)})",
            flush=True,
        )
        del tensors, sorted_tensors, save_tensors, perms
        if device.type == "cuda":
            torch.cuda.empty_cache()

    close_loader = getattr(original_tensor_loader, "close", None)
    if callable(close_loader):
        close_loader()
    if world_size > 1:
        torch_dist.barrier()
    if rank == 0:
        all_perms: dict[str, list[int] | dict] = {}
        rotation_sidecar: dict[str, torch.Tensor] = {}
        tensor_inventory = {
            "all": 0,
            "embedding": 0,
            "lm_head": 0,
            "mtp": 0,
            "vision": 0,
            "moe": 0,
            "latent_moe": 0,
        }
        for rel_path in weight_files:
            safe_name = str(rel_path).replace("/", "__")
            shard_perms_path = metadata_dir / f"{safe_name}.pt"
            if not shard_perms_path.is_file():
                raise RuntimeError(f"missing shard permutation metadata: {shard_perms_path}")
            shard_metadata = torch.load(
                shard_perms_path,
                map_location="cpu",
                weights_only=True,
            )
            if shard_metadata.get("structure_verified") is not True:
                raise RuntimeError(f"shard structure was not verified: {rel_path}")
            for category, count in shard_metadata["inventory"].items():
                tensor_inventory[category] += int(count)
            perms = shard_metadata["permutations"]
            for key, value in perms.items():
                if key in all_perms:
                    continue
                if torch.is_tensor(value) and value.ndim > 1:
                    rotation_sidecar[key] = value.cpu()
                    all_perms[key] = {
                        "sidecar": "sorted_rotations.pt",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                else:
                    all_perms[key] = value.tolist() if torch.is_tensor(value) else value

        for layer_idx, transform in latent_transform_cache.items():
            latent_values = {
                f"moe.latent_input_rotation.{layer_idx}": transform.input_basis,
                f"moe.latent_output_basis.{layer_idx}": transform.output_basis,
                f"moe.latent_output_compressor.{layer_idx}": transform.output_compressor,
            }
            for key, value in latent_values.items():
                value = value.to(dtype=torch.float32, device="cpu")
                rotation_sidecar[key] = value
                all_perms[key] = {
                    "sidecar": "sorted_rotations.pt",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }

        (output_dir / "sorted_permutations.json").write_text(json.dumps(all_perms))
        if rotation_sidecar:
            torch.save(rotation_sidecar, output_dir / "sorted_rotations.pt")
        (output_dir / "parallel_sort_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "world_size": world_size,
                    "use_cuda": use_cuda,
                    "weight_files": len(weight_files),
                    "latent_transforms": len(latent_transform_cache),
                    "deferred_axes": sorted(deferred_axes_set),
                    "mamba_state_score_key": mamba_state_score_key,
                    "tensor_inventory": tensor_inventory,
                    "protected_tensor_equality_verified": True,
                    "status": "complete",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        shutil.rmtree(work_dir)
        logger.info("Wrote sorted teacher to %s", output_dir)
    if world_size > 1:
        torch_dist.barrier()
    return output_dir
