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

"""Model-agnostic surgery primitives for FFN-channel and attention-head pruning.

These are **pure tensor functions** — no model, descriptor, or parallelism dependency — that the
descriptor wires up by passing the standard HF projection weights (``gate/up/down``,
``q/k/v/o``) and a head layout. They back the whole "sorted teacher + dynamic prune" flow:

* **permute_\\***: reorder a contracted dim by importance. Because the reorder is applied
  consistently to every weight that touches that dim, the module's output is **unchanged** — this
  is how the sorted teacher stays functionally identical to the teacher (unit-tested).
* **slice_\\***: keep a subset of channels/heads -> physically smaller weights (bypass / realize).
* **\\*_keep_mask**: a boolean mask over the projection's *input* features used to prune by zeroing
  activations at the ``down_proj`` / ``o_proj`` input during scoring (exact for removal: the masked
  forward equals the sliced model's forward).

Conventions (standard HF GQA layout): ``gate``/``up`` are ``[intermediate, hidden]`` (rows =
intermediate channels), ``down`` is ``[hidden, intermediate]`` (cols = intermediate channels);
``q`` is usually ``[num_q*head_dim, hidden]``, ``k``/``v`` are
``[num_kv*head_dim, hidden]``, and ``o`` is ``[hidden, num_q*head_dim]``. Some
families store multiple row groups per query head in ``q`` (for example Qwen3.5
stores query and gate rows); those row groups are carried together when query
heads are permuted or sliced. Query heads are grouped by KV head (group-major:
``q_index = group*n_heads_in_group + head_in_group``).
"""

import torch

__all__ = [
    "ffn_permutation",
    "permute_ffn_weights",
    "slice_ffn_weights",
    "ffn_keep_mask",
    "attention_permutations",
    "grouped_attention_permutations",
    "attention_permutations_from_scores",
    "permute_query_rows_by_head",
    "permute_attention_weights",
    "slice_query_rows_by_head",
    "slice_attention_weights",
    "sorted_attention_keep_indices",
    "attention_keep_mask",
    "aggregate_query_scores_to_kv",
]


# --------------------------------------------------------------------------- FFN
def ffn_permutation(channel_importance: torch.Tensor) -> torch.Tensor:
    """Descending-importance permutation of the intermediate channels (``[intermediate]`` -> idx)."""
    return torch.argsort(channel_importance, descending=True)


def permute_ffn_weights(gate, up, down, perm):
    """Reorder the intermediate dim consistently. Output is unchanged (permutation invariant)."""
    return gate[perm], up[perm], down[:, perm]


def slice_ffn_weights(gate, up, down, keep):
    """Keep the ``keep`` intermediate channels (LongTensor / slice) -> smaller weights."""
    return gate[keep], up[keep], down[:, keep]


def ffn_keep_mask(intermediate_size: int, keep, device=None) -> torch.Tensor:
    """Boolean ``[intermediate]`` mask (True = keep) for masking the ``down_proj`` input."""
    mask = torch.zeros(intermediate_size, dtype=torch.bool, device=device)
    mask[keep] = True
    return mask


# ----------------------------------------------------------------------- attention
def attention_permutations(
    query_importance: torch.Tensor, num_kv_heads: int, n_heads_in_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-major query-head permutation + KV-head permutation, both by importance.

    Groups are ordered by aggregate (summed) query-head importance; query heads within each group
    are ordered by their own importance. Returns ``(q_perm [num_q], kv_perm [num_kv])`` such that
    applying them to q/o (q_perm) and k/v (kv_perm) yields the sorted, functionally-identical layout.
    """
    q = query_importance.view(num_kv_heads, n_heads_in_group)
    group_perm = torch.argsort(q.sum(dim=1), descending=True)  # [num_kv]
    within = torch.argsort(q, dim=1, descending=True)  # [num_kv, n_in_group] (original group order)
    # Original query index of (reordered group i, within-rank j) = g_orig*n_in_group + within[g_orig][j].
    q_perm = (group_perm.view(-1, 1) * n_heads_in_group + within[group_perm]).reshape(-1)
    return q_perm, group_perm


def grouped_attention_permutations(
    kv_group_importance: torch.Tensor,
    query_importance: torch.Tensor,
    num_kv_heads: int,
    n_heads_in_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort KV groups and query heads from explicit grouped-attention scores.

    ``kv_group_importance`` ranks whole KV groups. ``query_importance`` ranks query
    heads within each original KV group. The returned query permutation is
    group-major after the KV group sort and prefix-sliceable for both supported
    attention operations: drop whole KV groups and keep the same number of query
    heads per surviving group.
    """
    kv = kv_group_importance.reshape(num_kv_heads)
    q = query_importance.reshape(num_kv_heads, n_heads_in_group)
    group_perm = torch.argsort(kv, descending=True)
    within = torch.argsort(q, dim=1, descending=True)
    q_perm = (group_perm.view(-1, 1) * n_heads_in_group + within[group_perm]).reshape(-1)
    return q_perm, group_perm


def aggregate_query_scores_to_kv(query_importance: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Sum a per-query-head ``[num_q]`` importance into per-KV-head ``[num_kv]`` (group-major)."""
    return query_importance.view(num_kv_heads, -1).sum(dim=1)


def attention_permutations_from_scores(
    attn_importance: torch.Tensor, num_q_heads: int, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_perm [num_q], kv_perm [num_kv])`` from either a per-query or per-KV importance vector.

    Legacy compatibility helper for callers that still pass one flat attention score vector.
    New Puzzletron attention scoring should prefer :func:`grouped_attention_permutations`, which
    takes explicit KV-group and within-group query-head scores.
    """
    n_in = num_q_heads // num_kv_heads
    numel = attn_importance.numel()
    if numel == num_q_heads:
        return attention_permutations(attn_importance, num_kv_heads, n_in)
    if numel == num_kv_heads:
        group_perm = torch.argsort(attn_importance, descending=True)  # [num_kv]
        # Group-major query indices for each reordered group, original within-group order.
        q_perm = (group_perm.view(-1, 1) * n_in + torch.arange(n_in)).reshape(-1)
        return q_perm, group_perm
    raise ValueError(
        f"attention importance has {numel} entries; expected num_q={num_q_heads} (per-query score) "
        f"or num_kv={num_kv_heads} (per-KV score)"
    )


def _query_row_group_factor(q: torch.Tensor, num_q_heads: int, head_dim: int) -> int:
    base_rows = num_q_heads * head_dim
    if base_rows <= 0 or q.shape[0] % base_rows != 0:
        raise ValueError(
            f"query projection has {q.shape[0]} rows; expected a multiple of "
            f"num_q_heads*head_dim ({num_q_heads}*{head_dim}={base_rows})"
        )
    return q.shape[0] // base_rows


def permute_query_rows_by_head(
    q: torch.Tensor, q_perm: torch.Tensor, head_dim: int, num_q_heads: int
) -> torch.Tensor:
    """Permute query-head rows, preserving any extra per-head row groups."""
    row_groups = _query_row_group_factor(q, num_q_heads, head_dim)
    return q.view(num_q_heads, row_groups, head_dim, *q.shape[1:])[q_perm].reshape(q.shape)


def slice_query_rows_by_head(
    q: torch.Tensor, keep_q_heads: torch.Tensor, head_dim: int, num_q_heads: int
) -> torch.Tensor:
    """Slice query-head rows, preserving any extra per-head row groups."""
    row_groups = _query_row_group_factor(q, num_q_heads, head_dim)
    return q.view(num_q_heads, row_groups, head_dim, *q.shape[1:])[keep_q_heads].reshape(
        -1, *q.shape[1:]
    )


def permute_attention_weights(q, k, v, o, q_perm, kv_perm, head_dim):
    """Reorder query heads (q rows, o cols) and KV heads (k/v rows). Output unchanged (invariant)."""
    num_q = o.shape[-1] // head_dim
    num_kv = k.shape[0] // head_dim
    hidden_out = o.shape[0]
    q = permute_query_rows_by_head(q, q_perm, head_dim, num_q)
    k = k.view(num_kv, head_dim, -1)[kv_perm].reshape(num_kv * head_dim, -1)
    v = v.view(num_kv, head_dim, -1)[kv_perm].reshape(num_kv * head_dim, -1)
    o = o.view(hidden_out, num_q, head_dim)[:, q_perm].reshape(hidden_out, num_q * head_dim)
    return q, k, v, o


def slice_attention_weights(q, k, v, o, keep_q_heads, keep_kv_heads, head_dim):
    """Keep the given query heads (q rows, o cols) and KV heads (k/v rows) -> smaller weights."""
    num_q = o.shape[-1] // head_dim
    num_kv = k.shape[0] // head_dim
    hidden_out = o.shape[0]
    q = slice_query_rows_by_head(q, keep_q_heads, head_dim, num_q)
    k = k.view(num_kv, head_dim, -1)[keep_kv_heads].reshape(-1, k.shape[1])
    v = v.view(num_kv, head_dim, -1)[keep_kv_heads].reshape(-1, v.shape[1])
    o = o.view(hidden_out, num_q, head_dim)[:, keep_q_heads].reshape(hidden_out, -1)
    return q, k, v, o


def sorted_attention_keep_indices(
    target_kv_heads: int, target_heads_in_group: int, orig_heads_in_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heads to keep when pruning a *sorted* attention layer to (target q, target kv).

    On the sorted teacher, keeping the most important (q, kv) = the first ``target_kv_heads`` groups
    and, within each, its first ``target_heads_in_group`` query heads. Returns
    ``(keep_q_heads, keep_kv_heads)`` index tensors (the query set is blocked, not a flat prefix).
    """
    keep_kv = torch.arange(target_kv_heads)
    keep_q = torch.cat(
        [
            torch.arange(target_heads_in_group) + g * orig_heads_in_group
            for g in range(target_kv_heads)
        ]
    )
    return keep_q, keep_kv


def attention_keep_mask(num_q_heads: int, keep_q_heads, head_dim: int, device=None) -> torch.Tensor:
    """Boolean ``[num_q*head_dim]`` mask (True = keep) for masking the ``o_proj`` input.

    Zeroing the removed query heads' columns at the o_proj input reproduces the pruned attention
    output exactly (each query head contributes independently to o_proj).
    """
    mask = torch.zeros(num_q_heads, head_dim, dtype=torch.bool, device=device)
    mask[keep_q_heads] = True
    return mask.reshape(num_q_heads * head_dim)
