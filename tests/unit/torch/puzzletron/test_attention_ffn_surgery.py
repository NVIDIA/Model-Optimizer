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

"""CPU tests for the model-agnostic FFN/attention surgery primitives.

The keystone property of the sorted-teacher design is **permutation invariance**: reordering a
contracted dim by importance must not change the module's output. These tests verify that for a
SwiGLU FFN and a GQA attention, plus slice/mask correctness.
"""

import torch
import torch.nn.functional as F

from modelopt.torch.puzzletron.pruning.attention_ffn_surgery import (
    aggregate_query_scores_to_kv,
    attention_keep_mask,
    attention_permutations,
    attention_permutations_from_scores,
    ffn_keep_mask,
    ffn_permutation,
    permute_attention_weights,
    permute_ffn_weights,
    slice_attention_weights,
    slice_ffn_weights,
    sorted_attention_keep_indices,
)


def _swiglu(gate, up, down, x):
    return (F.silu(x @ gate.t()) * (x @ up.t())) @ down.t()


def _gqa_attn(q, k, v, o, x, num_q, num_kv, head_dim):
    t = x.shape[0]
    qh = (x @ q.t()).view(t, num_q, head_dim)
    kh = (x @ k.t()).view(t, num_kv, head_dim)
    vh = (x @ v.t()).view(t, num_kv, head_dim)
    n_in = num_q // num_kv
    out = torch.empty(t, num_q, head_dim, dtype=x.dtype)
    for h in range(num_q):
        g = h // n_in
        p = torch.softmax(qh[:, h] @ kh[:, g].t() / head_dim**0.5, dim=-1)
        out[:, h] = p @ vh[:, g]
    return out.reshape(t, num_q * head_dim) @ o.t()


def test_attention_permutations_from_scores_both_shapes():
    torch.manual_seed(0)
    num_q, num_kv = 12, 4
    n_in = num_q // num_kv
    # Per-query score [num_q]: sorts groups (by aggregate) AND query heads within each group.
    q_score = torch.rand(num_q)
    q_perm, kv_perm = attention_permutations_from_scores(q_score, num_q, num_kv)
    q_perm_ref, kv_perm_ref = attention_permutations(q_score, num_kv, n_in)
    assert torch.equal(q_perm, q_perm_ref) and torch.equal(kv_perm, kv_perm_ref)

    # Per-KV score [num_kv] (independent_kv_head_contribution): sorts only the KV groups,
    # query heads keep original within-group order. This is the case that used to crash.
    kv_score = torch.tensor([0.1, 0.9, 0.3, 0.7])
    q_perm, kv_perm = attention_permutations_from_scores(kv_score, num_q, num_kv)
    assert torch.equal(kv_perm, torch.tensor([1, 3, 2, 0]))  # argsort desc
    # group 1 first -> query heads 4..7, then group 3 -> 12..15, etc., original within-group order.
    expected_q = torch.cat([torch.arange(n_in) + g * n_in for g in [1, 3, 2, 0]])
    assert torch.equal(q_perm, expected_q)


def test_attention_permutations_from_scores_bad_size():
    import pytest

    with pytest.raises(ValueError):
        attention_permutations_from_scores(torch.rand(7), num_q_heads=12, num_kv_heads=4)


def test_aggregate_query_scores_to_kv():
    # group-major: [num_kv, n_in] summed over n_in.
    q = torch.tensor([1.0, 2.0, 10.0, 20.0])  # 2 kv groups of 2
    assert torch.equal(aggregate_query_scores_to_kv(q, num_kv_heads=2), torch.tensor([3.0, 30.0]))


def test_ffn_permutation_is_output_invariant():
    torch.manual_seed(0)
    inter, hidden = 12, 6
    gate, up = torch.randn(inter, hidden, dtype=torch.float64), torch.randn(inter, hidden, dtype=torch.float64)
    down = torch.randn(hidden, inter, dtype=torch.float64)
    x = torch.randn(4, hidden, dtype=torch.float64)

    perm = ffn_permutation(torch.rand(inter))
    g2, u2, d2 = permute_ffn_weights(gate, up, down, perm)
    assert torch.allclose(_swiglu(gate, up, down, x), _swiglu(g2, u2, d2, x), atol=1e-10)


def test_attention_permutation_is_output_invariant():
    torch.manual_seed(1)
    num_q, num_kv, head_dim, hidden = 6, 3, 4, 8
    q = torch.randn(num_q * head_dim, hidden, dtype=torch.float64)
    k = torch.randn(num_kv * head_dim, hidden, dtype=torch.float64)
    v = torch.randn(num_kv * head_dim, hidden, dtype=torch.float64)
    o = torch.randn(hidden, num_q * head_dim, dtype=torch.float64)
    x = torch.randn(5, hidden, dtype=torch.float64)

    ref = _gqa_attn(q, k, v, o, x, num_q, num_kv, head_dim)
    q_perm, kv_perm = attention_permutations(torch.rand(num_q), num_kv, num_q // num_kv)
    q2, k2, v2, o2 = permute_attention_weights(q, k, v, o, q_perm, kv_perm, head_dim)
    got = _gqa_attn(q2, k2, v2, o2, x, num_q, num_kv, head_dim)
    assert torch.allclose(ref, got, atol=1e-10), (ref - got).abs().max()


def test_ffn_slice_and_mask_agree():
    torch.manual_seed(2)
    inter, hidden = 10, 4
    gate, up = torch.randn(inter, hidden), torch.randn(inter, hidden)
    down = torch.randn(hidden, inter)
    x = torch.randn(3, hidden)
    keep = torch.tensor([0, 1, 2, 3, 4])  # first 5 of a sorted layer

    # Slicing the weights == zeroing the dropped channels at the down_proj input.
    g2, u2, d2 = slice_ffn_weights(gate, up, down, keep)
    sliced_out = _swiglu(g2, u2, d2, x)
    mask = ffn_keep_mask(inter, keep)
    h = F.silu(x @ gate.t()) * (x @ up.t())
    masked_out = (h * mask) @ down.t()
    assert torch.allclose(sliced_out, masked_out, atol=1e-5)


def test_sorted_keep_indices_and_attention_slice_mask_agree():
    torch.manual_seed(3)
    num_q, num_kv, head_dim, hidden = 6, 3, 4, 8  # 2 heads/group
    q = torch.randn(num_q * head_dim, hidden)
    k = torch.randn(num_kv * head_dim, hidden)
    v = torch.randn(num_kv * head_dim, hidden)
    o = torch.randn(hidden, num_q * head_dim)
    x = torch.randn(4, hidden)

    # Prune sorted layer to (4 q, 2 kv): keep first 2 groups, first 1 head each? here m = 4//2 = 2.
    target_kv, m, orig_in = 2, 2, 2
    keep_q, keep_kv = sorted_attention_keep_indices(target_kv, m, orig_in)
    assert keep_kv.tolist() == [0, 1]
    assert keep_q.tolist() == [0, 1, 2, 3]  # groups 0,1 fully (2 heads each)

    q2, k2, v2, o2 = slice_attention_weights(q, k, v, o, keep_q, keep_kv, head_dim)
    sliced = _gqa_attn(q2, k2, v2, o2, x, target_kv * m, target_kv, head_dim)
    # Masking the o_proj input for the dropped query heads reproduces the sliced attention output,
    # because dropped groups' kv heads only feed their (now-masked) query heads.
    full = torch.empty(x.shape[0], num_q, head_dim)
    qh = (x @ q.t()).view(x.shape[0], num_q, head_dim)
    kh = (x @ k.t()).view(x.shape[0], num_kv, head_dim)
    vh = (x @ v.t()).view(x.shape[0], num_kv, head_dim)
    for h in range(num_q):
        g = h // (num_q // num_kv)
        p = torch.softmax(qh[:, h] @ kh[:, g].t() / head_dim**0.5, dim=-1)
        full[:, h] = p @ vh[:, g]
    mask = attention_keep_mask(num_q, keep_q, head_dim)
    masked = (full.reshape(x.shape[0], -1) * mask) @ o.t()
    assert torch.allclose(sliced, masked, atol=1e-5), (sliced - masked).abs().max()
