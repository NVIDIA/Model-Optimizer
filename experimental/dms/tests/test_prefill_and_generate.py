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

"""Tests for DMS flex attention prefill and inference."""

import pytest
import torch

from experimental.dms.tests.utils import add_dms_to_path, ignore_flex_attention_warnings

try:
    from dms.attention import dms_attn_eval_mode
    from dms.attention_prefill import dms_run_prefill_flex, wrapped_flex_attention
    from dms.cache import DMSCombinedCacheLayer
except ImportError:
    add_dms_to_path()
    from dms.attention import dms_attn_eval_mode
    from dms.attention_prefill import dms_run_prefill_flex, wrapped_flex_attention
    from dms.cache import DMSCombinedCacheLayer

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MASK_VALUE = -1e9


def fake_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool,
    softmax_scale: float,
    block_table: torch.Tensor,
    return_softmax_lse: bool = False,
):
    """Naive flash_attn_with_kvcache replacement that reconstructs KV from paged blocks."""
    k = k_blocks
    v = v_blocks

    _num_blocks, block_size, _, head_dim = k.size()
    page_batch, seq_len, q_per_kv, _head_dim = q.size()

    cont_cache_keys = []
    cont_cache_values = []

    max_seq_len = cache_seqlens.max()
    max_blocks = (max_seq_len + block_size - 1) // block_size

    for pb in range(page_batch):
        cont_cache_keys.append([])
        cont_cache_values.append([])
        for b in range(max_blocks):
            block_id = block_table[pb, b]
            cont_cache_keys[pb].append(k[block_id, :, 0, :])
            cont_cache_values[pb].append(v[block_id, :, 0, :])

        cont_cache_keys[pb] = torch.cat(cont_cache_keys[pb], dim=0)
        cont_cache_values[pb] = torch.cat(cont_cache_values[pb], dim=0)

    cont_cache_keys = torch.stack(cont_cache_keys, dim=0)
    cont_cache_values = torch.stack(cont_cache_values, dim=0)

    attention_mask = (
        torch.arange(cont_cache_keys.size(1), device=q.device)[None, :] >= cache_seqlens[:, None]
    )

    attn_scores = torch.einsum("bsqd,bld->bqsl", q, cont_cache_keys * softmax_scale)

    attention_mask = attention_mask[:, None, None, :].broadcast_to(attn_scores.size())

    iter_k = torch.arange(cont_cache_keys.size(1), device=q.device)
    iter_q = (
        torch.arange(seq_len, device=q.device)[None, None, :, None]
        + cache_seqlens[:, None, None, None]
        - seq_len
    )
    causal_mask = iter_q[:, :, :, :] < iter_k[None, None, None, :]
    if not causal:
        causal_mask = torch.zeros_like(causal_mask, dtype=torch.bool)

    attention_mask = torch.logical_or(attention_mask, causal_mask)

    attn_scores[attention_mask] = -1e9

    logsumexp = torch.logsumexp(attn_scores, dim=-1)

    attn_scores = torch.softmax(attn_scores, dim=-1)

    result = torch.einsum("bqsl,bld->bsqd", attn_scores, cont_cache_values)

    if return_softmax_lse:
        return result, logsumexp
    else:
        return result


def _simple_code_for_dms_exact_attention(q, k, v, d, a, state, attn_scaling, window_size):
    """Reference implementation of DMS exact attention for prefill verification."""
    page_batch, seq_len_q, q_per_kv, head_dim = q.size()
    batch, head_k, seq_len_k, head_dim_k = k.size()
    assert seq_len_q == seq_len_k
    k = k.reshape(page_batch, seq_len_k, head_dim)
    v = v.reshape(page_batch, seq_len_k, head_dim)
    d = d.reshape(page_batch, seq_len_k)
    a = a.reshape(page_batch, seq_len_k)

    org_a = a.clone()

    if len(state) != 0:
        k = torch.cat([state["k"], k], dim=1)
        v = torch.cat([state["v"], v], dim=1)
        d = torch.cat([state["d"], d], dim=1)
        a = torch.cat([state["a"], a], dim=1)

    attn_scores = torch.einsum("psgh,plh->pgsl", q, k * attn_scaling)

    seq_len_k = k.shape[1]
    offset = seq_len_k - seq_len_q
    id_q = torch.arange(seq_len_q, device=q.device, dtype=torch.int32) + offset
    id_k = torch.arange(seq_len_k, device=k.device, dtype=torch.int32)

    causal_mask = id_q[:, None] < id_k[None, :]

    attn_mask = (
        torch.logical_not(a).reshape(page_batch, 1, 1, seq_len_k).broadcast_to(attn_scores.size())
    )
    causal_mask = causal_mask[None, None, :, :].broadcast_to(attn_scores.size())
    attn_mask = torch.logical_or(attn_mask, causal_mask)
    dms_within_window = (id_q[:, None] - id_k[None, :]) < window_size

    shifted_d = torch.nn.functional.pad(d[:, 1:], (0, 1), value=0)
    dms_eviction = shifted_d == 1
    dms_masked = torch.logical_and(
        torch.logical_not(dms_within_window[None, None, :, :]),
        dms_eviction[:, None, None, :],
    )

    attn_mask = torch.logical_or(attn_mask, dms_masked)
    attn_scores[attn_mask] = MASK_VALUE

    state["k"] = k
    state["v"] = v
    state["d"] = d
    state["a"] = a

    attn_scores = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.einsum("pgsl,plh->psgh", attn_scores, v)

    tmp = org_a[:, :, None, None].broadcast_to(attn_output.size())
    attn_output[torch.logical_not(tmp)] = 0

    return attn_output.reshape(batch, head_k, seq_len_q, q_per_kv, head_dim)


def _simple_code_for_dms_fast_attention_inference(q, k, v, d, a, attn_scaling, window_size):
    """Reference implementation of DMS attention for single-step inference verification."""
    page_batch, seq_len_q, q_per_kv, head_dim = q.size()
    batch, head_k, seq_len_k, head_dim_k = k.size()
    assert seq_len_q == 1
    assert seq_len_k > seq_len_q
    k = k.reshape(page_batch, seq_len_k, head_dim)
    v = v.reshape(page_batch, seq_len_k, head_dim)
    d = d.reshape(page_batch, seq_len_k)
    a = a.reshape(page_batch, seq_len_k)

    attn_scores = torch.einsum("psgh,plh->pgsl", q, k * attn_scaling)

    shifted_d = torch.nn.functional.pad(d[:, 1:], (0, 1), value=0)
    k_iter = torch.arange(seq_len_k - 1, -1, -1, device=d.device, dtype=torch.int32)
    should_be_evicted = torch.logical_and(shifted_d == 1, k_iter[None, :] >= window_size)

    attn_mask = (
        torch.logical_not(a).reshape(page_batch, 1, 1, seq_len_k).broadcast_to(attn_scores.size())
    )
    attn_mask = torch.logical_or(attn_mask, should_be_evicted[:, None, None, :])

    attn_scores[attn_mask] = MASK_VALUE
    attn_scores = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.einsum("pgsl,plh->psgh", attn_scores, v)

    return attn_output.reshape(batch, head_k, seq_len_q, q_per_kv, head_dim)


def _generate_random_test_params(seed):
    """Generate randomized test parameters from a seed."""
    torch.manual_seed(seed)
    batch = torch.randint(1, 5, (1,)).item()
    heads_kv = torch.randint(1, 5, (1,)).cuda().item()
    gqa_factor = torch.randint(1, 4, (1,)).cuda().item()
    seq_len = torch.randint(8, 1024, (1,)).cuda().item()
    head_dim = 3
    chunk_size = torch.randint(1, 128, (1,)).cuda().item()
    dms_block_size = torch.randint(2, 32, (1,)).cuda().item()
    dms_window_size = torch.randint(dms_block_size + 1, 128, (1,)).cuda().item()

    return {
        "batch": batch,
        "heads_kv": heads_kv,
        "gqa_factor": gqa_factor,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "chunk_size": chunk_size,
        "dms_block_size": dms_block_size,
        "dms_window_size": dms_window_size,
    }


def _run_prefill(params):
    """Run chunked prefill and verify against the reference implementation.

    Returns the tensors and cache needed for the subsequent inference test.
    """
    batch = params["batch"]
    heads_kv = params["heads_kv"]
    gqa_factor = params["gqa_factor"]
    seq_len = params["seq_len"]
    head_dim = params["head_dim"]
    chunk_size = params["chunk_size"]
    dms_block_size = params["dms_block_size"]
    dms_window_size = params["dms_window_size"]

    query = torch.randn(
        (batch * heads_kv, seq_len, gqa_factor, head_dim), dtype=torch.float64
    ).cuda()
    key = torch.randn((batch, heads_kv, seq_len, head_dim), dtype=torch.float64).cuda()
    value = torch.randn((batch, heads_kv, seq_len, head_dim), dtype=torch.float64).cuda()
    decisions = (torch.randint(0, 100, (batch, heads_kv, seq_len)) <= 90).to(torch.long).cuda()
    attention_mask = torch.ones((batch, seq_len), dtype=torch.bool).cuda()

    for i in range(batch):
        rnd = torch.randint(0, 2, (1,)).item()
        if rnd == 0:
            pad_len = torch.randint(0, 6, (1,)).item()
            attention_mask[i, :pad_len] = False

    cache = DMSCombinedCacheLayer(
        dms_window_size=dms_window_size,
        max_context_length=8192,
        block_size=dms_block_size,
    )
    cache.prefill_mode()
    state = {}

    for chunk_idx in range((seq_len + chunk_size - 1) // chunk_size):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, seq_len)
        q = query[:, start_idx:end_idx, :, :]
        k = key[:, :, start_idx:end_idx, :]
        v = value[:, :, start_idx:end_idx, :]
        d = decisions[:, :, start_idx:end_idx]
        a = attention_mask[:, start_idx:end_idx]

        attn_output_actual = dms_run_prefill_flex(
            q_flash=q,
            keys=k,
            values=v,
            decisions=d,
            attn_mask=a,
            cache=cache,
            attn_scaling=1.0,
            flash_attn_fn=fake_flash_attn_with_kvcache,
            flex_attention_fn=wrapped_flex_attention,
        )

        a_expanded = a.reshape(a.shape[0], 1, a.shape[1]).broadcast_to(
            (a.shape[0], heads_kv, a.shape[1])
        )
        attn_output_expected = _simple_code_for_dms_exact_attention(
            q=q,
            k=k,
            v=v,
            d=d,
            a=a_expanded,
            state=state,
            attn_scaling=1.0,
            window_size=dms_window_size,
        )

        diff = (attn_output_actual - attn_output_expected).abs().max()
        assert diff < 1e-6, f"Prefill chunk {chunk_idx}: max diff = {diff}"

    return query, key, value, decisions, attention_mask, cache


@requires_cuda
@ignore_flex_attention_warnings
class TestDMSPrefill:
    """Tests for DMS prefill attention matching the reference implementation."""

    @pytest.mark.parametrize("seed", range(5))
    def test_prefill_matches_reference(self, seed):
        """Chunked prefill output should match the naive reference implementation."""
        params = _generate_random_test_params(seed)
        _run_prefill(params)


@requires_cuda
@ignore_flex_attention_warnings
class TestDMSInferenceAfterPrefill:
    """Tests for DMS inference-mode attention after prefill."""

    @pytest.mark.parametrize("seed", range(5))
    def test_generate_after_prefill_matches_reference(self, seed):
        """Single-step generation output should match the naive reference implementation."""
        params = _generate_random_test_params(seed)
        query, total_k, total_v, total_d, total_a, cache = _run_prefill(params)

        torch.manual_seed(seed)
        cache.inference_mode()

        page_batch, _seq_len_q, q_per_kv, head_dim = query.size()
        batch, head_k, _seq_len_k, head_dim_k = total_k.size()
        dms_window_size = cache.dms_window_size

        num_generate_steps = 10
        for step in range(num_generate_steps):
            q = torch.randn((page_batch, 1, q_per_kv, head_dim), dtype=torch.float64).cuda()
            k = torch.randn((batch, head_k, 1, head_dim_k), dtype=torch.float64).cuda()
            v = torch.randn((batch, head_k, 1, head_dim_k), dtype=torch.float64).cuda()
            d = torch.randint(0, 2, (batch, head_k, 1), dtype=torch.long).cuda()
            a = torch.ones((batch, 1), dtype=torch.bool).cuda()

            total_k = torch.cat([total_k, k], dim=2)
            total_v = torch.cat([total_v, v], dim=2)
            total_d = torch.cat([total_d, d], dim=2)
            total_a = torch.cat([total_a, a], dim=1)

            attn_output_actual = dms_attn_eval_mode(
                new_q_flash=q.clone(),
                new_k=k.clone(),
                new_v=v.clone(),
                decisions=d.clone(),
                attention_mask=None,
                layer_idx=0,
                dms_cache=[cache],
                attn_scaling=1.0,
                flash_attn_fn=fake_flash_attn_with_kvcache,
                prefill_attn_fn_kwargs={
                    "flex_attention_fn": wrapped_flex_attention,
                },
            )

            total_a_expanded = total_a.reshape(batch, 1, total_a.shape[1]).broadcast_to(
                (batch, head_k, total_a.shape[1])
            )
            attn_output_expected = _simple_code_for_dms_fast_attention_inference(
                q=q,
                k=total_k,
                v=total_v,
                d=total_d,
                a=total_a_expanded,
                attn_scaling=1.0,
                window_size=dms_window_size,
            )

            diff = (attn_output_actual - attn_output_expected).abs().max()
            assert diff < 1e-7, f"Generate step {step}: max diff = {diff}"
