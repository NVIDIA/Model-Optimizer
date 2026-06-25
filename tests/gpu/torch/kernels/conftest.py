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

"""Shared fixtures and helpers for Triton flash attention tests."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_KERNELS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items):
    """Silence noisy third-party warnings (triton/torch) for all kernel tests.

    Consolidated here so individual kernel test modules don't each repeat the same
    ``pytest.mark.filterwarnings`` block. Scoped to this directory only — the rest of
    the suite keeps surfacing warnings.
    """
    ignore_marks = [
        pytest.mark.filterwarnings("ignore::UserWarning"),
        pytest.mark.filterwarnings("ignore::RuntimeWarning"),
        pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    ]
    for item in items:
        if item.path.is_relative_to(_KERNELS_DIR):
            for mark in ignore_marks:
                item.add_marker(mark)


def make_qkv(total, num_heads, num_kv_heads, head_dim, device="cuda", dtype=torch.float16):
    """Create packed Q, K, V tensors."""
    q = torch.randn(total, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=dtype)
    return q, k, v


def make_varlen_meta(seq_lens, device="cuda"):
    """Create b_start_loc and b_seq_len from a list of sequence lengths."""
    b_seq_len = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(len(seq_lens), device=device, dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
    return b_start_loc, b_seq_len


def scatter_to_paged_cache(k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size):
    """Scatter contiguous K/V into a paged KV cache + block table.

    Returns ``(k_cache, v_cache, block_table)`` where the caches are shaped
    ``[num_blocks, page_size, num_kv_heads, head_dim]`` and ``block_table`` is
    ``[batch, max_blocks_per_seq]`` — the layout the paged Triton kernels read.
    """
    batch = b_seq_len.shape[0]
    device, dtype = k.device, k.dtype

    blocks_per_seq = [(int(b_seq_len[b].item()) + page_size - 1) // page_size for b in range(batch)]
    num_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    k_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)

    g = 0
    for b in range(batch):
        start = int(b_start_loc[b].item())
        slen = int(b_seq_len[b].item())
        for blk in range(blocks_per_seq[b]):
            block_table[b, blk] = g
            ts = blk * page_size
            te = min(ts + page_size, slen)
            n = te - ts
            k_cache[g, :n] = k[start + ts : start + te]
            v_cache[g, :n] = v[start + ts : start + te]
            g += 1
    return k_cache, v_cache, block_table


def sdpa_reference(q, k, v, b_start_loc, b_seq_len, is_causal=True):
    """SDPA reference. Supports GQA. Returns [total_tokens, num_heads, dim]."""
    batch = b_seq_len.shape[0]
    num_q, num_kv = q.shape[1], k.shape[1]
    parts = []
    for b in range(batch):
        s, n = int(b_start_loc[b].item()), int(b_seq_len[b].item())
        qb = q[s : s + n].unsqueeze(0).permute(0, 2, 1, 3)
        kb = k[s : s + n].unsqueeze(0).permute(0, 2, 1, 3)
        vb = v[s : s + n].unsqueeze(0).permute(0, 2, 1, 3)
        if num_q != num_kv:
            r = num_q // num_kv
            kb = kb.repeat_interleave(r, dim=1)
            vb = vb.repeat_interleave(r, dim=1)
        ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=is_causal)
        parts.append(ob.permute(0, 2, 1, 3).squeeze(0))
    return torch.cat(parts, dim=0)


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    """Tiny Llama: 2 layers, 64 hidden, 4 q-heads, 2 kv-heads, head_dim=16."""
    from _test_utils.torch.transformers_models import create_tiny_llama_dir

    return create_tiny_llama_dir(
        tmp_path_factory.mktemp("tiny_llama"),
        with_tokenizer=True,
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
