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

"""GPU tests for the vLLM sparse attention plugin (ModelOptSparseAttentionImpl).

Covers the integration-critical metadata translation done in
``ModelOptSparseAttentionImpl.forward``:

* ``query_start_loc``       -> ``b_start_loc`` / ``b_seq_len``
* ``seq_lens``              -> ``b_seq_len_k``
* ``kv_cache.unbind(0)``    -> key_cache / value_cache (axis order)
* ``k_cache.shape[1]``      -> ``page_size``

Asserted against a contiguous reference call to the underlying Triton kernel.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    collect_calibration_stats,
    disable_calibration,
    enable_calibration,
    fit_calibration,
)

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention import attention as triton_attention

_ACTIVE_PREFILL_SPARSE_KW = {
    "sparsity_n": 2,
    "sparsity_m": 4,
    "dense_sink_tokens": 0,
    "dense_recent_tokens": 0,
}


def _make_paged_cache(k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size):
    """Scatter contiguous K/V into a paged KV cache stacked as [2, ...].

    Returns a single ``kv_cache`` tensor (matching vLLM's layout that
    ``ModelOptSparseAttentionImpl`` consumes via ``kv_cache.unbind(0)``).
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

    # Stack on a new leading axis so kv_cache.unbind(0) recovers (k_cache, v_cache).
    kv_cache = torch.stack([k_cache, v_cache], dim=0)
    return kv_cache, block_table


def _make_impl(num_heads, head_dim, num_kv_heads):
    """Construct ModelOptSparseAttentionImpl with minimal valid kwargs."""
    return ModelOptSparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestModelOptSparseAttentionImpl:
    """Verify forward() metadata translation matches a contiguous reference."""

    def test_prefill_matches_contiguous(self):
        """Prefill: paged forward == contiguous Triton call on the same K/V."""
        batch = 2
        seq_len = 64
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        total = batch * seq_len
        dtype = torch.float16

        torch.manual_seed(0)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)

        # Per-sequence offsets / lengths.
        seq_lens = torch.tensor([seq_len, seq_len], device="cuda", dtype=torch.int32)
        # vLLM-style cumulative query_start_loc has shape [batch + 1].
        query_start_loc = torch.tensor([0, seq_len, 2 * seq_len], device="cuda", dtype=torch.int32)
        b_start_loc = query_start_loc[:batch]
        b_seq_len = seq_lens

        # Contiguous reference output (what the kernel would return without paging).
        out_ref = triton_attention(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            seq_len,
            softmax_scale=1.0 / (head_dim**0.5),
            **_ACTIVE_PREFILL_SPARSE_KW,
        )

        # Build paged kv_cache shaped [2, num_blocks, page_size, num_kv_heads, head_dim].
        kv_cache, block_table = _make_paged_cache(
            k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size
        )

        attn_metadata = SimpleNamespace(
            num_actual_tokens=total,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        output = torch.empty_like(q)
        out_paged = impl.forward(
            layer=None,
            query=q,
            key=k,
            value=v,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

        torch.testing.assert_close(out_paged, out_ref, rtol=1e-2, atol=1e-2)

    def test_chunked_prefill_is_forwarded_to_kernel(self):
        """Chunked prefill metadata is handled by the suffix-aware causal mask."""
        impl = _make_impl(num_heads=2, head_dim=64, num_kv_heads=2)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        attn_metadata = SimpleNamespace(
            num_actual_tokens=4,
            max_query_len=4,  # chunk length
            max_seq_len=16,  # full sequence length > chunk
            query_start_loc=torch.tensor([0, 4], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor([16], device="cuda", dtype=torch.int32),
            block_table=torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        )
        q = torch.zeros(4, 2, 64, device="cuda", dtype=torch.float16)
        kv_cache = torch.zeros(2, 1, 16, 2, 64, device="cuda", dtype=torch.float16)
        out = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=torch.empty_like(q),
        )

        assert torch.isfinite(out).all()

    def test_mixed_prefill_decode_is_forwarded_to_kernel(self):
        """Mixed prefill/decode batches are valid with suffix-aware causal masking."""
        prefill_len = 64
        decode_q_len = 1
        total_q = prefill_len + decode_q_len
        num_heads, num_kv_heads, head_dim = 2, 2, 64
        page_size = 16
        dtype = torch.float16

        q = torch.zeros(total_q, num_heads, head_dim, device="cuda", dtype=dtype)

        # Sequence 0 is a full prefill. Sequence 1 is decode with one query
        # token but a longer KV cache. max_query_len == max_seq_len, so the
        # older max-only guard would not catch this mixed batch.
        seq_lens = torch.tensor([prefill_len, prefill_len], device="cuda", dtype=torch.int32)
        query_start_loc = torch.tensor([0, prefill_len, total_q], device="cuda", dtype=torch.int32)
        b_start_loc_k = torch.tensor([0, prefill_len], device="cuda", dtype=torch.int32)
        k = torch.zeros(prefill_len * 2, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.zeros_like(k)
        kv_cache, block_table = _make_paged_cache(
            k, v, b_start_loc_k, seq_lens, num_kv_heads, head_dim, page_size
        )

        attn_metadata = SimpleNamespace(
            num_actual_tokens=total_q,
            max_query_len=prefill_len,
            max_seq_len=prefill_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        out = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=torch.empty_like(q),
        )

        assert torch.isfinite(out).all()

    def test_decode_delegates_to_vllm(self, monkeypatch):
        """Decode-only sparse work is not routed through the ModelOpt paged kernel."""
        batch = 2
        q_len = 1
        kv_lens = torch.tensor([17, 33], device="cuda", dtype=torch.int32)
        total_q = batch * q_len
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        dtype = torch.float16

        torch.manual_seed(2)
        q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(
            int(kv_lens.sum().item()), num_kv_heads, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn_like(k)

        query_start_loc = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
        kv_start_loc = torch.tensor([0, int(kv_lens[0].item())], device="cuda", dtype=torch.int32)

        kv_cache, block_table = _make_paged_cache(
            k, v, kv_start_loc, kv_lens, num_kv_heads, head_dim, page_size
        )
        attn_metadata = SimpleNamespace(
            num_actual_tokens=total_q,
            max_query_len=q_len,
            max_seq_len=int(kv_lens.max().item()),
            query_start_loc=query_start_loc,
            seq_lens=kv_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        output = torch.empty_like(q)
        called = {}

        def fake_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache_arg,
            attn_metadata_arg,
            output_arg=None,
            output_scale=None,
            output_block_scale=None,
        ):
            called["attn_metadata"] = attn_metadata_arg
            output_arg.fill_(9)
            return output_arg

        monkeypatch.setattr(FlashAttentionImpl, "forward", fake_forward)

        result = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
        assert called["attn_metadata"] is attn_metadata
        assert result is output
        assert torch.all(result == 9)

    def test_profiling_run_returns_zeros(self):
        """attn_metadata=None (vLLM profiling pass) must zero-fill output and return."""
        impl = _make_impl(num_heads=2, head_dim=64, num_kv_heads=2)
        output = torch.full((4, 2, 64), 7.0, device="cuda", dtype=torch.float16)
        result = impl.forward(
            layer=None,
            query=output,
            key=output,
            value=output,
            kv_cache=torch.empty(0),
            attn_metadata=None,
            output=output,
        )
        assert torch.all(result == 0)

    def test_page_size_inferred_from_k_cache(self):
        """page_size passed to the kernel must equal k_cache.shape[1]."""
        # Use the smallest valid power-of-two page_size to confirm it's not hardcoded.
        seq_len = 32
        num_heads, num_kv_heads, head_dim = 2, 2, 64
        page_size = 8  # deliberately != default 16
        dtype = torch.float16

        torch.manual_seed(1)
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)
        b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        query_start_loc = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

        out_ref = triton_attention(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            seq_len,
            softmax_scale=1.0 / (head_dim**0.5),
            **_ACTIVE_PREFILL_SPARSE_KW,
        )

        kv_cache, block_table = _make_paged_cache(
            k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size
        )
        # Sanity: kv_cache axis 1 is page_size.
        assert kv_cache.shape == (2, seq_len // page_size, page_size, num_kv_heads, head_dim)

        attn_metadata = SimpleNamespace(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            query_start_loc=query_start_loc,
            seq_lens=b_seq_len,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        output = torch.empty_like(q)
        out_paged = impl.forward(
            layer=None,
            query=q,
            key=k,
            value=v,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
        torch.testing.assert_close(out_paged, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestModelOptSparseAttentionCalibration:
    """Calibration mode: ``forward`` measures tile-skip stats via the paged kernel.

    Output must stay dense (calibration computes full attention) while per-request
    records accumulate, ready to fit the exponential ``(a, b)`` model.
    """

    _TRIALS = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 9e-1]

    def test_prefill_calibration_records_and_dense_output(self):
        """Prefill: output equals dense attention; one record per request."""
        lengths = [128, 256]
        total = sum(lengths)
        num_heads, num_kv_heads, head_dim, page_size = 4, 2, 64, 16
        dtype = torch.float16

        torch.manual_seed(0)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)

        seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        query_start_loc = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
        kv_cache, block_table = _make_paged_cache(
            k, v, query_start_loc[:2], seq_lens, num_kv_heads, head_dim, page_size
        )
        attn_metadata = SimpleNamespace(
            num_actual_tokens=total,
            max_query_len=max(lengths),
            max_seq_len=max(lengths),
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = {}
        enable_calibration([impl], self._TRIALS)
        output = torch.empty_like(q)
        out = impl.forward(
            layer=None,
            query=q,
            key=k,
            value=v,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

        # Output is dense per-request causal attention (full attention, no skip).
        for i, length in enumerate(lengths):
            start = int(query_start_loc[i].item())
            locs = torch.zeros(1, device="cuda", dtype=torch.int32)
            lens = torch.tensor([length], device="cuda", dtype=torch.int32)
            ref = triton_attention(
                q[start : start + length],
                k[start : start + length],
                v[start : start + length],
                locs,
                lens,
                length,
                softmax_scale=1.0 / (head_dim**0.5),
                is_causal=True,
            )
            torch.testing.assert_close(out[start : start + length], ref, rtol=5e-3, atol=5e-3)

        stats = collect_calibration_stats([impl])
        assert len(stats["prefill"]) == len(lengths)
        assert [r["sample_length"] for r in stats["prefill"]] == lengths
        assert all(len(r["sparsity"]) == len(self._TRIALS) for r in stats["prefill"])
        assert not stats["decode"]

    def test_decode_calibration_records_decode_phase(self):
        """Decode (seq_q=1, long cache): a decode record with real sparsity."""
        seq_k = 2048
        num_heads, num_kv_heads, head_dim, page_size = 4, 2, 64, 16
        dtype = torch.float16

        # A dominant sink at position 0 makes the distant cache skippable.
        q = torch.ones(1, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.zeros(seq_k, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        k[0] = 20.0
        v = torch.randn(seq_k, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        kv_start = torch.zeros(1, device="cuda", dtype=torch.int32)
        kv_len = torch.tensor([seq_k], device="cuda", dtype=torch.int32)
        kv_cache, block_table = _make_paged_cache(
            k, v, kv_start, kv_len, num_kv_heads, head_dim, page_size
        )
        attn_metadata = SimpleNamespace(
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=seq_k,
            query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            seq_lens=kv_len,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = {}
        enable_calibration([impl], self._TRIALS)
        impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=torch.empty_like(q),
        )

        stats = collect_calibration_stats([impl])
        assert not stats["prefill"]
        assert len(stats["decode"]) == 1
        record = stats["decode"][0]
        assert record["sample_length"] == seq_k
        assert max(record["sparsity"]) > 0.8  # sink => most tiles skippable

    _FIT_TRIALS = [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 7e-1, 9e-1]

    def test_fit_calibration_produces_exponential_params(self):
        """Multiple lengths feed fit_calibration into a usable (a, b) per phase."""
        num_heads, num_kv_heads, head_dim, page_size = 4, 2, 64, 16
        dtype = torch.float16
        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = {}
        enable_calibration([impl], self._FIT_TRIALS)

        torch.manual_seed(0)
        for length in (512, 1024, 2048):
            # Localized attention (key norm decays with distance + a sink at 0)
            # gives a graded skip sweep across thresholds, so the exponential
            # fit's (10%, 90%) window has enough data points — unlike uniform
            # random keys, which skip all-or-nothing.
            q = torch.randn(length, num_heads, head_dim, device="cuda", dtype=dtype)
            pos = torch.arange(length, device="cuda").float()
            decay = torch.exp(-pos / (length * 0.15))[:, None, None]
            k = (torch.randn(length, num_kv_heads, head_dim, device="cuda") * decay).to(dtype)
            k[0] = 8.0  # sink
            v = torch.randn(length, num_kv_heads, head_dim, device="cuda", dtype=dtype)
            locs = torch.zeros(1, device="cuda", dtype=torch.int32)
            lens = torch.tensor([length], device="cuda", dtype=torch.int32)
            qsl = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            kv_cache, block_table = _make_paged_cache(
                k, v, locs, lens, num_kv_heads, head_dim, page_size
            )
            attn_metadata = SimpleNamespace(
                num_actual_tokens=length,
                max_query_len=length,
                max_seq_len=length,
                query_start_loc=qsl,
                seq_lens=lens,
                block_table=block_table,
            )
            impl.forward(
                layer=None,
                query=q,
                key=k,
                value=v,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=torch.empty_like(q),
            )

        disable_calibration([impl])
        params = fit_calibration([impl], self._FIT_TRIALS)
        assert "prefill" in params
        assert params["prefill"]["a"] > 0.0
        assert params["prefill"]["b"] > 0.0
