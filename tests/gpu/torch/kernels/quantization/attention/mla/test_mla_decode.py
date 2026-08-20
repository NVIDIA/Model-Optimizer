# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Golden tests for the split-K absorbed-MLA decode kernel."""

import pytest
import torch

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.quantization.attention.mla import mla_attention_decode
    from modelopt.torch.kernels.quantization.attention.mla.reference import (
        mla_decode_reference,
        nvfp4_fake_quant,
    )

NATIVE_E4M3_AVAILABLE = TRITON_KERNEL_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)
requires_native_e4m3 = pytest.mark.skipif(
    not NATIVE_E4M3_AVAILABLE, reason="Native E4M3 requires compute capability >= 8.9"
)

pytestmark = pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton attention kernel requires CUDA + triton"
)

_RANK = 128  # kv_lora_rank kept small so tests run fast; 16-divisible
_ROPE = 64
_DIM = _RANK + _ROPE


def _make_decode_inputs(
    seq_lens: list[int],
    num_heads: int = 32,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
):
    """Build an absorbed query, a dense latent, and its paged copy."""
    torch.manual_seed(seed)
    batch = len(seq_lens)
    max_seq = max(seq_lens)
    q = torch.randn(batch, num_heads, _DIM, device="cuda", dtype=dtype)
    latent_dense = torch.randn(batch, max_seq, _DIM, device="cuda", dtype=dtype)

    max_blocks = (max_seq + page_size - 1) // page_size
    cache = torch.zeros(batch * max_blocks, page_size, _DIM, device="cuda", dtype=dtype)
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32, device="cuda")
    # Shuffled page assignment to exercise the block-table walk.
    perm = torch.randperm(batch * max_blocks)
    next_page = 0
    for b, s in enumerate(seq_lens):
        for blk in range((s + page_size - 1) // page_size):
            page = int(perm[next_page])
            next_page += 1
            block_table[b, blk] = page
            lo = blk * page_size
            hi = min(lo + page_size, s)
            cache[page, : hi - lo] = latent_dense[b, lo:hi]
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return q, latent_dense, cache, block_table, seq_lens_t


def _dense_decode(q, latent_dense, seq_lens, scale):
    """Straightforward fp32 decode oracle (no schedule replication)."""
    batch, num_heads, _ = q.shape
    out = torch.zeros(batch, num_heads, _RANK, device=q.device, dtype=torch.float32)
    lse = torch.zeros(batch, num_heads, device=q.device, dtype=torch.float32)
    for b in range(int(seq_lens.shape[0])):
        s = int(seq_lens[b])
        kv = latent_dense[b, :s].float()  # [s, DIM]
        scores = q[b].float() @ kv.T * scale  # [H, s]
        p = torch.softmax(scores, dim=-1)
        out[b] = p @ kv[:, :_RANK]
        lse[b] = torch.logsumexp(scores, dim=-1)
    return out, lse


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


class TestMLADecodeBaseline:
    @pytest.mark.parametrize("num_kv_splits", [1, 32])
    @pytest.mark.parametrize("page_size", [16, 32, 64])
    def test_no_quant_matches_dense(self, num_kv_splits, page_size):
        seq_lens = [200, 7, 64]
        q, latent, cache, block_table, seq_t = _make_decode_inputs(seq_lens, page_size=page_size)
        scale = _DIM**-0.5
        out, lse = mla_attention_decode(
            q,
            cache,
            block_table,
            seq_t,
            softmax_scale=scale,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
            page_size=page_size,
            num_kv_splits=num_kv_splits,
        )
        ref, ref_lse = _dense_decode(q, latent, seq_t, scale)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(lse, ref_lse, rtol=5e-3, atol=5e-3)

    def test_small_head_count(self):
        """Head counts below BLOCK_H exercise the head-group masking."""
        q, latent, cache, block_table, seq_t = _make_decode_inputs([33], num_heads=2, seed=1)
        scale = 0.13
        out, _ = mla_attention_decode(
            q,
            cache,
            block_table,
            seq_t,
            softmax_scale=scale,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
        )
        ref, _ = _dense_decode(q, latent, seq_t, scale)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)

    def test_output_dtype_follows_cache(self):
        q, _, cache, block_table, seq_t = _make_decode_inputs([16], dtype=torch.bfloat16, seed=2)
        out, lse = mla_attention_decode(
            q,
            cache,
            block_table,
            seq_t,
            softmax_scale=0.1,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
        )
        assert out.dtype == torch.bfloat16
        assert lse.dtype == torch.float32


class TestMLADecodeQuant:
    @requires_native_e4m3
    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    def test_p_qdq_matches_split_local_oracle(self, mode):
        seq_lens = [150, 40]
        q, latent, cache, block_table, seq_t = _make_decode_inputs(seq_lens, seed=3)
        scale = _DIM**-0.5
        out, _ = mla_attention_decode(
            q,
            cache,
            block_table,
            seq_t,
            softmax_scale=scale,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
            p_qdq=mode,
        )
        dense_out, _ = mla_attention_decode(
            q,
            cache,
            block_table,
            seq_t,
            softmax_scale=scale,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
        )
        assert not torch.equal(out, dense_out)  # P quant actually applied
        ref = mla_decode_reference(
            q, latent, seq_t, scale, _RANK, p_mode=mode, num_kv_splits=32, block_n=32
        )
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=2e-2)

    @requires_native_e4m3
    def test_full_nvfp4_recipe_cosine(self):
        """Write-once cache QDQ + fp32-carrier Q QDQ + fused NVFP4 P."""
        seq_lens = [96]
        q, latent, cache, block_table, seq_t = _make_decode_inputs(
            seq_lens, dtype=torch.bfloat16, seed=4
        )
        scale = _DIM**-0.5
        # Emulate the module-level write-once QDQ: quantize the cache rows
        # along the feature axis (kv_c and k_pe global scales both 1.0, so a
        # single 16-block pass over the full row is equivalent).
        cache_q = nvfp4_fake_quant(cache.float(), block_axis=-1).to(cache.dtype)
        q_carrier = nvfp4_fake_quant(q.float(), block_axis=-1)  # FP32 QDQ carrier
        out, _ = mla_attention_decode(
            q_carrier,
            cache_q,
            block_table,
            seq_t,
            softmax_scale=scale,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
            p_qdq="nvfp4",
        )
        ref, _ = _dense_decode(q, latent, seq_t, scale)
        assert torch.isfinite(out.float()).all()
        assert _cos(out, ref) > 0.98

    def test_quantized_cache_consumed_as_is(self):
        """BMM2 reads the cache values unchanged (no on-read re-quant)."""
        seq_lens = [64]
        q, _, cache, block_table, seq_t = _make_decode_inputs(seq_lens, seed=5)
        scale = 0.1
        # Any cache contents must flow through V untouched: compare against
        # the dense oracle computed from the exact same (arbitrary) cache.
        latent_view = torch.zeros(1, 64, _DIM, device="cuda", dtype=cache.dtype)
        for blk in range(64 // 16):
            latent_view[0, blk * 16 : (blk + 1) * 16] = cache[int(block_table[0, blk])]
        out, _ = mla_attention_decode(
            q,
            cache,
            block_table,
            seq_t,
            softmax_scale=scale,
            kv_lora_rank=_RANK,
            qk_rope_head_dim=_ROPE,
        )
        ref, _ = _dense_decode(q, latent_view, seq_t, scale)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)


class TestMLADecodeErrors:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_kv_splits": 0}, "num_kv_splits"),
            ({"num_kv_splits": 64}, "num_kv_splits"),
            ({"page_size": 8}, "page_size"),
            ({"p_qdq": "int8"}, "p_qdq must be one of"),
        ],
    )
    def test_invalid_config(self, kwargs, match):
        q, _, cache, block_table, seq_t = _make_decode_inputs([16])
        with pytest.raises(ValueError, match=match):
            mla_attention_decode(
                q,
                cache,
                block_table,
                seq_t,
                softmax_scale=0.1,
                kv_lora_rank=_RANK,
                qk_rope_head_dim=_ROPE,
                **kwargs,
            )

    def test_nvfp4_requires_divisible_dims(self):
        q = torch.randn(1, 4, 88 + 8, device="cuda", dtype=torch.float16)
        cache = torch.randn(4, 16, 96, device="cuda", dtype=torch.float16)
        block_table = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
        seq_t = torch.tensor([16], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="divisible by 16"):
            mla_attention_decode(
                q,
                cache,
                block_table,
                seq_t,
                softmax_scale=0.1,
                kv_lora_rank=88,
                qk_rope_head_dim=8,
                p_qdq="nvfp4",
            )
