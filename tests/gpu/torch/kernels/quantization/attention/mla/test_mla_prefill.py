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

"""Golden tests for the varlen MLA prefill kernel against eager torch oracles."""

import pytest
import torch

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.quantization.attention.mla import mla_prefill_attention
    from modelopt.torch.kernels.quantization.attention.mla.reference import mla_attention_reference

NATIVE_E4M3_AVAILABLE = TRITON_KERNEL_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)
requires_native_e4m3 = pytest.mark.skipif(
    not NATIVE_E4M3_AVAILABLE, reason="Native E4M3 requires compute capability >= 8.9"
)

pytestmark = pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton attention kernel requires CUDA + triton"
)


def _cu_seqlens(seq_lens: list[int], device: str = "cuda") -> torch.Tensor:
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(seq_lens, device=device), dim=0)
    return cu


def _make_mla_qkv(
    q_lens: list[int],
    k_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    lqk: int,
    lv: int,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    total_q, total_k = sum(q_lens), sum(k_lens)
    q = torch.randn(total_q, num_heads, lqk, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_kv_heads, lqk, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_kv_heads, lv, device="cuda", dtype=dtype)
    return q, k, v, _cu_seqlens(q_lens), _cu_seqlens(k_lens)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


class TestMLAPrefillBaseline:
    def test_no_quant_matches_reference(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([37, 64], [37, 64], 4, 4, 192, 128)
        out = mla_prefill_attention(q, k, v, cu_q, cu_k, 64)
        ref = mla_attention_reference(q, k, v, cu_q, cu_k)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)

    def test_gqa_no_quant(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([48, 33], [48, 33], 8, 2, 192, 128, seed=1)
        out = mla_prefill_attention(q, k, v, cu_q, cu_k, 48)
        ref = mla_attention_reference(q, k, v, cu_q, cu_k)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)

    @pytest.mark.parametrize(
        ("num_heads", "num_kv_heads", "lqk", "lv"),
        [(8, 8, 192, 128), (8, 2, 64, 32), (4, 4, 576, 512)],
    )
    def test_dim_sweep_no_quant(self, num_heads, num_kv_heads, lqk, lv):
        q, k, v, cu_q, cu_k = _make_mla_qkv([50], [50], num_heads, num_kv_heads, lqk, lv, seed=2)
        out = mla_prefill_attention(q, k, v, cu_q, cu_k, 50)
        ref = mla_attention_reference(q, k, v, cu_q, cu_k)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)

    def test_causal_suffix_qk(self):
        """Causal with k_len > q_len: Q is the suffix of the KV span."""
        q, k, v, cu_q, cu_k = _make_mla_qkv([16, 8], [80, 40], 4, 4, 192, 128, seed=3)
        out = mla_prefill_attention(q, k, v, cu_q, cu_k, 16)
        ref = mla_attention_reference(q, k, v, cu_q, cu_k)
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)

    def test_lse_matches_logsumexp(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([40], [40], 4, 4, 192, 128, seed=4)
        _, lse = mla_prefill_attention(q, k, v, cu_q, cu_k, 40, causal=False, return_lse=True)
        scale = 192**-0.5
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        expected = torch.logsumexp(scores, dim=-1)  # [H, total_q]
        torch.testing.assert_close(lse, expected, rtol=5e-3, atol=5e-3)

    def test_two_chunk_lse_merge_matches_single_shot(self):
        """Non-causal chunked K/V merged via LSE equals the single-shot result."""
        q, k, v, cu_q, cu_k = _make_mla_qkv([32], [128], 4, 4, 192, 128, seed=5)
        full = mla_prefill_attention(q, k, v, cu_q, cu_k, 32, causal=False)

        outs, lses = [], []
        for lo, hi in ((0, 64), (64, 128)):
            o, lse = mla_prefill_attention(
                q,
                k[lo:hi],
                v[lo:hi],
                cu_q,
                _cu_seqlens([hi - lo]),
                32,
                causal=False,
                return_lse=True,
            )
            outs.append(o.float())
            lses.append(lse)
        lse_all = torch.logaddexp(lses[0], lses[1])
        w0 = torch.exp(lses[0] - lse_all).transpose(0, 1).unsqueeze(-1)  # [total_q, H, 1]
        w1 = torch.exp(lses[1] - lse_all).transpose(0, 1).unsqueeze(-1)
        merged = outs[0] * w0 + outs[1] * w1
        torch.testing.assert_close(merged, full.float(), rtol=5e-3, atol=5e-3)

    def test_empty_kv_rows_are_zero_with_neg_inf_lse(self):
        q, k, v, cu_q, _ = _make_mla_qkv([8], [8], 4, 4, 192, 128, seed=6)
        cu_k_empty = _cu_seqlens([0])
        out, lse = mla_prefill_attention(
            q, k[:0], v[:0], cu_q, cu_k_empty, 8, causal=False, return_lse=True
        )
        assert torch.all(out == 0)
        assert torch.all(torch.isneginf(lse))


class TestMLAPrefillSparse:
    def test_sparse24_no_quant(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([96], [96], 4, 4, 192, 128, seed=7)
        dense = mla_prefill_attention(q, k, v, cu_q, cu_k, 96)
        out = mla_prefill_attention(
            q, k, v, cu_q, cu_k, 96, sparsity_n=2, sparsity_m=4, dense_recent_tokens=16
        )
        ref = mla_attention_reference(
            q, k, v, cu_q, cu_k, sparsity_n=2, sparsity_m=4, dense_recent_tokens=16
        )
        assert not torch.equal(out, dense)  # sparsity actually applied
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)

    @requires_native_e4m3
    def test_sparse24_nvfp4_composition(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([96], [96], 4, 4, 192, 128, seed=8)
        out = mla_prefill_attention(
            q,
            k,
            v,
            cu_q,
            cu_k,
            96,
            q_quant="nvfp4",
            k_quant="nvfp4",
            p_quant="nvfp4",
            v_quant="nvfp4",
            sparsity_n=2,
            sparsity_m=4,
            dense_recent_tokens=16,
        )
        ref = mla_attention_reference(
            q,
            k,
            v,
            cu_q,
            cu_k,
            q_mode="nvfp4",
            k_mode="nvfp4",
            p_mode="nvfp4",
            v_mode="nvfp4",
            sparsity_n=2,
            sparsity_m=4,
            dense_recent_tokens=16,
        )
        assert _cos(out, ref) > 0.98


class TestMLAPrefillQuant:
    @requires_native_e4m3
    @pytest.mark.parametrize("mode", ["fp8", "nvfp4"])
    def test_qkpv_quant_matches_reference(self, mode):
        q, k, v, cu_q, cu_k = _make_mla_qkv([64, 41], [64, 41], 4, 4, 192, 128, seed=9)
        kwargs = {f"{op}_quant": mode for op in ("q", "k", "p", "v")}
        out = mla_prefill_attention(q, k, v, cu_q, cu_k, 64, **kwargs)
        dense = mla_prefill_attention(q, k, v, cu_q, cu_k, 64)
        assert not torch.equal(out, dense)  # quant actually applied
        ref = mla_attention_reference(
            q, k, v, cu_q, cu_k, **{f"{op}_mode": mode for op in ("q", "k", "p", "v")}
        )
        if mode == "fp8":
            torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=2e-2)
        else:
            assert _cos(out, ref) > 0.98

    @requires_native_e4m3
    def test_p_only_nvfp4(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([80], [80], 4, 4, 192, 128, seed=10)
        out = mla_prefill_attention(q, k, v, cu_q, cu_k, 80, p_quant="nvfp4")
        ref = mla_attention_reference(q, k, v, cu_q, cu_k, p_mode="nvfp4")
        torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=2e-2)

    @requires_native_e4m3
    def test_amax_scales_change_results(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([64], [64], 4, 4, 192, 128, seed=11)
        out_default = mla_prefill_attention(q, k, v, cu_q, cu_k, 64, q_quant="fp8")
        out_amax = mla_prefill_attention(q, k, v, cu_q, cu_k, 64, q_quant="fp8", q_amax=4.0)
        assert not torch.equal(out_default, out_amax)
        ref = mla_attention_reference(q, k, v, cu_q, cu_k, q_mode="fp8", q_amax=4.0)
        torch.testing.assert_close(out_amax.float(), ref, rtol=5e-3, atol=2e-2)


class TestMLAPrefillErrors:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"q_quant": "int8"}, "q_quant must be one of"),
            ({"block_n": 24}, "multiples of 16"),
            ({"p_quant": "fp8", "p_amax": -1.0}, "finite positive"),
        ],
    )
    def test_invalid_config(self, kwargs, match):
        q, k, v, cu_q, cu_k = _make_mla_qkv([16], [16], 4, 4, 192, 128)
        with pytest.raises(ValueError, match=match):
            mla_prefill_attention(q, k, v, cu_q, cu_k, 16, **kwargs)

    def test_nvfp4_requires_divisible_qk_dim(self):
        q, k, v, cu_q, cu_k = _make_mla_qkv([16], [16], 4, 4, 40, 32)
        with pytest.raises(ValueError, match="qk_head_dim % 16"):
            mla_prefill_attention(q, k, v, cu_q, cu_k, 16, q_quant="nvfp4")
