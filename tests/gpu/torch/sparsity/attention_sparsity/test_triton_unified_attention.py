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

"""GPU tests for Triton unified attention kernel."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.sparsity.attention_sparsity.kernels import (
    IS_AVAILABLE as TRITON_KERNEL_AVAILABLE,
)

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.sparsity.attention_sparsity.kernels import (
        context_attention_fwd,
        unified_attention,
    )


def _sdpa_reference(q, k, v, b_start_loc, b_seq_len):
    """SDPA causal reference. Supports GQA. Returns [total_tokens, num_heads, dim]."""
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
        ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
        parts.append(ob.permute(0, 2, 1, 3).squeeze(0))
    return torch.cat(parts, dim=0)


def _sparse24_top2(x0, x1, x2, x3):
    """Top-2-of-4 mask (same logic as Triton _sparse24_noabs_ops)."""
    a1, a2, a3 = x0 > x1, x0 > x2, x0 > x3
    a4, a5, a6 = x1 > x2, x1 > x3, x2 > x3
    m0 = (a2 and a3) or (a1 and a2) or (a1 and a3)
    m1 = (not a1 and a5) or (a4 and a5) or (not a1 and a4)
    m2 = (not a2 and not a4) or (not a2 and a6) or (not a4 and a6)
    m3 = (not a3 and not a5) or (not a3 and not a6) or (not a5 and not a6)
    return m0, m1, m2, m3


def _attention_sparse24_ref(q, k, v, scale, bq, ts, skip_diag=True):
    """Reference attention with 2:4 sparsity + diagonal skip. [seq, dim] -> [seq, dim]."""
    n = q.shape[0]
    scores = scale * (q @ k.T)
    scores.masked_fill_(
        torch.triu(torch.ones(n, n, device=scores.device, dtype=torch.bool), 1), float("-inf")
    )
    nqb = (n + bq - 1) // bq
    ntiles = (n + ts - 1) // ts
    for qb in range(nqb):
        qs, qe = qb * bq, min((qb + 1) * bq, n)
        for t in range(ntiles):
            ks, ke = t * ts, min((t + 1) * ts, n)
            if skip_diag and ks < qe and ke > qs:
                continue
            for row in range(qs, qe):
                for g in range((ke - ks) // 4):
                    c = ks + g * 4
                    vals = [scores[row, c + i].item() for i in range(4)]
                    mask = _sparse24_top2(*vals)
                    for i in range(4):
                        if not mask[i]:
                            scores[row, c + i] = float("-inf")
    return F.softmax(scores.float(), dim=-1).to(q.dtype) @ v


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


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestUnifiedAttentionVsSdpa:
    """Triton unified attention matches PyTorch SDPA for prefill and decode."""

    @pytest.mark.parametrize(
        ("dtype", "num_heads", "num_kv_heads", "head_dim", "tol"),
        [
            (torch.float32, 2, 2, 32, 1e-2),
            (torch.float16, 4, 2, 64, 2e-2),
        ],
        ids=["fp32_mha", "fp16_gqa"],
    )
    def test_prefill_matches_sdpa(self, dtype, num_heads, num_kv_heads, head_dim, tol):
        """Prefill via context_attention_fwd matches SDPA (variable-length batch)."""
        seq_lens = [8, 12]
        total = sum(seq_lens)
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(123)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        locs = torch.tensor([0, seq_lens[0]], device="cuda", dtype=torch.int32)
        lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            b_start_loc=locs,
            b_seq_len=lens,
            max_input_len=max(seq_lens),
            is_causal=True,
            softmax_scale=scale,
        )
        torch.testing.assert_close(o, _sdpa_reference(q, k, v, locs, lens), rtol=tol, atol=tol)

    def test_cross_attention_matches_sdpa(self):
        """Non-causal cross-attention: different Q and K/V lengths, matches SDPA."""
        seq_q, seq_k = 6, 10
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(501)
        q = torch.randn(seq_q, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(seq_k, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v = torch.randn(seq_k, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_q], device="cuda", dtype=torch.int32),
            max_input_len=seq_q,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len_k=torch.tensor([seq_k], device="cuda", dtype=torch.int32),
            max_input_len_k=seq_k,
        )

        # Reference: SDPA non-causal
        q_ref = q.unsqueeze(0).permute(0, 2, 1, 3)  # [1, heads, seq_q, dim]
        k_ref = k.unsqueeze(0).permute(0, 2, 1, 3)
        v_ref = v.unsqueeze(0).permute(0, 2, 1, 3)
        k_ref = k_ref.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_ref = v_ref.repeat_interleave(num_heads // num_kv_heads, dim=1)
        o_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
        o_ref = o_ref.permute(0, 2, 1, 3).squeeze(0)

        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    def test_decode_matches_sdpa(self):
        """Decode with GQA paged KV cache matches per-sample SDPA."""
        batch, ctx_lens = 2, [4, 8]
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        block_size = ((max(ctx_lens) + 1 + 31) // 32) * 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(103)
        q_dec = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float32)
        kc = torch.randn(
            batch, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float32
        )
        vc = torch.randn(
            batch, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float32
        )
        for i, cl in enumerate(ctx_lens):
            kc[i, cl + 1 :] = 0
            vc[i, cl + 1 :] = 0

        bt = torch.arange(batch, device="cuda", dtype=torch.int32).unsqueeze(1)
        cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        sk = torch.tensor([c + 1 for c in ctx_lens], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q_dec)

        unified_attention(
            q=q_dec,
            k=kc,
            v=vc,
            out=out,
            cu_seqlens_q=cu,
            max_seqlen_q=1,
            seqused_k=sk,
            max_seqlen_k=block_size,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=bt,
        )

        for i in range(batch):
            sl = ctx_lens[i] + 1
            qb = q_dec[i : i + 1].unsqueeze(2)
            kb = kc[i, :sl].unsqueeze(0).permute(0, 2, 1, 3)
            vb = vc[i, :sl].unsqueeze(0).permute(0, 2, 1, 3)
            kb = kb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vb = vb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).squeeze(2)
            torch.testing.assert_close(out[i : i + 1], ref, rtol=1e-2, atol=1e-2)

    def test_prefill_decode_consistency(self):
        """Last token of prefill matches decode output for the same sequence."""
        seq_len = 8
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        block_size = ((seq_len + 15) // 16) * 16
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(104)
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        # Prefill
        o_pf = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_pf,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )

        # Decode (last token as query, full KV in cache)
        kc = torch.zeros(1, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        vc = torch.zeros(1, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        kc[0, :seq_len] = k
        vc[0, :seq_len] = v
        o_dec = torch.empty_like(q[:1])
        unified_attention(
            q=q[-1:],
            k=kc,
            v=vc,
            out=o_dec,
            cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_seqlen_k=block_size,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        )

        torch.testing.assert_close(o_pf[-1:], o_dec, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparse24Attention:
    """2:4 sparse attention applied inside the Triton kernel."""

    def test_sparse24_output_differs_from_dense(self):
        """Sparse24 enabled produces different (but valid) output vs dense."""
        seq_lens, total = [48, 64], 112
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(789)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        locs = torch.tensor([0, seq_lens[0]], device="cuda", dtype=torch.int32)
        lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

        kw = {
            "b_start_loc": locs,
            "b_seq_len": lens,
            "max_input_len": max(seq_lens),
            "is_causal": True,
            "softmax_scale": scale,
        }

        o_dense = torch.empty_like(q)
        context_attention_fwd(q, k, v, o_dense, apply_sparse24=False, **kw)
        o_sparse = torch.empty_like(q)
        context_attention_fwd(
            q, k, v, o_sparse, apply_sparse24=True, skip_diagonal_blocks=True, **kw
        )

        assert not torch.equal(o_dense, o_sparse), "Sparse should differ from dense"
        assert not torch.isnan(o_sparse).any() and not torch.isinf(o_sparse).any()

    def test_sparse24_matches_reference(self):
        """Sparse24 with GQA (4 q-heads, 2 kv-heads) matches Python reference."""
        seq_len = 32
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        nqkv = num_heads // num_kv_heads
        scale = 1.0 / (head_dim**0.5)
        bq, ts = 16 // nqkv, 32

        torch.manual_seed(303)
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        o_tri = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_tri,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
            apply_sparse24=True,
            skip_diagonal_blocks=True,
        )

        o_ref = torch.empty_like(q)
        for h in range(num_heads):
            o_ref[:, h] = _attention_sparse24_ref(
                q[:, h],
                k[:, h // nqkv],
                v[:, h // nqkv],
                scale,
                bq,
                ts,
            )

        torch.testing.assert_close(o_tri, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseAttentionIntegration:
    """HF model + mtsa.sparsify integration."""

    def test_triton_forward_and_generate(self, tiny_llama_dir):
        """modelopt_triton attention: prefill logits valid, generate produces tokens."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        assert not torch.isnan(logits).any() and not torch.isinf(logits).any()

        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id
            )
        assert out.shape[1] == ids.shape[1] + 5

    def test_sparsify_sparse24_produces_valid_output(self, tiny_llama_dir):
        """mtsa.sparsify(model, SPARSE24_TRITON) forward produces valid logits."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import modelopt.torch.sparsity.attention_sparsity as mtsa
        from modelopt.torch.sparsity.attention_sparsity.config import SPARSE24_TRITON

        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model = mtsa.sparsify(model, SPARSE24_TRITON)
        model.eval()

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = tok("Hello world", return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            logits = model(input_ids=ids).logits
        assert not torch.isnan(logits).any() and not torch.isinf(logits).any()
