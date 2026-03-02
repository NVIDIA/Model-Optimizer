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
    from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention_fwd


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


def _get_prefill_block_size():
    """Return the BLOCK size the prefill kernel uses on the current GPU."""
    cap = torch.cuda.get_device_capability()
    return 128 if cap[0] >= 8 else 64


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
        """Decode via context_attention_fwd(is_causal=False) matches per-sample SDPA."""
        batch = 2
        seq_lens_k = [5, 9]  # KV lengths (context + current token)
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(103)
        # Q: one token per batch element -> flat [batch, num_heads, head_dim]
        q_flat = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float32)

        # K/V: variable-length, packed into flat tensors
        total_kv = sum(seq_lens_k)
        k_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        cumsum = [0]
        for sl in seq_lens_k:
            cumsum.append(cumsum[-1] + sl)
        b_start_loc_q = torch.arange(batch, device="cuda", dtype=torch.int32)
        b_seq_len_q = torch.ones(batch, device="cuda", dtype=torch.int32)
        b_start_loc_k = torch.tensor(cumsum[:-1], device="cuda", dtype=torch.int32)
        b_seq_len_k = torch.tensor(seq_lens_k, device="cuda", dtype=torch.int32)

        out = torch.empty_like(q_flat)
        context_attention_fwd(
            q_flat,
            k_flat,
            v_flat,
            out,
            b_start_loc=b_start_loc_q,
            b_seq_len=b_seq_len_q,
            max_input_len=1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max(seq_lens_k),
        )

        for i in range(batch):
            sl = seq_lens_k[i]
            s = cumsum[i]
            qb = q_flat[i : i + 1].unsqueeze(2)  # [1, heads, 1, dim]
            kb = k_flat[s : s + sl].unsqueeze(0).permute(0, 2, 1, 3)
            vb = v_flat[s : s + sl].unsqueeze(0).permute(0, 2, 1, 3)
            kb = kb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vb = vb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).squeeze(2)
            torch.testing.assert_close(out[i : i + 1], ref, rtol=1e-2, atol=1e-2)

    def test_prefill_decode_consistency(self):
        """Last token of prefill matches decode output for the same sequence."""
        seq_len = 8
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(104)
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        # Prefill (causal)
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

        # Decode: last token as query, full K/V (non-causal to attend to all)
        q_dec = q[-1:].contiguous()
        o_dec = torch.empty_like(q_dec)
        context_attention_fwd(
            q_dec,
            k,
            v,
            o_dec,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([1], device="cuda", dtype=torch.int32),
            max_input_len=1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len_k=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len_k=seq_len,
        )

        torch.testing.assert_close(o_pf[-1:], o_dec, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparse24Attention:
    """2:4 sparse attention applied inside the Triton kernel."""

    def test_sparse24_output_differs_from_dense(self):
        """Sparse24 enabled produces different (but valid) output vs dense."""
        block = _get_prefill_block_size()
        seq_lens = [block * 2, block * 3]
        total = sum(seq_lens)
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
        block = _get_prefill_block_size()
        seq_len = block * 2 + block // 2  # ensure non-trivial diagonal + off-diagonal tiles
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        nqkv = num_heads // num_kv_heads
        scale = 1.0 / (head_dim**0.5)
        bq, ts = block, block

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

        torch.testing.assert_close(o_tri, o_ref, rtol=5e-2, atol=5e-2)


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


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestBackward:
    """Backward pass gradient correctness tests."""

    def _sdpa_backward_ref(self, q, k, v, scale, is_causal=True):
        """Run SDPA forward+backward, return output and gradients."""
        q_ref = q.clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        k_ref = k.clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        v_ref = v.clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        num_q, num_kv = q_ref.shape[1], k_ref.shape[1]
        if num_q != num_kv:
            r = num_q // num_kv
            k_exp = k_ref.repeat_interleave(r, dim=1)
            v_exp = v_ref.repeat_interleave(r, dim=1)
        else:
            k_exp, v_exp = k_ref, v_ref
        o_ref = F.scaled_dot_product_attention(
            q_ref, k_exp, v_exp, is_causal=is_causal, scale=scale
        )
        o_ref.sum().backward()
        dq = q_ref.grad.permute(0, 2, 1, 3).squeeze(0)
        dk = k_ref.grad.permute(0, 2, 1, 3).squeeze(0)
        dv = v_ref.grad.permute(0, 2, 1, 3).squeeze(0)
        return o_ref.permute(0, 2, 1, 3).squeeze(0).detach(), dq.detach(), dk.detach(), dv.detach()

    def test_backward_causal_matches_sdpa(self):
        """dQ, dK, dV match SDPA backward for causal self-attention."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        seq_len = 16
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = context_attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
            q.detach(), k.detach(), v.detach(), scale, is_causal=True
        )

        torch.testing.assert_close(q.grad, dq_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.grad, dk_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v.grad, dv_ref, rtol=1e-2, atol=1e-2)

    def test_backward_gqa(self):
        """Backward with GQA (4 q-heads, 2 kv-heads) matches SDPA."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        seq_len = 16
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(43)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = context_attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
            q.detach(), k.detach(), v.detach(), scale, is_causal=True
        )

        torch.testing.assert_close(q.grad, dq_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.grad, dk_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v.grad, dv_ref, rtol=1e-2, atol=1e-2)

    def test_backward_sparse24_finite(self):
        """Backward with sparse24 produces finite, non-zero gradients."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        block = _get_prefill_block_size()
        seq_len = block * 2 + block // 2
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(44)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = context_attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
            apply_sparse24=True,
            skip_diagonal_blocks=True,
        )
        o.sum().backward()

        for name, grad in [("dQ", q.grad), ("dK", k.grad), ("dV", v.grad)]:
            assert grad is not None, f"{name} gradient is None"
            assert not torch.isnan(grad).any(), f"{name} has NaN"
            assert not torch.isinf(grad).any(), f"{name} has Inf"
            assert grad.abs().sum() > 0, f"{name} is all zeros"

    def test_backward_multi_batch_variable_length(self):
        """Multi-batch variable-length causal backward matches per-sample SDPA."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        seq_lens = [8, 12]
        total = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(45)
        q = torch.randn(
            total, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        locs = torch.tensor([0, seq_lens[0]], device="cuda", dtype=torch.int32)
        lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

        o = context_attention(
            q,
            k,
            v,
            b_start_loc=locs,
            b_seq_len=lens,
            max_input_len=max(seq_lens),
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        # Per-sample SDPA reference
        dq_ref = torch.zeros_like(q)
        dk_ref = torch.zeros_like(k)
        dv_ref = torch.zeros_like(v)
        for b in range(len(seq_lens)):
            s, n = int(locs[b].item()), seq_lens[b]
            _, dq_b, dk_b, dv_b = self._sdpa_backward_ref(
                q.detach()[s : s + n],
                k.detach()[s : s + n],
                v.detach()[s : s + n],
                scale,
                is_causal=True,
            )
            dq_ref[s : s + n] = dq_b
            dk_ref[s : s + n] = dk_b
            dv_ref[s : s + n] = dv_b

        torch.testing.assert_close(q.grad, dq_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.grad, dk_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v.grad, dv_ref, rtol=1e-2, atol=1e-2)

    def test_backward_cross_attention(self):
        """Non-causal cross-attention backward with different Q and K/V lengths."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        seq_q, seq_k = 6, 10
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(46)
        q = torch.randn(
            seq_q, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_k, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_k, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = context_attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_q], device="cuda", dtype=torch.int32),
            max_input_len=seq_q,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len_k=torch.tensor([seq_k], device="cuda", dtype=torch.int32),
            max_input_len_k=seq_k,
        )
        o.sum().backward()

        # SDPA reference (non-causal, GQA-expanded)
        q_ref = q.detach().clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        k_ref = k.detach().clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        v_ref = v.detach().clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        r = num_heads // num_kv_heads
        k_exp = k_ref.repeat_interleave(r, dim=1)
        v_exp = v_ref.repeat_interleave(r, dim=1)
        o_ref = F.scaled_dot_product_attention(q_ref, k_exp, v_exp, is_causal=False, scale=scale)
        o_ref.sum().backward()

        torch.testing.assert_close(
            q.grad, q_ref.grad.permute(0, 2, 1, 3).squeeze(0), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            k.grad, k_ref.grad.permute(0, 2, 1, 3).squeeze(0), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            v.grad, v_ref.grad.permute(0, 2, 1, 3).squeeze(0), rtol=1e-2, atol=1e-2
        )

    def test_backward_sparse24_matches_reference(self):
        """Sparse24 backward dQ/dK/dV match a Python reference with manual 2:4 masking."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        block = _get_prefill_block_size()
        seq_len = block * 2 + block // 2
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)
        bq, ts = block, block

        torch.manual_seed(47)
        q_data = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k_data = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v_data = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        # Triton backward
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        o = context_attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
            apply_sparse24=True,
            skip_diagonal_blocks=True,
        )
        o.sum().backward()

        # Python reference backward (per head, using _attention_sparse24_ref logic)
        dq_ref = torch.zeros_like(q_data)
        dk_ref = torch.zeros_like(k_data)
        dv_ref = torch.zeros_like(v_data)
        for h in range(num_heads):
            kv_h = h // (num_heads // num_kv_heads)
            q_h = q_data[:, h].clone().requires_grad_(True)
            k_h = k_data[:, kv_h].clone().requires_grad_(True)
            v_h = v_data[:, kv_h].clone().requires_grad_(True)
            o_h = _attention_sparse24_ref(q_h, k_h, v_h, scale, bq, ts)
            o_h.sum().backward()
            dq_ref[:, h] = q_h.grad
            dk_ref[:, kv_h] += k_h.grad
            dv_ref[:, kv_h] += v_h.grad

        torch.testing.assert_close(q.grad, dq_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.grad, dk_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v.grad, dv_ref, rtol=1e-2, atol=1e-2)

    def test_backward_matches_sdpa_all_grads(self):
        """All three gradients match SDPA across multiple configs (smoke test)."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        configs = [
            (4, 2, 2, 16, True),  # small causal
            (8, 4, 2, 32, True),  # GQA causal
            (6, 2, 2, 32, False),  # non-causal
        ]
        for seq_len, num_heads, num_kv_heads, head_dim, is_causal in configs:
            scale = 1.0 / (head_dim**0.5)
            torch.manual_seed(48)
            q = torch.randn(
                seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
            )
            k = torch.randn(
                seq_len,
                num_kv_heads,
                head_dim,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            v = torch.randn(
                seq_len,
                num_kv_heads,
                head_dim,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )

            o = context_attention(
                q,
                k,
                v,
                b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
                b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
                max_input_len=seq_len,
                is_causal=is_causal,
                softmax_scale=scale,
            )
            o.sum().backward()

            _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
                q.detach(), k.detach(), v.detach(), scale, is_causal=is_causal
            )
            tag = f"seq={seq_len},heads={num_heads}/{num_kv_heads},causal={is_causal}"
            torch.testing.assert_close(q.grad, dq_ref, rtol=1e-2, atol=1e-2, msg=f"dQ {tag}")
            torch.testing.assert_close(k.grad, dk_ref, rtol=1e-2, atol=1e-2, msg=f"dK {tag}")
            torch.testing.assert_close(v.grad, dv_ref, rtol=1e-2, atol=1e-2, msg=f"dV {tag}")

    def test_backward_longer_sequences(self):
        """Backward with seq_len=256 exercises multi-tile loops (BLOCK=128)."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        seq_len = 256
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(49)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = context_attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
            q.detach(), k.detach(), v.detach(), scale, is_causal=True
        )

        torch.testing.assert_close(q.grad, dq_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.grad, dk_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v.grad, dv_ref, rtol=1e-2, atol=1e-2)

    def test_forward_backward_matches_forward_only(self):
        """context_attention forward output matches context_attention_fwd."""
        from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention

        seq_len = 16
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(50)
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        locs = torch.tensor([0], device="cuda", dtype=torch.int32)
        lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

        # Forward-only path
        o_fwd = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_fwd,
            b_start_loc=locs,
            b_seq_len=lens,
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )

        # Forward+backward path (just use forward output)
        o_bwd = context_attention(
            q,
            k,
            v,
            b_start_loc=locs,
            b_seq_len=lens,
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )

        torch.testing.assert_close(o_fwd, o_bwd, rtol=1e-5, atol=1e-5)
