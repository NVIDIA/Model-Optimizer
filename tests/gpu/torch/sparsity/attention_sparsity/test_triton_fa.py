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

"""GPU tests for Triton flash attention kernel."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    import triton
    import triton.language as tl

    from modelopt.torch.kernels import attention, register_triton_attention
    from modelopt.torch.kernels.triton_fa import _apply_sparse_nm_to_qk_tile

    if register_triton_attention is not None:
        register_triton_attention()

    @triton.jit
    def _test_apply_sparse_nm(In, Out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              SPARSITY_N: tl.constexpr, SPARSITY_M: tl.constexpr):
        """Test wrapper: apply N:M sparsity to a tile and store result."""
        offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        qk = tl.load(In + offs)
        tl.store(Out + offs, _apply_sparse_nm_to_qk_tile(qk, BLOCK_M, BLOCK_N, SPARSITY_N, SPARSITY_M))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sdpa_reference(q, k, v, b_start_loc, b_seq_len, is_causal=True):
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


def _make_qkv(total, num_heads, num_kv_heads, head_dim, device="cuda", dtype=torch.float16):
    """Create packed Q, K, V tensors."""
    q = torch.randn(total, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, num_kv_heads, head_dim, device=device, dtype=dtype)
    return q, k, v


def _make_varlen_meta(seq_lens, device="cuda"):
    """Create b_start_loc and b_seq_len from a list of sequence lengths."""
    b_seq_len = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(len(seq_lens), device=device, dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
    return b_start_loc, b_seq_len


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------
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
class TestForward:
    """Forward pass correctness for dense and sparse attention."""

    @pytest.mark.parametrize(
        ("dtype", "num_heads", "num_kv_heads", "head_dim"),
        [
            (torch.float32, 2, 2, 32),
            (torch.float16, 4, 2, 64),
            (torch.bfloat16, 4, 2, 128),
        ],
        ids=["fp32_mha", "fp16_gqa", "bf16_gqa_hdim128"],
    )
    def test_prefill_matches_sdpa(self, dtype, num_heads, num_kv_heads, head_dim):
        """Dense prefill matches SDPA."""
        seq_lens = [8, 12]
        total = sum(seq_lens)
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(123)
        q, k, v = _make_qkv(total, num_heads, num_kv_heads, head_dim, dtype=dtype)
        locs, lens = _make_varlen_meta(seq_lens)

        o = attention(q, k, v, locs, lens, max(seq_lens), softmax_scale=scale)
        torch.testing.assert_close(o, _sdpa_reference(q, k, v, locs, lens), rtol=1e-3, atol=1e-3)

    def test_decode_matches_sdpa(self):
        """Dense decode matches SDPA."""
        batch = 2
        seq_lens_k = [5, 9]
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(103)
        q_flat = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float32)
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

        out = attention(
            q_flat,
            k_flat,
            v_flat,
            b_start_loc_q,
            b_seq_len_q,
            1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max(seq_lens_k),
        )

        for i in range(batch):
            sl = seq_lens_k[i]
            s = cumsum[i]
            qb = q_flat[i : i + 1].unsqueeze(2)
            kb = k_flat[s : s + sl].unsqueeze(0).permute(0, 2, 1, 3)
            vb = v_flat[s : s + sl].unsqueeze(0).permute(0, 2, 1, 3)
            kb = kb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vb = vb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).squeeze(2)
            torch.testing.assert_close(out[i : i + 1], ref, rtol=1e-3, atol=1e-3)

    def test_sparse_disabled_matches_dense(self):
        """sparsity_n=0 produces bit-identical output to default (dense)."""
        seq_lens = [128, 128]
        total = sum(seq_lens)
        scale = 1.0 / (64**0.5)

        torch.manual_seed(99)
        q, k, v = _make_qkv(total, 4, 2, 64)
        locs, lens = _make_varlen_meta(seq_lens)

        out_dense = attention(q, k, v, locs, lens, 128, softmax_scale=scale)
        out_n0 = attention(q, k, v, locs, lens, 128, softmax_scale=scale, sparsity_n=0)
        assert torch.equal(out_dense, out_n0)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestBackward:
    """Backward pass gradient correctness for dense and sparse attention."""

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
        return dq.detach(), dk.detach(), dv.detach()

    # --- Dense backward vs SDPA ---
    def test_dense_causal_matches_sdpa(self):
        """dQ, dK, dV match SDPA for causal self-attention."""
        seq_len, num_heads, num_kv_heads, head_dim = 16, 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        q, k, v = _make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = _make_varlen_meta([seq_len])

        attention(q, k, v, locs, lens, seq_len, softmax_scale=scale).sum().backward()
        dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(q.detach(), k.detach(), v.detach(), scale)

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    def test_dense_gqa_matches_sdpa(self):
        """Dense backward with GQA (4 q-heads, 2 kv-heads), seq_len=256."""
        seq_len, num_heads, num_kv_heads, head_dim = 256, 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(43)
        q, k, v = _make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = _make_varlen_meta([seq_len])

        attention(q, k, v, locs, lens, seq_len, softmax_scale=scale).sum().backward()
        dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(q.detach(), k.detach(), v.detach(), scale)

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    def test_dense_multi_batch_variable_length(self):
        """Multi-batch variable-length backward matches per-sample SDPA."""
        seq_lens = [8, 12]
        total = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(45)
        q, k, v = _make_qkv(total, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = _make_varlen_meta(seq_lens)

        attention(q, k, v, locs, lens, max(seq_lens), softmax_scale=scale).sum().backward()

        dq_ref = torch.zeros_like(q)
        dk_ref = torch.zeros_like(k)
        dv_ref = torch.zeros_like(v)
        for b in range(len(seq_lens)):
            s, n = int(locs[b].item()), seq_lens[b]
            dq_b, dk_b, dv_b = self._sdpa_backward_ref(
                q.detach()[s : s + n],
                k.detach()[s : s + n],
                v.detach()[s : s + n],
                scale,
            )
            dq_ref[s : s + n] = dq_b
            dk_ref[s : s + n] = dk_b
            dv_ref[s : s + n] = dv_b

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    def test_dense_longer_sequences(self):
        """Dense backward with seq_len=512, GQA, exercises multi-tile loops."""
        seq_len, num_heads, num_kv_heads, head_dim = 512, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(49)
        q, k, v = _make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = _make_varlen_meta([seq_len])

        attention(q, k, v, locs, lens, seq_len, softmax_scale=scale).sum().backward()
        dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(q.detach(), k.detach(), v.detach(), scale)

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    # --- Sparse backward sanity checks ---

    @pytest.mark.parametrize(
        ("n", "m"),
        [(2, 4), (4, 8)],
        ids=["2:4", "4:8"],
    )
    def test_sparse_gradients_finite(self, n, m):
        """Backward with N:M sparsity produces finite, non-zero gradients."""
        seq_len, num_heads, num_kv_heads, head_dim = 128, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(55)
        q, k, v = _make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = _make_varlen_meta([seq_len])

        attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            sparsity_n=n,
            sparsity_m=m,
        ).sum().backward()

        for name, grad in [("dQ", q.grad), ("dK", k.grad), ("dV", v.grad)]:
            assert grad is not None, f"{name} is None for {n}:{m}"
            assert not torch.isnan(grad).any(), f"NaN in {name} for {n}:{m}"
            assert not torch.isinf(grad).any(), f"Inf in {name} for {n}:{m}"
            assert grad.abs().sum() > 0, f"{name} is all zeros for {n}:{m}"

    def test_sparse_gradients_differ_from_dense(self):
        """Gradients with 2:4 sparsity should differ from dense gradients."""
        seq_len, num_heads, num_kv_heads, head_dim = 256, 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(66)
        q, k, v = _make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        locs, lens = _make_varlen_meta([seq_len])

        q_d = q.clone().requires_grad_(True)
        k_d = k.clone().requires_grad_(True)
        v_d = v.clone().requires_grad_(True)
        attention(q_d, k_d, v_d, locs, lens, seq_len, softmax_scale=scale).sum().backward()

        q_s = q.clone().requires_grad_(True)
        k_s = k.clone().requires_grad_(True)
        v_s = v.clone().requires_grad_(True)
        attention(
            q_s,
            k_s,
            v_s,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            sparsity_n=2,
            sparsity_m=4,
        ).sum().backward()

        assert not torch.allclose(q_d.grad, q_s.grad, atol=1e-3), (
            "dQ same with and without sparsity"
        )
        assert not torch.allclose(k_d.grad, k_s.grad, atol=1e-3), (
            "dK same with and without sparsity"
        )


# ---------------------------------------------------------------------------
# N:M sparsity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseNM:
    """N:M structured sparsity behavior on attention scores (prefill only)."""

    def _make_inputs(self, batch=2, seq_len=256, num_heads=4, num_kv_heads=2, head_dim=64):
        total = batch * seq_len
        torch.manual_seed(99)
        q, k, v = _make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = _make_varlen_meta([seq_len] * batch)
        return q, k, v, locs, lens

    @pytest.mark.parametrize(
        ("n", "m"),
        [(1, 4), (2, 4), (3, 4), (1, 8), (2, 8), (4, 8)],
        ids=["1:4", "2:4", "3:4", "1:8", "2:8", "4:8"],
    )
    def test_output_shape(self, n, m):
        """Output shape matches Q shape for all N:M patterns."""
        q, k, v, locs, lens = self._make_inputs()
        out = attention(
            q, k, v, locs, lens, 256, softmax_scale=1.0 / 8.0, sparsity_n=n, sparsity_m=m
        )
        assert out.shape == q.shape

    @pytest.mark.parametrize(
        ("n", "m"),
        [(1, 4), (2, 4), (3, 4), (1, 8), (2, 8), (4, 8)],
        ids=["1:4", "2:4", "3:4", "1:8", "2:8", "4:8"],
    )
    def test_no_nan(self, n, m):
        """All N:M patterns produce finite output."""
        q, k, v, locs, lens = self._make_inputs()
        out = attention(
            q, k, v, locs, lens, 256, softmax_scale=1.0 / 8.0, sparsity_n=n, sparsity_m=m
        )
        assert not torch.isnan(out).any(), f"NaN in output for {n}:{m}"
        assert not torch.isinf(out).any(), f"Inf in output for {n}:{m}"

    @pytest.mark.parametrize(
        ("n", "m"),
        [(1, 4), (2, 4), (1, 8), (4, 8)],
        ids=["1:4", "2:4", "1:8", "4:8"],
    )
    def test_sparse_differs_from_dense(self, n, m):
        """Sparse output should differ from dense for long sequences."""
        q, k, v, locs, lens = self._make_inputs(seq_len=512)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 512, softmax_scale=scale)
        out_sparse = attention(
            q, k, v, locs, lens, 512, softmax_scale=scale, sparsity_n=n, sparsity_m=m
        )
        assert not torch.allclose(out_sparse, out_dense, atol=1e-3)

    @pytest.mark.parametrize(
        ("n_values", "m"),
        [([1, 2, 3], 4), ([1, 2, 4], 8)],
        ids=["m4", "m8"],
    )
    def test_more_sparsity_more_error(self, n_values, m):
        """Keeping more elements should deviate less from dense (monotonic decreasing error)."""
        q, k, v, locs, lens = self._make_inputs(seq_len=512)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 512, softmax_scale=scale)
        errors = []
        for n in n_values:
            out = attention(
                q, k, v, locs, lens, 512, softmax_scale=scale, sparsity_n=n, sparsity_m=m
            )
            errors.append((out - out_dense).abs().mean().item())
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Errors not monotonically decreasing for M={m}: "
                + ", ".join(f"{n}:{m}={e:.6f}" for n, e in zip(n_values, errors))
            )

    @pytest.mark.parametrize(
        ("n", "m"),
        [(2, 4), (4, 8)],
        ids=["2:4", "4:8"],
    )
    def test_dense_window_preserves_local(self, n, m):
        """Large dense_window_blocks makes sparse output closer to dense."""
        q, k, v, locs, lens = self._make_inputs(seq_len=256)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 256, softmax_scale=scale)
        out_small = attention(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            sparsity_n=n,
            sparsity_m=m,
            dense_window_blocks=1,
        )
        out_large = attention(
            q,
            k,
            v,
            locs,
            lens,
            256,
            softmax_scale=scale,
            sparsity_n=n,
            sparsity_m=m,
            dense_window_blocks=100,
        )
        err_small = (out_small - out_dense).abs().mean().item()
        err_large = (out_large - out_dense).abs().mean().item()
        assert err_large < err_small

    @pytest.mark.parametrize(
        ("n", "m"),
        [(1, 4), (2, 4), (3, 4), (1, 8), (2, 8), (4, 8)],
        ids=["1:4", "2:4", "3:4", "1:8", "2:8", "4:8"],
    )
    def test_sparsity_structure(self, n, m):
        """Verify N:M structure: exactly N kept per group of M."""
        BM, BN = 32, 64
        torch.manual_seed(88)
        tile = torch.randn(BM, BN, device="cuda", dtype=torch.float32)
        out = torch.empty_like(tile)
        _test_apply_sparse_nm[(1,)](tile, out, BLOCK_M=BM, BLOCK_N=BN, SPARSITY_N=n, SPARSITY_M=m)

        kept = (out.reshape(BM, BN // m, m) != float("-inf")).sum(dim=-1)
        assert (kept == n).all(), f"Expected {n} kept per group of {m}, got min={kept.min()}, max={kept.max()}"


# ---------------------------------------------------------------------------
# HuggingFace integration
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestHFIntegration:
    """HF model integration with Triton attention backend."""

    def test_triton_matches_eager(self, tiny_llama_dir):
        """Triton attention produces same logits and generated tokens as eager."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")

        model_eager = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_eager.eval()
        with torch.no_grad():
            logits_eager = model_eager(input_ids=ids).logits
            out_eager = model_eager.generate(
                ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        del model_eager

        model_triton = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_triton.eval()
        with torch.no_grad():
            logits_triton = model_triton(input_ids=ids).logits
            out_triton = model_triton.generate(
                ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

        torch.testing.assert_close(logits_triton, logits_eager, rtol=2e-2, atol=2e-2)
        assert torch.equal(out_triton, out_eager), (
            f"Generated tokens differ:\n  eager:  {out_eager}\n  triton: {out_triton}"
        )

    def test_triton_padded_batch(self, tiny_llama_dir):
        """Padded batch produces valid logits."""
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
        tok.padding_side = "right"

        inputs = tok(
            ["Hello world", "The capital of France is Paris and"],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        assert not torch.isnan(logits).any() and not torch.isinf(logits).any()
