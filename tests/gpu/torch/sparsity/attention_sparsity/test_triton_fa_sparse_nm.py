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

# ruff: noqa: N803 — Triton JIT wrapper uses uppercase for constexpr and tensor args

"""GPU tests for N:M sparse softmax on the Triton flash attention kernel."""

import pytest
import torch
from conftest import make_qkv, make_varlen_meta

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    import triton
    import triton.language as tl

    from modelopt.torch.kernels import attention
    from modelopt.torch.kernels.triton_fa import _apply_sparse_nm_to_qk_tile

    @triton.jit
    def _test_apply_sparse_nm(
        In,
        Out,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPARSITY_N: tl.constexpr,
        SPARSITY_M: tl.constexpr,
    ):
        """Test wrapper: apply N:M sparsity to a tile and store result."""
        offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        qk = tl.load(In + offs)
        tl.store(
            Out + offs,
            _apply_sparse_nm_to_qk_tile(qk, BLOCK_M, BLOCK_N, SPARSITY_N, SPARSITY_M),
        )


# ---------------------------------------------------------------------------
# N:M sparsity behavior (prefill only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseNM:
    """N:M sparse softmax behavior on attention scores."""

    def _make_inputs(self, batch=2, seq_len=256, num_heads=4, num_kv_heads=2, head_dim=64):
        total = batch * seq_len
        torch.manual_seed(99)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len] * batch)
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
        """Large dense_window_size makes sparse output closer to dense."""
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
            dense_window_size=64,
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
            dense_window_size=100000,
        )
        err_small = (out_small - out_dense).abs().mean().item()
        err_large = (out_large - out_dense).abs().mean().item()
        assert err_large < err_small

    @pytest.mark.parametrize(
        ("n", "m"),
        [(2, 4), (4, 8)],
        ids=["2:4", "4:8"],
    )
    def test_sink_tokens_preserve_early_kv(self, n, m):
        """num_sink_tokens keeps early KV positions dense, reducing error vs fully sparse."""
        q, k, v, locs, lens = self._make_inputs(seq_len=512)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 512, softmax_scale=scale)
        out_no_sink = attention(
            q,
            k,
            v,
            locs,
            lens,
            512,
            softmax_scale=scale,
            sparsity_n=n,
            sparsity_m=m,
            num_sink_tokens=0,
        )
        out_with_sink = attention(
            q,
            k,
            v,
            locs,
            lens,
            512,
            softmax_scale=scale,
            sparsity_n=n,
            sparsity_m=m,
            num_sink_tokens=128,
        )
        err_no_sink = (out_no_sink - out_dense).abs().mean().item()
        err_with_sink = (out_with_sink - out_dense).abs().mean().item()
        assert err_with_sink < err_no_sink, (
            f"Sink tokens should reduce error: no_sink={err_no_sink:.6f}, with_sink={err_with_sink:.6f}"
        )

    # NOTE: N:M sparse attention is for prefill only, not decode.


# ---------------------------------------------------------------------------
# Sparsity tile structure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseTileStructure:
    """Direct unit tests for _apply_sparse_nm_to_qk_tile via wrapper kernel."""

    @pytest.mark.parametrize(
        ("n", "m"),
        [(1, 4), (2, 4), (3, 4), (1, 8), (2, 8), (4, 8)],
        ids=["1:4", "2:4", "3:4", "1:8", "2:8", "4:8"],
    )
    def test_sparsity_structure(self, n, m):
        """Verify N:M structure: exactly N kept per group of M."""
        bm, bn = 32, 64
        torch.manual_seed(88)
        tile = torch.randn(bm, bn, device="cuda", dtype=torch.float32)
        out = torch.empty_like(tile)
        _test_apply_sparse_nm[(1,)](tile, out, BLOCK_M=bm, BLOCK_N=bn, SPARSITY_N=n, SPARSITY_M=m)

        kept = (out.reshape(bm, bn // m, m) != float("-inf")).sum(dim=-1)
        assert (kept == n).all(), (
            f"Expected {n} kept per group of {m}, got min={kept.min()}, max={kept.max()}"
        )

    @pytest.mark.parametrize(
        ("n", "m"),
        [(2, 4), (4, 8)],
        ids=["2:4", "4:8"],
    )
    def test_sparsity_structure_ties(self, n, m):
        """M=4 keeps exactly N on ties; M=8 (tl.sort) may keep >= N on ties."""
        bm, bn = 32, 64
        tile = torch.ones(bm, bn, device="cuda", dtype=torch.float32)
        out = torch.empty_like(tile)
        _test_apply_sparse_nm[(1,)](tile, out, BLOCK_M=bm, BLOCK_N=bn, SPARSITY_N=n, SPARSITY_M=m)

        kept = (out.reshape(bm, bn // m, m) != float("-inf")).sum(dim=-1)
        if m == 4:
            assert (kept == n).all(), (
                f"M=4 tie: expected {n}, got min={kept.min()}, max={kept.max()}"
            )
        else:
            assert (kept >= n).all(), f"M=8 tie: expected >= {n}, got min={kept.min()}"
            assert (kept <= m).all(), f"M=8 tie: expected <= {m}, got max={kept.max()}"


# ---------------------------------------------------------------------------
# Sparse backward sanity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseBackward:
    """Backward pass sanity checks with N:M sparsity enabled."""

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
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta([seq_len])

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
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        locs, lens = make_varlen_meta([seq_len])

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
        assert not torch.allclose(v_d.grad, v_s.grad, atol=1e-3), (
            "dV same with and without sparsity"
        )


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseNMIntegration:
    """N:M sparse softmax integration tests."""

    def test_sparse_disabled_matches_dense(self):
        """sparsity_n=0 produces bit-identical output to default (dense)."""
        seq_lens = [128, 128]
        total = sum(seq_lens)
        scale = 1.0 / (64**0.5)

        torch.manual_seed(99)
        q, k, v = make_qkv(total, 4, 2, 64)
        locs, lens = make_varlen_meta(seq_lens)

        out_dense = attention(q, k, v, locs, lens, 128, softmax_scale=scale)
        out_n0 = attention(q, k, v, locs, lens, 128, softmax_scale=scale, sparsity_n=0)
        assert torch.equal(out_dense, out_n0)

    def test_sparse_nm_via_sparsify(self, tiny_llama_dir):
        """mtsa.sparsify() with N:M sparse softmax produces finite logits that differ from dense."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import modelopt.torch.sparsity.attention_sparsity as mtsa

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = torch.randint(1, tok.vocab_size, (1, 64), device="cuda")

        # Dense baseline (triton backend, no sparsity)
        model_dense = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_dense.eval()
        with torch.no_grad():
            logits_dense = model_dense(input_ids=ids).logits
        del model_dense

        # Sparse via mtsa.sparsify() with dense_window_size=0 to force sparsity on all tiles
        sparse_cfg = {
            "sparse_cfg": {
                "*attn*": {
                    "method": "triton_sparse_softmax",
                    "sparsity_n": 2,
                    "sparsity_m": 4,
                    "num_sink_tokens": 0,
                    "dense_window_size": 0,
                    "backend": "triton",
                    "enable": True,
                },
                "default": {"enable": False},
            },
        }
        model_sparse = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        mtsa.sparsify(model_sparse, sparse_cfg)
        assert model_sparse.config._attn_implementation == "modelopt_triton"
        model_sparse.eval()
        with torch.no_grad():
            logits_sparse = model_sparse(input_ids=ids).logits

        assert not torch.isnan(logits_sparse).any(), "NaN in sparse logits"
        assert not torch.isinf(logits_sparse).any(), "Inf in sparse logits"
        assert not torch.allclose(logits_sparse, logits_dense, atol=1e-2), (
            "Sparse logits identical to dense — sparsity may not be applied"
        )
