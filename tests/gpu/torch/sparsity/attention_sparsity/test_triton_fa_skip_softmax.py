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

"""GPU tests for skip-softmax (BLASST) on the Triton flash attention kernel."""

import pytest
import torch
from conftest import make_varlen_meta

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels import attention, register_triton_attention

    if register_triton_attention is not None:
        register_triton_attention()


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSkipSoftmax:
    """Skip-softmax tile-skipping approximation tests."""

    def _make_inputs(self, batch=2, seq_len=256, num_heads=4, num_kv_heads=2, head_dim=64):
        total = batch * seq_len
        torch.manual_seed(77)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        locs, lens = make_varlen_meta([seq_len] * batch)
        return q, k, v, locs, lens

    def test_disabled_matches_dense(self):
        """skip_softmax_threshold=None/0.0 produces bit-identical output to dense."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out_none = attention(q, k, v, locs, lens, 256, softmax_scale=scale)
        out_zero = attention(
            q, k, v, locs, lens, 256, softmax_scale=scale, skip_softmax_threshold=0.0
        )
        assert torch.equal(out_none, out_zero)

    def test_small_threshold_close_to_dense(self):
        """A small threshold (1e-3) should produce output very close to dense."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 256, softmax_scale=scale)
        out_skip = attention(
            q, k, v, locs, lens, 256, softmax_scale=scale, skip_softmax_threshold=1e-3
        )
        torch.testing.assert_close(out_skip, out_dense, rtol=5e-2, atol=5e-2)

    def test_large_threshold_differs_from_dense(self):
        """A large threshold should produce noticeably different output on spiky data.

        Random data distributes attention uniformly so few tiles are skipped.
        Use long sequences with spiky attention (one hot-key per query) to
        ensure the BLASST algorithm actually skips negligible tiles.
        """
        batch, seq_len, num_heads, head_dim = 1, 4096, 4, 64
        total = batch * seq_len
        torch.manual_seed(77)
        # Create spiky attention: each query attends strongly to one key
        q = torch.zeros(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        # Set each query to match a single key so attention is concentrated
        for i in range(total):
            q[i] = k[i]
        locs = torch.zeros(batch, device="cuda", dtype=torch.int32)
        lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

        scale = 1.0 / (head_dim**0.5)
        out_dense = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale)
        out_skip = attention(
            q, k, v, locs, lens, seq_len, softmax_scale=scale, skip_softmax_threshold=0.5
        )
        assert not torch.allclose(out_skip, out_dense, atol=1e-3)

    def test_output_shape_unchanged(self):
        """Skip-softmax does not change output shape."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out = attention(q, k, v, locs, lens, 256, softmax_scale=scale, skip_softmax_threshold=1e-2)
        assert out.shape == q.shape

    def test_monotonic_approximation_error(self):
        """Larger threshold -> larger error vs dense (monotonic degradation)."""
        q, k, v, locs, lens = self._make_inputs(seq_len=512)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 512, softmax_scale=scale)
        errors = []
        for threshold in [1e-4, 1e-2, 1e-1]:
            out_skip = attention(
                q, k, v, locs, lens, 512, softmax_scale=scale, skip_softmax_threshold=threshold
            )
            errors.append((out_skip - out_dense).abs().mean().item())
        assert errors[0] <= errors[1] <= errors[2], f"Errors not monotonic: {errors}"

    def test_decode_single_token(self):
        """Skip-softmax works for decode (single Q token per sequence)."""
        batch = 2
        seq_lens_k = [64, 128]
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        q_flat = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        total_kv = sum(seq_lens_k)
        k_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        b_start_loc_q = torch.arange(batch, device="cuda", dtype=torch.int32)
        b_seq_len_q = torch.ones(batch, device="cuda", dtype=torch.int32)
        cumsum = [0]
        for sl in seq_lens_k:
            cumsum.append(cumsum[-1] + sl)
        b_start_loc_k = torch.tensor(cumsum[:-1], device="cuda", dtype=torch.int32)
        b_seq_len_k = torch.tensor(seq_lens_k, device="cuda", dtype=torch.int32)

        out_dense = attention(
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
        out_skip = attention(
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
            skip_softmax_threshold=1e-3,
        )
        torch.testing.assert_close(out_skip, out_dense, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSkipSoftmaxVsPytorchRef:
    """Cross-validate Triton skip-softmax against PyTorch flash_skip_softmax reference."""

    def test_triton_matches_pytorch_reference(self):
        """Triton skip-softmax output should be close to PyTorch reference with same threshold.

        The reference computes block-level BLASST masks using FlashSkipSoftmax.calculate_sparsity
        and applies them to standard softmax attention. The Triton kernel fuses the same skip
        logic into the online softmax inner loop.
        """
        from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import (
            FlashSkipSoftmax,
        )

        batch, seq_len = 1, 256
        num_heads, num_kv_heads, head_dim = 4, 4, 64  # MHA for simplicity
        scale = 1.0 / (head_dim**0.5)
        threshold = 1e-2

        torch.manual_seed(123)
        q_4d = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
        k_4d = torch.randn(
            batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float32
        )
        v_4d = torch.randn(
            batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float32
        )

        # --- PyTorch reference: eager attention with flash_skip_softmax ---
        scores = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * scale
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        # Apply BLASST mask via flash_skip_softmax
        method = FlashSkipSoftmax(
            method_config={
                "thresholds": {"prefill": [threshold]},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "is_causal": True,
            }
        )
        sparse_mask, _ = method.calculate_sparsity(scores)
        if sparse_mask is not None:
            scores = scores.masked_fill(~sparse_mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        ref_out = torch.matmul(p, v_4d)  # [batch, heads, seq, dim]

        # --- Triton kernel with skip-softmax ---
        total = batch * seq_len
        q_flat = q_4d.permute(0, 2, 1, 3).reshape(total, num_heads, head_dim).contiguous()
        k_flat = k_4d.permute(0, 2, 1, 3).reshape(total, num_kv_heads, head_dim).contiguous()
        v_flat = v_4d.permute(0, 2, 1, 3).reshape(total, num_kv_heads, head_dim).contiguous()
        locs = torch.arange(batch, device="cuda", dtype=torch.int32) * seq_len
        lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

        triton_out = attention(
            q_flat,
            k_flat,
            v_flat,
            locs,
            lens,
            seq_len,
            is_causal=True,
            softmax_scale=scale,
            skip_softmax_threshold=threshold,
        )
        triton_out_4d = triton_out.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Both outputs should be close — same algorithm, different implementations.
        # Observed max abs error ~2e-3 (online vs standard softmax precision diffs).
        torch.testing.assert_close(triton_out_4d, ref_out, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSkipSoftmaxHFIntegration:
    """HF integration for skip-softmax via mtsa.sparsify()."""

    def test_skip_softmax_via_sparsify(self, tiny_llama_dir):
        """mtsa.sparsify() with triton_skip_softmax produces finite logits."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import modelopt.torch.sparsity.attention_sparsity as mtsa

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = torch.randint(1, tok.vocab_size, (1, 64), device="cuda")

        # Dense baseline (triton backend, no skip)
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

        # Skip-softmax via mtsa.sparsify()
        model_skip = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        mtsa.sparsify(model_skip, mtsa.SKIP_SOFTMAX_TRITON_DEFAULT)
        model_skip.eval()
        with torch.no_grad():
            logits_skip = model_skip(input_ids=ids).logits

        assert not torch.isnan(logits_skip).any(), "NaN in skip-softmax logits"
        assert not torch.isinf(logits_skip).any(), "Inf in skip-softmax logits"
        # On short sequences (64 tokens), no tiles are skipped — output should match dense
        torch.testing.assert_close(logits_skip, logits_dense, rtol=1e-3, atol=1e-3)
