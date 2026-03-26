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

"""GPU tests for Triton flash attention kernel (dense path)."""

import pytest
import torch
import torch.nn.functional as F
from conftest import make_qkv, make_varlen_meta, sdpa_reference

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


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestForward:
    """Forward pass correctness for dense attention."""

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
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim, dtype=dtype)
        locs, lens = make_varlen_meta(seq_lens)

        o = attention(q, k, v, locs, lens, max(seq_lens), softmax_scale=scale)
        torch.testing.assert_close(o, sdpa_reference(q, k, v, locs, lens), rtol=1e-3, atol=1e-3)

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
        q, k, v = make_qkv(total, 4, 2, 64)
        locs, lens = make_varlen_meta(seq_lens)

        out_dense = attention(q, k, v, locs, lens, 128, softmax_scale=scale)
        out_n0 = attention(q, k, v, locs, lens, 128, softmax_scale=scale, sparsity_n=0)
        assert torch.equal(out_dense, out_n0)


# ---------------------------------------------------------------------------
# Backward correctness (dense)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestBackward:
    """Backward pass gradient correctness for dense attention."""

    def _sdpa_backward_ref(self, q, k, v, scale, is_causal=True):
        """Run SDPA forward+backward, return gradients."""
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

    def test_dense_causal_matches_sdpa(self):
        """dQ, dK, dV match SDPA for causal self-attention."""
        seq_len, num_heads, num_kv_heads, head_dim = 16, 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta([seq_len])

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
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta([seq_len])

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
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta(seq_lens)

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
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float32)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        locs, lens = make_varlen_meta([seq_len])

        attention(q, k, v, locs, lens, seq_len, softmax_scale=scale).sum().backward()
        dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(q.detach(), k.detach(), v.detach(), scale)

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)


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

    def test_sparse_nm_via_sparsify(self, tiny_llama_dir):
        """mtsa.sparsify() with N:M sparse softmax produces finite logits that differ from dense."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import modelopt.torch.sparsity.attention_sparsity as mtsa

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        # Use a long input (fill max_position_embeddings=64) so sparsity has tiles to prune
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

        # Sparse output should be finite
        assert not torch.isnan(logits_sparse).any(), "NaN in sparse logits"
        assert not torch.isinf(logits_sparse).any(), "Inf in sparse logits"
        # Sparse output should differ from dense (sparsity changes attention)
        assert not torch.allclose(logits_sparse, logits_dense, atol=1e-2), (
            "Sparse logits identical to dense — sparsity may not be applied"
        )
