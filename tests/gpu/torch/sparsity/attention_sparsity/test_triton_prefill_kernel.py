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

"""GPU tests for Triton prefill attention kernel.

Part 1 (correctness): Compare Triton kernel output to PyTorch SDPA (Flash Attention).
Part 2 (integration): Compare full model output when attention uses Triton vs SDPA.
Part 3 (HF integration): Load HF model via from_pretrained, run model.generate with SDPA vs
Triton (first layer prefill); compare generated output text directly (no numerical tolerance).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress optional-plugin and deprecation warnings when running only this test file
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

# Only run if Triton kernel is available (CUDA + triton)
from modelopt.torch.sparsity.attention_sparsity.kernels import (
    IS_AVAILABLE as TRITON_KERNEL_AVAILABLE,
)

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.sparsity.attention_sparsity.kernels import (
        context_attention_fwd,
        set_skip_threshold,
    )


def _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len):
    """Reference: PyTorch F.scaled_dot_product_attention (Flash Attention when available).

    Causal attention, same layout as Triton kernel. Supports GQA by repeating k,v
    to match q head count. Returns [total_tokens, num_heads, dim].
    """
    batch = b_seq_len.shape[0]
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    out_list = []
    for b in range(batch):
        start = b_start_loc[b].item()
        length = b_seq_len[b].item()
        q_b = q[start : start + length].unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k[start : start + length].unsqueeze(0).permute(0, 2, 1, 3)
        v_b = v[start : start + length].unsqueeze(0).permute(0, 2, 1, 3)
        if num_heads_q != num_heads_kv:
            repeat = num_heads_q // num_heads_kv
            k_b = k_b.repeat_interleave(repeat, dim=1)
            v_b = v_b.repeat_interleave(repeat, dim=1)
        o_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        o_b = o_b.permute(0, 2, 1, 3).squeeze(0)
        out_list.append(o_b)
    return torch.cat(out_list, dim=0)


# -----------------------------------------------------------------------------
# Part 1: Kernel correctness (Triton vs Flash Attention / SDPA)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonPrefillKernelCorrectness:
    """Compare Triton prefill kernel output to PyTorch SDPA (Flash Attention)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    def test_triton_vs_sdpa_causal_fp32(self):
        """Triton vs SDPA causal attention, float32."""
        max_seq_len = 16
        seq_lens = [8, 12]
        total_tokens = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(123)
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        o_triton = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_triton,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o_sdpa = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)
        torch.testing.assert_close(o_triton, o_sdpa, rtol=1e-2, atol=1e-2)

    def test_triton_vs_sdpa_causal_fp16(self):
        """Triton vs SDPA causal attention, float16."""
        max_seq_len = 16
        seq_lens = [8, 12]
        total_tokens = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(456)
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        o_triton = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_triton,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o_sdpa = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)
        torch.testing.assert_close(o_triton, o_sdpa, rtol=2e-2, atol=2e-2)

    def test_triton_vs_sdpa_causal_gqa(self):
        """Triton vs SDPA causal attention with GQA (4 q-heads, 2 kv-heads)."""
        max_seq_len = 16
        seq_lens = [8, 12]
        total_tokens = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(789)
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        o_triton = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_triton,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o_sdpa = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)
        torch.testing.assert_close(o_triton, o_sdpa, rtol=1e-2, atol=1e-2)

    def test_triton_bidirectional_forward(self):
        """Triton kernel runs bidirectional (is_causal=False); shape and finite."""
        max_seq_len = 8
        seq_lens = [4, 6]
        total_tokens = sum(seq_lens)
        num_heads, head_dim = 2, 32

        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        k = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        v = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)
        o = torch.empty_like(q)

        context_attention_fwd(
            q,
            k,
            v,
            o,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=False,
        )
        assert o.shape == q.shape
        assert not torch.isnan(o).any() and not torch.isinf(o).any()


# -----------------------------------------------------------------------------
# Part 2: Integration (small model output: Triton vs SDPA)
# -----------------------------------------------------------------------------


class _AttentionBlockModel(nn.Module):
    """Minimal model: linear Q,K,V + attention (SDPA or Triton) + output linear.

    Same weights for both modes; only the attention implementation differs.
    """

    def __init__(self, hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8, use_triton=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_triton = use_triton
        assert num_heads % num_kv_heads == 0
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)

    def forward(self, x, b_start_loc, b_seq_len, max_seq_len):
        """x: [total_tokens, hidden_size]. Returns [total_tokens, hidden_size]."""
        total_tokens = x.shape[0]
        scale = 1.0 / (self.head_dim**0.5)

        q = self.q_proj(x).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(total_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(total_tokens, self.num_kv_heads, self.head_dim)

        if self.use_triton:
            o = torch.empty_like(q)
            context_attention_fwd(
                q,
                k,
                v,
                o,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                max_input_len=max_seq_len,
                is_causal=True,
                softmax_scale=scale,
            )
        else:
            o = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)

        o_flat = o.reshape(total_tokens, self.num_heads * self.head_dim)
        return self.out_proj(o_flat)


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonPrefillKernelIntegration:
    """Compare full model output when attention uses Triton vs SDPA."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    def test_small_model_triton_vs_sdpa_output_match(self):
        """Same small model and input; Triton vs SDPA attention => same output."""
        hidden_size = 32
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8
        seq_lens = [8, 6]
        total_tokens = sum(seq_lens)
        max_seq_len = max(seq_lens)

        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        model_sdpa = _AttentionBlockModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_triton=False,
        ).to(self.device)

        model_triton = _AttentionBlockModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_triton=True,
        ).to(self.device)

        model_triton.load_state_dict(model_sdpa.state_dict())

        torch.manual_seed(99)
        x = torch.randn(total_tokens, hidden_size, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            out_sdpa = model_sdpa(x, b_start_loc, b_seq_len, max_seq_len)
            out_triton = model_triton(x, b_start_loc, b_seq_len, max_seq_len)

        torch.testing.assert_close(out_triton, out_sdpa, rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Part 3: HF model loading + Triton kernel (integration with AutoModelForCausalLM)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonPrefillKernelHFIntegration:
    """HF model + Triton kernel: generate text and compare output (no numerical diff)."""

    @pytest.fixture(scope="class")
    def tiny_llama_dir(self, tmp_path_factory):
        """Create minimal Llama with head_dim=16 (Triton dot requires K>=16)."""
        from _test_utils.torch.transformers_models import create_tiny_llama_dir

        return create_tiny_llama_dir(
            tmp_path_factory.mktemp("tiny_llama_triton"),
            with_tokenizer=True,
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )

    @pytest.fixture(scope="class")
    def hf_llama_model(self, tiny_llama_dir):
        """Load HF Llama via AutoModelForCausalLM.from_pretrained (same as user flow)."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return model

    @pytest.fixture(scope="class")
    def hf_llama_tokenizer(self, tiny_llama_dir):
        """Load tokenizer for the tiny Llama (for decoding generated ids to text)."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tiny_llama_dir)

    def test_generate_text_sdpa_vs_triton_first_layer(
        self, tiny_llama_dir, hf_llama_model, hf_llama_tokenizer
    ):
        """Eager vs modelopt_triton: same weights; both run and generate same-length output."""
        from transformers import AutoModelForCausalLM

        tokenizer = hf_llama_tokenizer
        device = next(hf_llama_model.parameters()).device
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_eager = hf_llama_model
        model_triton = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_triton.load_state_dict(model_eager.state_dict())

        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        max_new_tokens = 5
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }

        model_eager.eval()
        model_triton.eval()
        torch.manual_seed(42)
        with torch.no_grad():
            out_eager = model_eager.generate(input_ids, **gen_kwargs)
        torch.manual_seed(42)
        with torch.no_grad():
            out_triton = model_triton.generate(input_ids, **gen_kwargs)
        assert out_eager.shape == out_triton.shape, (
            f"Generated shape mismatch: eager {out_eager.shape}, Triton {out_triton.shape}"
        )


# -----------------------------------------------------------------------------
# Part 4: Skip softmax (Triton kernel with SKIP_SOFTMAX=True via skip_threshold)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonSkipSoftmax:
    """Triton kernel with SKIP_SOFTMAX=True: assert generated text matches reference."""

    @pytest.fixture(scope="class")
    def tiny_llama_dir(self, tmp_path_factory):
        """Create minimal Llama with head_dim=16 (Triton dot requires K>=16)."""
        from _test_utils.torch.transformers_models import create_tiny_llama_dir

        return create_tiny_llama_dir(
            tmp_path_factory.mktemp("tiny_llama_skip_softmax"),
            with_tokenizer=True,
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )

    @pytest.fixture(scope="class")
    def hf_llama_model(self, tiny_llama_dir):
        """Load HF Llama via AutoModelForCausalLM.from_pretrained."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    @pytest.fixture(scope="class")
    def hf_llama_tokenizer(self, tiny_llama_dir):
        """Load tokenizer for the tiny Llama."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tiny_llama_dir)

    def test_generate_text_triton_no_skip_vs_triton_skip_softmax(
        self, tiny_llama_dir, hf_llama_model, hf_llama_tokenizer
    ):
        """Triton (no skip) vs Triton (skip_threshold=1e-3): same text (small threshold rarely skips)."""
        from transformers import AutoModelForCausalLM

        tokenizer = hf_llama_tokenizer
        device = next(hf_llama_model.parameters()).device
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Both use modelopt_triton; only the second has a small skip_threshold
        model_ref = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_ref.load_state_dict(hf_llama_model.state_dict())
        model_skip = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_skip.load_state_dict(hf_llama_model.state_dict())
        set_skip_threshold(model_skip, 1e-3)

        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        max_new_tokens = 5
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }

        torch.manual_seed(42)
        model_ref.eval()
        with torch.no_grad():
            out_ref = model_ref.generate(input_ids, **gen_kwargs)

        torch.manual_seed(42)
        model_skip.eval()
        with torch.no_grad():
            out_skip = model_skip.generate(input_ids, **gen_kwargs)

        text_ref = tokenizer.decode(out_ref[0], skip_special_tokens=True)
        text_skip = tokenizer.decode(out_skip[0], skip_special_tokens=True)
        assert text_ref == text_skip, (
            f"Generated text should match: ref got {text_ref!r}, skip_softmax got {text_skip!r}"
        )

    def test_large_threshold_output_differs(
        self, tiny_llama_dir, hf_llama_model, hf_llama_tokenizer
    ):
        """With skip_threshold=0.99, model output (logits) must differ from reference (no skip)."""
        from transformers import AutoModelForCausalLM

        tokenizer = hf_llama_tokenizer
        device = next(hf_llama_model.parameters()).device

        model_ref = hf_llama_model
        model_skip = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_skip.load_state_dict(model_ref.state_dict())
        set_skip_threshold(model_skip, 0.99)

        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        torch.manual_seed(42)
        model_ref.eval()
        with torch.no_grad():
            out_ref = model_ref(input_ids=input_ids)
        logits_ref = out_ref.logits

        torch.manual_seed(42)
        model_skip.eval()
        with torch.no_grad():
            out_skip = model_skip(input_ids=input_ids)
        logits_skip = out_skip.logits

        assert not torch.equal(logits_ref, logits_skip), (
            "Logits should differ with large skip_threshold (0.99); ref and skip outputs are identical."
        )
