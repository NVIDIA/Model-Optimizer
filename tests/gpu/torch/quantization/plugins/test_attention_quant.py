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

"""GPU tests for HF attention quantization (softmax-P qdq via the Triton kernel).

The CPU-only config-detection test lives in the mirror unit test
(tests/unit/torch/quantization/plugins/test_attention_quant.py::test_p_qdq_mode_detection).
"""

import inspect

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

try:
    import kitchen
except ImportError:
    kitchen = None

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_FA_AVAILABLE
from modelopt.torch.quantization.plugins.huggingface import _QuantAttention

pytest.importorskip("transformers")


def _make_quant_attention(hidden_size=128, num_q_heads=4, num_kv_heads=2):
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
    )
    quant_attention = _QuantAttention.convert(LlamaAttention(config, layer_idx=0))
    quant_attention.config._attn_implementation = "sdpa"
    return quant_attention


@pytest.mark.skipif(not TRITON_FA_AVAILABLE, reason="Triton attention kernel unavailable")
def test_p_qdq_fa():
    """FP8/NVFP4 p_bmm_quantizer runs on the built-in Triton kernel (no kitchen)."""
    batch_size, num_q_heads, num_kv_heads, seqlen, head_dim = 2, 4, 2, 32, 64

    quant_attention = _make_quant_attention(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        getattr(quant_attention, name).disable()

    torch.manual_seed(29)
    q_states = torch.randn(
        batch_size, num_q_heads, seqlen, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    k_states = torch.randn(
        batch_size, num_kv_heads, seqlen, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    v_states = torch.randn(
        batch_size, num_kv_heads, seqlen, head_dim, dtype=torch.bfloat16, device="cuda"
    )

    module = inspect.getmodule(quant_attention.get_attn_type(quant_attention))
    orig_attn_fn = module.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def run():
        return quant_attention._quantized_attention(
            orig_attn_fn,
            quant_attention,
            q_states,
            k_states,
            v_states,
            attention_mask=None,
        )[0]

    quant_attention.p_bmm_quantizer.disable()
    expected = run()

    quant_attention.p_bmm_quantizer.enable()
    for num_bits, block_sizes, tol in [
        ((4, 3), None, 0.1),  # FP8
        ((2, 1), {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}, 0.4),  # NVFP4
    ]:
        quant_attention.p_bmm_quantizer.num_bits = num_bits
        quant_attention.p_bmm_quantizer.block_sizes = block_sizes
        output = run()
        assert output.shape == expected.shape
        assert not torch.equal(output, expected), "softmax qdq should perturb the output"
        torch.testing.assert_close(output, expected, atol=tol, rtol=tol)

    # A user-set (or calibrated) per-tensor amax on the quantizer overrides the
    # kernel's default of 1.0 and changes the quantization grid. Use a non-power-of-2
    # amax (3.0): a power-of-2 change is a fixed point of FP quant and wouldn't differ.
    quant_attention.p_bmm_quantizer.num_bits = (4, 3)
    quant_attention.p_bmm_quantizer.block_sizes = None
    out_default = run()
    quant_attention.p_bmm_quantizer.amax = torch.tensor(3.0)
    out_amax = run()
    assert not torch.equal(out_amax, out_default), "user-set amax should change the output"
    torch.testing.assert_close(out_amax, expected, atol=0.1, rtol=0.1)


@pytest.mark.skipif(not TRITON_FA_AVAILABLE, reason="Triton attention kernel unavailable")
def test_p_qdq_unsupported_cases_raise():
    """The Triton qdq dispatch rejects attention semantics the kernel cannot honor."""
    batch_size, num_q_heads, num_kv_heads, seqlen, head_dim = 2, 4, 2, 32, 64

    quant_attention = _make_quant_attention(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        getattr(quant_attention, name).disable()
    quant_attention.p_bmm_quantizer.num_bits = (4, 3)  # FP8 mode

    def make_qkv(seq_q=seqlen, seq_k=seqlen):
        q = torch.randn(batch_size, num_q_heads, seq_q, head_dim, device="cuda")
        k = torch.randn(batch_size, num_kv_heads, seq_k, head_dim, device="cuda")
        v = torch.randn(batch_size, num_kv_heads, seq_k, head_dim, device="cuda")
        return q, k, v

    def run(seq_q=seqlen, seq_k=seqlen, **kwargs):
        q, k, v = make_qkv(seq_q, seq_k)
        return quant_attention._quantized_attention(None, quant_attention, q, k, v, **kwargs)

    with pytest.raises(NotImplementedError, match="sliding-window"):
        run(sliding_window=128)
    with pytest.raises(NotImplementedError, match="attention sinks"):
        run(s_aux=torch.zeros(num_q_heads, device="cuda"))
    with pytest.raises(NotImplementedError, match="softcapping"):
        run(softcap=50.0)
    with pytest.raises(NotImplementedError, match="dropout"):
        run(dropout=0.1)
    with pytest.raises(NotImplementedError, match="KV cache"):
        run(seq_q=4, seq_k=20)  # chunked prefill / assisted decoding
    with pytest.raises(NotImplementedError, match="cached decode"):
        # FA2-style [batch, kv_len] padding mask during single-token decode
        run(seq_q=1, seq_k=20, attention_mask=torch.ones(batch_size, 20, device="cuda"))
    with pytest.raises(NotImplementedError, match="left-padded"):
        left_pad = torch.ones(batch_size, seqlen, device="cuda")
        left_pad[0, :5] = 0
        run(attention_mask=left_pad)
    with pytest.raises(NotImplementedError, match="non-contiguously padded"):
        # A hole in the mask would sum to the wrong per-sequence length.
        holey = torch.ones(batch_size, seqlen, device="cuda")
        holey[0, 3] = 0
        run(attention_mask=holey)
    with pytest.raises(NotImplementedError, match="padding or non-causal"):
        padded_4d = torch.zeros(batch_size, 1, seqlen, seqlen, device="cuda")
        padded_4d[..., -4:] = torch.finfo(torch.float32).min  # last 4 kv positions padded
        run(attention_mask=padded_4d)

    # A purely causal 4D mask is safe to ignore: the kernel masks causally itself.
    causal_4d = torch.zeros(batch_size, 1, seqlen, seqlen, device="cuda")
    causal_4d.masked_fill_(
        torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device="cuda"), diagonal=1),
        torch.finfo(torch.float32).min,
    )
    output = run(attention_mask=causal_4d)[0]
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not TRITON_FA_AVAILABLE, reason="Triton attention kernel unavailable")
def test_p_qdq_non_causal_falls_back_to_eager():
    """Non-causal attention (e.g. ViT) is outside the causal-only Triton kernel's
    envelope, so p_bmm_quantizer is applied through the eager softmax wrapper
    instead of raising -- keeping the softmax-P quant in an export-traceable graph."""
    batch_size, num_q_heads, num_kv_heads, seqlen, head_dim = 2, 4, 2, 32, 64

    quant_attention = _make_quant_attention(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
        getattr(quant_attention, name).disable()

    torch.manual_seed(29)
    q = torch.randn(batch_size, num_q_heads, seqlen, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch_size, num_kv_heads, seqlen, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch_size, num_kv_heads, seqlen, head_dim, dtype=torch.bfloat16, device="cuda")

    # Eager interface so the wrapper's F.softmax swap actually fires (SDPA fuses softmax).
    module = inspect.getmodule(quant_attention.get_attn_type(quant_attention))
    eager_fn = module.eager_attention_forward

    def run():
        return quant_attention._quantized_attention(
            eager_fn,
            quant_attention,
            q,
            k,
            v,
            attention_mask=None,
            scaling=head_dim**-0.5,
            is_causal=False,
        )[0]

    quant_attention.p_bmm_quantizer.disable()
    expected = run()

    quant_attention.p_bmm_quantizer.enable()
    quant_attention.p_bmm_quantizer.num_bits = (4, 3)  # FP8
    quant_attention.p_bmm_quantizer.amax = torch.tensor(1.0, device="cuda")  # softmax P in [0, 1]
    output = run()

    assert output.shape == expected.shape
    assert not torch.equal(output, expected), "softmax qdq should perturb the output"
    torch.testing.assert_close(output, expected, atol=0.1, rtol=0.1)


@pytest.mark.skipif(kitchen is None, reason="kitchen is not installed.")
def test_kitchen_fa():
    batch_size = 2
    num_q_heads = 4
    num_kv_heads = 2
    seqlen = 8
    hidden_size = 128

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
    )
    original_attention = LlamaAttention(config, layer_idx=0)

    q_states = torch.randn(
        batch_size, num_q_heads, seqlen, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    k_states = torch.randn(
        batch_size, num_kv_heads, seqlen, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    v_states = torch.randn(
        batch_size, num_kv_heads, seqlen, hidden_size, dtype=torch.bfloat16, device="cuda"
    )

    # Convert it to _QuantAttention using the convert() class method
    quant_attention = _QuantAttention.convert(original_attention)
    quant_attention.config._attn_implementation = "sdpa"
    assert hasattr(quant_attention, "q_bmm_quantizer")
    assert hasattr(quant_attention, "k_bmm_quantizer")
    assert hasattr(quant_attention, "v_bmm_quantizer")
    assert hasattr(quant_attention, "p_bmm_quantizer")
    quant_attention.p_bmm_quantizer.disable()
    module = inspect.getmodule(quant_attention.get_attn_type(quant_attention))
    orig_attn_fn = module.ALL_ATTENTION_FUNCTIONS["sdpa"]

    output = quant_attention._quantized_attention(
        orig_attn_fn,
        quant_attention,
        q_states,
        k_states,
        v_states,
        attention_mask=None,
    )
    expected = output[0]

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
    )
    original_attention = LlamaAttention(config, layer_idx=0)
    quant_attention = _QuantAttention.convert(original_attention)
    quant_attention.config._attn_implementation = "sdpa"
    quant_attention.p_bmm_quantizer.num_bits = (4, 3)
    quant_attention.p_bmm_quantizer.block_sizes = {
        -1: 32,
        "type": "dynamic",
        "scale_bits": (8, 0),
    }
    output = quant_attention._quantized_attention(
        None,
        quant_attention,
        q_states,
        k_states,
        v_states,
        attention_mask=None,
    )
    diff = (expected - output[0]).abs()
    assert torch.allclose(expected, output[0], atol=0.75, rtol=0.75), (
        f"{diff.max().item(), diff.mean().item(), diff.std().item()}"
    )
