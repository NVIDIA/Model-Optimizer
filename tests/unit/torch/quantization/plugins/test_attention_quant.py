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

import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.transformers_models import get_tiny_bert, get_tiny_llama, get_tiny_t5
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

try:
    import kitchen
except ImportError:
    kitchen = None

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.huggingface import _QuantAttention

transformers = pytest.importorskip("transformers")


class MatmulAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        q, k, v = hidden_states, hidden_states, hidden_states
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        return torch.matmul(a, v), None


class BMMAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        q, k, v = hidden_states, hidden_states, hidden_states
        a = torch.softmax(torch.bmm(q, k.transpose(-2, -1)), dim=-1)
        return torch.bmm(a, v), None


class BinMatmulAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        q, k, v = hidden_states, hidden_states, hidden_states
        return torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v, None


class SDPAAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        q, k, v = hidden_states, hidden_states, hidden_states
        return F.scaled_dot_product_attention(q, k, v), None


kv_cache_config = {
    "quant_cfg": {
        "*[kv]_bmm_quantizer": {"num_bits": 4, "enable": True},
        "*softmax_quantizer": {"enable": False},
    },
    "algorithm": "max",
}


@pytest.mark.parametrize(
    ("model_getter", "attn_cls"),
    [
        (get_tiny_llama, None),
        (get_tiny_llama, MatmulAttention),
        (get_tiny_llama, BMMAttention),
        (get_tiny_llama, BinMatmulAttention),
        (get_tiny_llama, SDPAAttention),
        (get_tiny_t5, None),
    ],
)
def test_kv_quant_hf(model_getter, attn_cls):
    model_test = model_getter()
    print(model_test)
    input_ids = torch.randint(0, model_test.config.vocab_size, (1, 4))
    if getattr(model_test.config, "is_encoder_decoder", False):
        kwargs = {"decoder_input_ids": input_ids}
        attention_module = "SelfAttention"
    else:
        kwargs = {}
        attention_module = "self_attn"

    original_is_compatible_attention = None
    if attn_cls is not None:
        # Test case for transformers < 4.48
        # This needs:
        # 1) replace the attention class with the test attention class
        # 2) set _QuantAttention.is_compatible_attention output to False to fall back to the transformers < 4.48 support
        for name, module in model_test.named_modules():
            if name.endswith(attention_module):
                if original_is_compatible_attention is None:
                    original_is_compatible_attention = _QuantAttention.is_compatible_attention
                    _QuantAttention.is_compatible_attention = classmethod(lambda cls, x: False)

                parent = model_test.get_submodule(name.split(f".{attention_module}")[0])
                setattr(parent, attention_module, attn_cls())

    model_test(input_ids, **kwargs)
    mtq.quantize(model_test, kv_cache_config, lambda model: model(input_ids, **kwargs))

    for name, module in model_test.named_modules():
        if name.endswith(attention_module):
            assert hasattr(module, "k_bmm_quantizer")
            assert hasattr(module, "v_bmm_quantizer")
            assert module.k_bmm_quantizer.amax is not None
            assert module.v_bmm_quantizer.amax is not None

    model_test(input_ids, **kwargs)

    if attn_cls is not None:
        _QuantAttention.is_compatible_attention = original_is_compatible_attention
        mtq.unregister(attn_cls)


def test_kv_quant_bert():
    """Test KV cache quantization on BERT model with decorated attention."""
    model_test = get_tiny_bert()
    input_ids = torch.randint(0, model_test.config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)

    # Run forward pass before quantization
    model_test(input_ids, attention_mask=attention_mask)

    # Quantize with KV cache quantization
    mtq.quantize(
        model_test,
        kv_cache_config,
        lambda model: model(input_ids, attention_mask=attention_mask),
    )

    # BERT attention modules are at encoder.layer.X.attention.self
    found_quantized_attention = False
    for name, module in model_test.named_modules():
        if "attention.self" in name or name.endswith(".self"):
            if hasattr(module, "k_bmm_quantizer") and hasattr(module, "v_bmm_quantizer"):
                found_quantized_attention = True
                # Verify quantizers were calibrated
                assert module.k_bmm_quantizer.amax is not None, f"k_bmm not calibrated in {name}"
                assert module.v_bmm_quantizer.amax is not None, f"v_bmm not calibrated in {name}"
                assert module.q_bmm_quantizer.amax is not None, f"q_bmm not calibrated in {name}"

    assert found_quantized_attention, "No quantized attention modules found in BERT model"

    # Run forward pass after quantization to ensure it works
    output = model_test(input_ids, attention_mask=attention_mask)
    assert output is not None
    assert output.start_logits is not None
    assert output.end_logits is not None


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
    assert hasattr(quant_attention, "softmax_quantizer")
    quant_attention.softmax_quantizer.disable()
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
    quant_attention.softmax_quantizer.num_bits = (4, 3)
    quant_attention.softmax_quantizer.block_sizes = {
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
