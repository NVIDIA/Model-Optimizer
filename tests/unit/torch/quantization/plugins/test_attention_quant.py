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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.transformers_models import get_tiny_bert, get_tiny_llama, get_tiny_t5

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
