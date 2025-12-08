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

"""Integration testing with locally created minimal Llama model."""

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.sparsity.attention_sparsity import SparseAttentionConfig
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    """Create minimal Llama model locally."""
    return create_tiny_llama_dir(
        tmp_path_factory.mktemp("tiny_llama"),
        with_tokenizer=True,
        num_hidden_layers=2,  # Minimal layers for fast testing
        hidden_size=32,
        intermediate_size=64,
    )


@pytest.fixture(scope="module")
def tinyllama_model(tiny_llama_dir):
    """Load locally created tiny Llama model."""
    model = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir,
        attn_implementation="eager",
        device_map="cuda",
    )
    return model


@pytest.fixture(scope="module")
def tinyllama_tokenizer(tiny_llama_dir):
    """Load tokenizer for tiny Llama model."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_llama_dir)
    return tokenizer


class TestTinyLlama:
    """TinyLlama sparse attention tests."""

    def test_load_and_sparsify(self, tinyllama_model):
        """Load TinyLlama and apply sparse attention."""
        model = tinyllama_model

        config = SparseAttentionConfig(
            sparse_cfg={
                "*attn*": {
                    "method": "flash_skip_softmax",
                    "threshold": 1e-3,
                    "br": 128,
                    "bc": 128,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Verify sparse attention modules were added
        sparse_count = sum(
            1 for m in sparse_model.modules() if isinstance(m, SparseAttentionModule)
        )
        assert sparse_count > 0, "No sparse attention modules found"

        # Our tiny llama has 2 layers, so should have 2 attention modules
        assert sparse_count == 2, f"Expected 2 sparse modules, got {sparse_count}"

    def test_forward_prefill(self, tinyllama_model, tinyllama_tokenizer):
        """Forward pass with seq_len=64 (prefill)."""
        model = tinyllama_model
        tokenizer = tinyllama_tokenizer

        config = SparseAttentionConfig(
            sparse_cfg={
                "*attn*": {
                    "threshold": 1e-3,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Create prefill input (seq_len > 1)
        test_text = "Once upon a time in a land far away"
        inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

        # Forward pass
        sparse_model.eval()
        with torch.no_grad():
            outputs = sparse_model(**inputs)

        # Verify output
        assert outputs.logits is not None
        assert not torch.isnan(outputs.logits).any()
        assert outputs.logits.shape[1] == inputs.input_ids.shape[1]  # seq_len preserved

    def test_forward_decode(self, tinyllama_model):
        """Forward pass with seq_len=1 (decode)."""
        model = tinyllama_model

        config = SparseAttentionConfig(
            sparse_cfg={
                "*attn*": {
                    "threshold": 1e-5,  # More conservative for decode
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Create decode input (seq_len = 1)
        input_ids = torch.randint(0, 32000, (1, 1), device="cuda")

        # Forward pass
        sparse_model.eval()
        with torch.no_grad():
            outputs = sparse_model(input_ids)

        # Verify output
        assert outputs.logits is not None
        assert not torch.isnan(outputs.logits).any()
        assert outputs.logits.shape == (1, 1, 32000)  # batch=1, seq=1, vocab_size

    def test_gqa_attention(self, tinyllama_model):
        """Verify GQA support (num_kv_heads < num_heads)."""
        model = tinyllama_model

        # Check if model uses GQA
        config = model.config
        has_gqa = hasattr(config, "num_key_value_heads") and (
            config.num_key_value_heads < config.num_attention_heads
        )

        if not has_gqa:
            pytest.skip("Model does not use GQA")

        # Apply sparse attention
        sparse_config = SparseAttentionConfig(
            sparse_cfg={
                "*attn*": {
                    "threshold": 1e-3,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, sparse_config)

        # Test forward pass with GQA
        input_ids = torch.randint(0, 32000, (1, 32), device="cuda")

        sparse_model.eval()
        with torch.no_grad():
            outputs = sparse_model(input_ids)

        assert outputs.logits is not None
        assert not torch.isnan(outputs.logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
