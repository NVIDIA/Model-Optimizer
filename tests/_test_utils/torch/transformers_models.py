# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
from pathlib import Path

import pytest
import torch
from _test_utils.torch.misc import set_seed
from packaging.version import Version

transformers = pytest.importorskip("transformers")
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

if Version(transformers.__version__) >= Version("4.55"):
    from transformers import GptOssConfig, GptOssForCausalLM

import modelopt.torch.opt as mto

SEED = 1234


def get_tiny_qwen3(**config_kwargs) -> "Qwen3ForCausalLM":
    set_seed(SEED)

    kwargs = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(**config_kwargs)
    tiny_qwen3 = Qwen3ForCausalLM(Qwen3Config(**kwargs))

    return tiny_qwen3


def get_tiny_qwen3_moe(**config_kwargs) -> "Qwen3MoeForCausalLM":
    set_seed(SEED)

    kwargs = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
    }
    kwargs.update(**config_kwargs)
    tiny_qwen3_moe = Qwen3MoeForCausalLM(Qwen3MoeConfig(**kwargs))

    return tiny_qwen3_moe


def get_tiny_llama(**config_kwargs) -> LlamaForCausalLM:
    set_seed(SEED)
    kwargs = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(**config_kwargs)
    tiny_llama = LlamaForCausalLM(LlamaConfig(**kwargs))

    return tiny_llama


def get_tiny_t5(**config_kwargs) -> T5ForConditionalGeneration:
    set_seed(SEED)
    kwargs = {
        "vocab_size": 32,
        "d_model": 32,
        "d_kv": 32,
        "d_ff": 32,
        "num_layers": 2,
        "num_heads": 16,
        "relative_attention_num_buckets": 8,
        "relative_attention_max_distance": 32,
        "decoder_start_token_id": 0,
    }
    kwargs.update(**config_kwargs)
    t5_model = T5ForConditionalGeneration(T5Config(**kwargs))

    return t5_model


def get_tiny_gpt_oss(**config_kwargs) -> "GptOssForCausalLM":
    set_seed(SEED)
    if Version(transformers.__version__) < Version("4.55"):
        pytest.skip("GptOssForCausalLM is not supported in transformers < 4.55")

    kwargs = {
        "num_hidden_layers": 4,
        "num_local_experts": 8,
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 32,
        "head_dim": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }
    kwargs.update(**config_kwargs)
    tiny_gpt_oss = GptOssForCausalLM(GptOssConfig(**kwargs))

    return tiny_gpt_oss


def create_tiny_llama_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path:
    llama_dir = Path(tmp_path) / "tiny_llama"
    if with_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        tokenizer.save_pretrained(llama_dir)
        config_kwargs["vocab_size"] = tokenizer.vocab_size

    tiny_llama = get_tiny_llama(**config_kwargs)
    tiny_llama = tiny_llama.to(torch.bfloat16)  # Use same dtype as TinyLlama-1.1B-Chat-v1.0
    tiny_llama.save_pretrained(llama_dir)
    return llama_dir


def create_tiny_t5_dir(tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs) -> Path:
    t5_dir = Path(tmp_path) / "tiny_t5"
    if with_tokenizer:
        tokenizer = T5Tokenizer.from_pretrained("hf-internal-testing/tiny-random-T5Model")
        tokenizer.save_pretrained(t5_dir)
        config_kwargs["vocab_size"] = tokenizer.vocab_size

    tiny_t5 = get_tiny_t5(**config_kwargs)
    tiny_t5 = tiny_t5.to(torch.bfloat16)
    tiny_t5.save_pretrained(t5_dir)
    return t5_dir


def create_tiny_bert_dir(tmp_path: Path) -> Path:
    set_seed(SEED)
    model = BertForQuestionAnswering(
        BertConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=16,
            max_position_embeddings=32,
        )
    )
    model.save_pretrained(tmp_path / "tiny_bert")
    return tmp_path / "tiny_bert"


def tf_output_tester(model_ref, model_test):
    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs)
    output_test = model_test(**inputs)
    if hasattr(output_ref, "logits"):
        assert torch.allclose(output_ref.logits, output_test.logits, atol=1e-6)
    else:
        assert torch.allclose(output_ref.start_logits, output_test.start_logits, atol=1e-6)
        assert torch.allclose(output_ref.end_logits, output_test.end_logits, atol=1e-6)


def tf_modelopt_state_and_output_tester(model_ref, model_test):
    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for module in model_test.modules():
        if hasattr(module, "_is_hf_initialized"):
            # AttributeError for PEFT models, PEFT models get `_is_hf_initialized` from model.base_model
            with contextlib.suppress(AttributeError):
                delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)
    assert model_ref_state == model_test_state

    tf_output_tester(model_ref, model_test)
