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

transformers = pytest.importorskip("transformers")
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertConfig,
    GptOssConfig,
    LlamaConfig,
    PreTrainedModel,
    Qwen3Config,
    Qwen3MoeConfig,
    T5Config,
    T5ForConditionalGeneration,
)

import modelopt.torch.opt as mto

SEED = 1234


##### Qwen3 #####
def get_tiny_qwen3(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(**config_kwargs)
    # NOTE: Use AutoModelForCausalLM.from_config() instead of Qwen3ForCausalLM() for correct dtype handling
    tiny_qwen3 = AutoModelForCausalLM.from_config(Qwen3Config(**kwargs))

    return tiny_qwen3


def create_tiny_qwen3_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, return_model: bool = False, **config_kwargs
) -> Path | tuple[Path, PreTrainedModel]:
    qwen3_dir = Path(tmp_path) / "tiny_qwen3"
    if with_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        tokenizer.save_pretrained(qwen3_dir)
        config_kwargs["vocab_size"] = tokenizer.vocab_size
    tiny_qwen3 = get_tiny_qwen3(**config_kwargs)
    tiny_qwen3.save_pretrained(qwen3_dir)

    if return_model:
        return qwen3_dir, tiny_qwen3
    else:
        return qwen3_dir


##### Qwen3 MoE #####
def get_tiny_qwen3_moe(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
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
    tiny_qwen3_moe = AutoModelForCausalLM.from_config(Qwen3MoeConfig(**kwargs))

    return tiny_qwen3_moe


##### GPT-OSS #####
def get_tiny_gpt_oss(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
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
    tiny_gpt_oss = AutoModelForCausalLM.from_config(GptOssConfig(**kwargs))

    return tiny_gpt_oss


##### LLAMA #####
def get_tiny_llama(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)
    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(**config_kwargs)
    tiny_llama = AutoModelForCausalLM.from_config(LlamaConfig(**kwargs))

    return tiny_llama


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

    get_tiny_llama(**config_kwargs).save_pretrained(llama_dir)
    return llama_dir


##### T5 #####
def get_tiny_t5(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)
    kwargs = {
        "dtype": torch.bfloat16,
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
    t5_model = T5ForConditionalGeneration(T5Config(**kwargs)).to(torch.bfloat16)

    return t5_model


def create_tiny_t5_dir(tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs) -> Path:
    set_seed(SEED)
    t5_dir = Path(tmp_path) / "tiny_t5"
    if with_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-T5Model")
        tokenizer.save_pretrained(t5_dir)
        config_kwargs["vocab_size"] = tokenizer.vocab_size

    get_tiny_t5(**config_kwargs).save_pretrained(t5_dir)
    return t5_dir


##### BERT #####
def get_tiny_bert(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(**config_kwargs)
    tiny_bert = AutoModelForQuestionAnswering.from_config(BertConfig(**kwargs))

    return tiny_bert


def create_tiny_bert_dir(tmp_path: Path | str, **config_kwargs) -> Path:
    set_seed(SEED)
    bert_dir = Path(tmp_path) / "tiny_bert"
    get_tiny_bert(**config_kwargs).save_pretrained(bert_dir)
    return bert_dir


##### TESTERS #####
def tf_output_tester(model_ref, model_test):
    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.to(model_ref.dtype).eval()
    output_ref = model_ref(**inputs)
    output_test = model_test(**inputs)
    atol = 1e-2 if model_ref.dtype == torch.bfloat16 else 1e-6
    if hasattr(output_ref, "logits"):
        assert torch.allclose(output_ref.logits, output_test.logits, atol=atol)
    else:
        assert torch.allclose(output_ref.start_logits, output_test.start_logits, atol=atol)
        assert torch.allclose(output_ref.end_logits, output_test.end_logits, atol=atol)


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
