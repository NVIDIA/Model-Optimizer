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

from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import run_mcore_inference
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer

import modelopt.torch.nas as mtn
from modelopt.torch.nas.modules import DynamicModuleList
from modelopt.torch.nas.plugins.megatron import (
    _DynamicColumnParallelLinear,
    _DynamicEmbedding,
    _DynamicLanguageModelEmbedding,
    _DynamicMCoreLanguageModel,
    _DynamicMLP,
    _DynamicMoELayer,
    _DynamicProjRowParallelLinear,
    _DynamicQKVColumnParallelLinear,
    _DynamicRowParallelLinear,
    _DynamicSelfAttention,
    _DynamicSequentialMLP,
    _DynamicTopKRouter,
    _DynamicTransformerLayer,
    expand_head_indices,
)
from modelopt.torch.opt.utils import named_dynamic_modules, search_space_size
from modelopt.torch.prune.plugins.mcore_minitron import get_mcore_minitron_config
from modelopt.torch.utils.random import centroid

SEED = 1234


def _test_gpt_search_space(
    num_attention_heads, num_query_groups, activation_func, normalization, rank, size
):
    channel_divisor = 4

    num_layers = min(size * 2, 8)
    hidden_size = channel_divisor * 4
    ffn_hidden_size = channel_divisor * 2
    max_sequence_length = 8
    vocab_size = 32
    batch_size = 2

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
    ).cuda()

    model = mtn.convert(model, [("mcore_minitron", get_mcore_minitron_config(channel_divisor))])

    assert isinstance(model, _DynamicMCoreLanguageModel)
    for m in model.modules():
        if isinstance(m, LanguageModelEmbedding):
            assert isinstance(m, _DynamicLanguageModelEmbedding)
            assert isinstance(m.word_embeddings, _DynamicEmbedding)
        elif isinstance(m, TransformerLayer):
            assert isinstance(m, _DynamicTransformerLayer)
        elif isinstance(m, MLP):
            assert isinstance(m, _DynamicMLP)
            assert isinstance(m.linear_fc1, _DynamicColumnParallelLinear)
            assert isinstance(m.linear_fc2, _DynamicRowParallelLinear)
        elif isinstance(m, SelfAttention):
            assert isinstance(m, _DynamicSelfAttention)
            assert isinstance(m.linear_qkv, _DynamicQKVColumnParallelLinear)
            assert isinstance(m.linear_proj, _DynamicProjRowParallelLinear)

    # NOTE: `search_space_size` does not reduce across TP/PP groups
    ss_size_per_pp = search_space_size(model)
    num_heads_choices = num_attention_heads // num_query_groups
    ffn_hidden_size_choices = ffn_hidden_size // channel_divisor
    hidden_size_choices = hidden_size // channel_divisor
    num_layers_per_pp = num_layers // size
    assert (
        ss_size_per_pp
        == (num_heads_choices * ffn_hidden_size_choices) ** num_layers_per_pp
        * num_layers
        * hidden_size_choices
    )

    # Make sure forward pass works on min and centroid subnets
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    for sample_func in [min, max, centroid]:
        mtn.sample(model, sample_func)
        output = run_mcore_inference(model, prompt_tokens)
        assert output.shape == (batch_size, max_sequence_length, vocab_size)

    # Make sure export and forward pass works on centroid model
    mtn.export(model)
    _ = run_mcore_inference(model, prompt_tokens, model.hidden_size)
    assert not any(named_dynamic_modules(model))


@pytest.mark.parametrize(
    ("num_attention_heads", "num_query_groups", "activation_func", "normalization"),
    [
        (8, 8, "squared_relu", "LayerNorm"),  # MHA
        (8, 4, "swiglu", "RMSNorm"),  # GQA
        # (8, 1, "swiglu", "RMSNorm"),  # MQA
    ],
)
def test_gpt_search_space(num_attention_heads, num_query_groups, activation_func, normalization):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_gpt_search_space,
            num_attention_heads,
            num_query_groups,
            activation_func,
            normalization,
        ),
        backend="nccl",
    )


def test_expand_head_indices():
    heads = torch.LongTensor([1, 3, 2, 0])
    hidden_size_per_head = 2
    assert expand_head_indices(heads, hidden_size_per_head).tolist() == [2, 3, 6, 7, 4, 5, 0, 1]


def _test_gpt_moe_search_space(rank, size):
    channel_divisor = 4

    num_layers = min(size * 2, 8)
    hidden_size = channel_divisor * 4
    num_attention_heads = 8
    num_query_groups = 4
    moe_ffn_hidden_size = channel_divisor * 2
    num_moe_experts = 4
    moe_shared_expert_intermediate_size = channel_divisor * 4
    max_sequence_length = 8
    vocab_size = 32
    batch_size = 2

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="squared_relu",
        num_moe_experts=num_moe_experts,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
    ).cuda()

    model = mtn.convert(model, [("mcore_minitron", get_mcore_minitron_config(channel_divisor))])

    moe = model.decoder.layers[0].mlp
    assert isinstance(moe, _DynamicMoELayer)
    assert isinstance(moe.router, _DynamicTopKRouter)
    assert isinstance(moe.experts, _DynamicSequentialMLP)
    assert isinstance(moe.experts.local_experts, DynamicModuleList)
    for expert in moe.experts.local_experts:
        assert isinstance(expert, _DynamicMLP)
    assert isinstance(moe.shared_experts, _DynamicMLP)

    # NOTE: `search_space_size` does not reduce across TP/PP groups
    ss_size_per_pp = search_space_size(model)
    num_heads_choices = num_attention_heads // num_query_groups
    moe_ffn_choices = moe_ffn_hidden_size // channel_divisor
    moe_shared_ffn_choices = moe_shared_expert_intermediate_size // channel_divisor
    hidden_size_choices = hidden_size // channel_divisor
    num_layers_per_pp = num_layers // size
    assert (
        ss_size_per_pp
        == (
            num_heads_choices
            * num_moe_experts
            * moe_ffn_choices**num_moe_experts
            * moe_shared_ffn_choices
        )
        ** num_layers_per_pp
        * num_layers
        * hidden_size_choices
    )

    # Make sure forward pass works on min and centroid subnets
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    output = run_mcore_inference(model, prompt_tokens)
    assert output.shape == (batch_size, max_sequence_length, vocab_size)

    # Make sure export and forward pass works on centroid model
    mtn.sample(model, centroid)
    mtn.export(model)
    _ = run_mcore_inference(model, prompt_tokens, model.hidden_size)
    assert not any(named_dynamic_modules(model))


def test_gpt_moe_search_space():
    spawn_multiprocess_job(
        size=torch.cuda.device_count(), job=_test_gpt_moe_search_space, backend="nccl"
    )
