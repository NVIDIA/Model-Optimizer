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
from _test_utils.torch.misc import compare_outputs

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)
from _test_utils.torch.misc import set_seed
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.parallel_state import destroy_model_parallel
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer

import modelopt.torch.nas as mtn
from modelopt.torch.nas.conversion import export_searchspace
from modelopt.torch.nas.modules import DynamicModuleList
from modelopt.torch.nas.plugins.megatron import (
    NumAttentionHeadsHp,
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
from modelopt.torch.prune.plugins.mcore_minitron import _convert_model_to_dynamic_space
from modelopt.torch.utils.random import centroid

SEED = 1234


def _test_gpt_search_space(
    num_attention_heads, num_query_groups, activation_func, normalization, rank, size
):
    channel_divisor = 64

    num_layers = min(size * 2, 8)
    hidden_size = 256
    ffn_hidden_size = 128
    max_sequence_length = 16
    vocab_size = 64
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

    model = mtn.convert(model, "mcore_minitron")

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


def _test_gpt_parameter_sorting(activation_func, rank, size):
    num_layers = size
    hidden_size = 128
    num_attention_heads = 8
    num_query_groups = 4
    ffn_hidden_size = 64
    max_sequence_length = 32
    vocab_size = 128
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
        bf16=False,
    ).cuda()

    # Randomize layernorm weights instead of all zeros or ones
    for n, m in model.named_modules():
        if "layernorm" in n and not isinstance(m, IdentityOp):
            m.weight.data = torch.randn_like(m.weight)

    model.eval()
    dynamic_space = _convert_model_to_dynamic_space(model)

    # Compute activations for sorting
    for _ in range(5):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    mtn.utils.sort_parameters(model)

    # check if all ffn_hidden_size, num_attention_heads, hidden_size have been sorted
    sortable_per_pp = [
        n for n, hp in dynamic_space.named_hparams(configurable=True) if hp.importance is not None
    ]
    # 2 hps per layer (num_attention_heads, ffn_hidden_size) + 1 for hidden_size (num_layers is not sorted!)
    assert len(sortable_per_pp) == 2 * num_layers // size + 1

    # sanity check if the model functionality is preserved after sorting
    y2 = run_mcore_inference(model, prompt_tokens)

    # check if the inference results after sorting is the same
    compare_outputs(y1, y2, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("activation_func", ["swiglu"])
def test_gpt_parameter_sorting(activation_func, need_2_gpus):
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_gpt_parameter_sorting, activation_func),
        backend="nccl",
    )


def test_expand_head_indices():
    heads = torch.LongTensor([1, 3, 2, 0])
    hidden_size_per_head = 2
    assert expand_head_indices(heads, hidden_size_per_head).tolist() == [2, 3, 6, 7, 4, 5, 0, 1]


def test_self_attention_head_sorting(distributed_setup_size_1):
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=1,
        hidden_size=16,
        num_attention_heads=8,
        num_query_groups=2,
        ffn_hidden_size=16,
        activation_func="squared_relu",
    ).cuda()

    model = mtn.convert(model, "mcore_minitron")

    self_attn = model.decoder.layers[0].self_attention
    assert isinstance(self_attn, _DynamicSelfAttention)
    assert isinstance(self_attn.linear_qkv, _DynamicQKVColumnParallelLinear)
    assert isinstance(self_attn.linear_proj, _DynamicProjRowParallelLinear)

    hp_num_attention_heads = self_attn.get_hparam("num_attention_heads")
    assert isinstance(hp_num_attention_heads, NumAttentionHeadsHp)

    # Choices are multiples of num_query_groups (2): [2, 4, 6, 8]
    assert hp_num_attention_heads.choices == [2, 4, 6, 8]
    assert hp_num_attention_heads._num_query_groups == 2

    # Set importance and slice order
    # Importance per head (group-aware): [2.2, 0.1, 1.1, 2.1, 3.0, 2.0, 0.0, 1.0]
    # Group 0 (heads 0-3): [2.2, 0.1, 1.1, 2.1] → sorted: [0, 3, 2, 1]
    # Group 1 (heads 4-7): [3.0, 2.0, 0.0, 1.0] → sorted: [4, 5, 7, 6]
    # Global ranking (group-aware, flattened): [0, 3, 2, 1, 4, 5, 7, 6]
    hp_num_attention_heads._get_importance = lambda: torch.tensor(
        [2.2, 0.1, 1.1, 2.1, 3.0, 2.0, 0.0, 1.0]
    )
    # _estimate_head_ranking returns ranking as 1D tensor
    expected_ranking = torch.tensor([0, 3, 2, 1, 4, 5, 7, 6])
    hp_num_attention_heads.enforce_order(expected_ranking)

    assert hp_num_attention_heads.active_slice.tolist() == [0, 3, 2, 1, 4, 5, 7, 6]

    # check if we get correct selection of sorted + pruned heads after setting active values
    hp_num_attention_heads.active = 4  # top 2 heads per group (2 groups * 2 heads = 4 total)

    # Expected: Top 2 heads from each group: [0, 3] from group 0, [4, 5] from group 1
    expected_q_heads = [0, 3, 4, 5]
    # In QKV layout (4 heads/group → 6 QKV heads/group):
    # Group 0: Q=[0, 3], K=4, V=5 → QKV indices [0, 3, 4, 5]
    # Group 1: Q=[4, 5], K=10, V=11 → QKV indices [6, 7, 10, 11]
    expected_qkv_heads = [0, 3, 4, 5, 6, 7, 10, 11]

    assert (
        self_attn.linear_qkv._get_output_size_indices().tolist()
        == expand_head_indices(
            torch.LongTensor(expected_qkv_heads), model.config.kv_channels
        ).tolist()
    )
    assert (
        self_attn.linear_proj._get_input_size_indices().tolist()
        == expand_head_indices(
            torch.LongTensor(expected_q_heads), model.config.kv_channels
        ).tolist()
    )

    # Clean up since this is not a spawned process
    destroy_model_parallel()


def _test_gpt_moe_search_space(rank, size):
    channel_divisor = 64

    num_layers = min(size * 2, 8)
    hidden_size = 256
    num_attention_heads = 8
    num_query_groups = 4
    moe_ffn_hidden_size = 128
    num_moe_experts = 4
    moe_shared_expert_intermediate_size = 256
    max_sequence_length = 16
    vocab_size = 64
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

    model = mtn.convert(model, "mcore_minitron")

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


def _test_gpt_moe_parameter_sorting(rank, size):
    num_layers = min(size * 2, 8)
    hidden_size = 256
    num_attention_heads = 8
    num_query_groups = 4
    moe_ffn_hidden_size = 128
    num_moe_experts = 4
    moe_shared_expert_intermediate_size = 256
    max_sequence_length = 16
    vocab_size = 64
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
        bf16=False,
    ).cuda()

    # Randomize layernorm weights instead of all zeros or ones
    for n, m in model.named_modules():
        if "layernorm" in n and not isinstance(m, IdentityOp):
            m.weight.data = torch.randn_like(m.weight)

    model.eval()
    dynamic_space = _convert_model_to_dynamic_space(model)

    # Compute activations for sorting
    for _ in range(10):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    mtn.utils.sort_parameters(model)

    # check if all num_moe_experts, moe_ffn, moe_shared_ffn, num_attention_heads, hidden_size
    # have been sorted
    sortable_per_pp = [
        n for n, hp in dynamic_space.named_hparams(configurable=True) if hp.importance is not None
    ]
    # (num_moe_experts + 3) hps per layer + 1 for hidden_size (num_layers is not sorted!)
    # Per layer: num_attention_heads, num_moe_experts, moe_ffn (per expert), moe_shared_ffn
    assert len(sortable_per_pp) == (num_moe_experts + 3) * num_layers // size + 1

    # sanity check if the model functionality is preserved after sorting
    export_searchspace(model, mtn.get_subnet_config(model))
    y2 = run_mcore_inference(model, prompt_tokens)

    # check if the inference results after sorting is the same
    compare_outputs(y1, y2, rtol=1e-5, atol=1e-3)


def test_gpt_moe_parameter_sorting(need_2_gpus):
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=_test_gpt_moe_parameter_sorting,
        backend="nccl",
    )
