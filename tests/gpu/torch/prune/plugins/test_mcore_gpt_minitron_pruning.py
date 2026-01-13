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
from _test_utils.torch.megatron.utils import (
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)
from _test_utils.torch.misc import compare_outputs, set_seed
from _test_utils.torch.nas_prune.minitron_common import prune_minitron
from megatron.core.parallel_state import destroy_model_parallel
from megatron.core.transformer.identity_op import IdentityOp

import modelopt.torch.nas as mtn
from modelopt.torch.nas.conversion import export_searchspace
from modelopt.torch.nas.plugins.megatron import (
    NumAttentionHeadsHp,
    _DynamicProjRowParallelLinear,
    _DynamicQKVColumnParallelLinear,
    _DynamicSelfAttention,
    expand_head_indices,
)
from modelopt.torch.prune.plugins.mcore_minitron import (
    ImportanceEstimatorRegistry,
    _convert_model_to_dynamic_space,
    get_mcore_minitron_config,
)

SEED = 1234


def _assert_approx(actual, expected, abs=1e-3):
    assert actual == pytest.approx(expected, abs=abs), f"{actual=} != {expected=}"


def _test_mcore_gpt_parameter_sorting(activation_func, rank, size):
    # Use relatively bigger model here for more accurate test for sorting
    channel_divisor = 64

    num_layers = size
    hidden_size = channel_divisor * 2
    num_attention_heads = 8
    num_query_groups = 4
    ffn_hidden_size = channel_divisor * 2
    max_sequence_length = 32
    vocab_size = channel_divisor * 2
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
    dynamic_space = _convert_model_to_dynamic_space(
        model, get_mcore_minitron_config(channel_divisor)
    )
    registry = ImportanceEstimatorRegistry(model)  # register imp estimators and forward hooks

    # Compute activations for sorting
    for _ in range(5):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    mtn.utils.sort_parameters(model)
    registry.cleanup()

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
def test_mcore_gpt_parameter_sorting(activation_func):
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_gpt_parameter_sorting, activation_func),
        backend="nccl",
    )


def test_mcore_gpt_self_attention_head_sorting(distributed_setup_size_1):
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


def _test_mcore_gpt_pruning(
    num_attention_heads,
    num_query_groups,
    activation_func,
    normalization,
    pruned_ffn_div,
    pruned_num_attention_heads_div,
    pruned_hidden_size_div,
    pruned_num_layers_div,
    uneven_pp,
    position_embedding_type,
    skip_sorting,
    ckpt_path,
    rank,
    size,
):
    channel_divisor = 4

    hidden_size = channel_divisor * 4
    ffn_hidden_size = channel_divisor * 4
    max_sequence_length = 8
    vocab_size = 16
    batch_size = 2

    num_layers = min(size * 2, 8)
    num_layers_in_first_pipeline_stage = None
    num_layers_in_last_pipeline_stage = None
    if uneven_pp and size > 1:
        num_layers = size * 2
        if size == 2:  # [1, 3]
            num_layers_in_first_pipeline_stage = 1
        elif size == 4:  # [3, 2, 2, 1]
            num_layers_in_first_pipeline_stage = 3
            num_layers_in_last_pipeline_stage = 1
        elif size == 8:  # [4, 1, 1, 1, 1, 1, 1, 6]
            num_layers_in_first_pipeline_stage = 4
            num_layers_in_last_pipeline_stage = 6
        else:
            raise ValueError(f"Unsupported size {size}")

    def _get_model(initialize_megatron=True):
        model = get_mcore_gpt_model(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=size,
            initialize_megatron=initialize_megatron,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            ffn_hidden_size=ffn_hidden_size,
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size,
            position_embedding_type=position_embedding_type,
            activation_func=activation_func,
            normalization=normalization,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
            use_cpu_initialization=True,  # Ensure deterministic weight init across CUDA versions
        ).cuda()
        return model

    model = _get_model()

    sd = model.state_dict()

    def forward_loop(m):
        for _ in range(5):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

    pruned_ffn = ffn_hidden_size // pruned_ffn_div
    pruned_num_attention_heads = num_attention_heads // pruned_num_attention_heads_div
    pruned_num_heads_per_group = pruned_num_attention_heads // num_query_groups
    pruned_hidden_size = hidden_size // pruned_hidden_size_div
    pruned_num_layers = num_layers // pruned_num_layers_div

    export_config = {}
    if pruned_ffn_div != 1:
        export_config["ffn_hidden_size"] = pruned_ffn
    if pruned_num_attention_heads_div != 1:
        export_config["num_attention_heads"] = pruned_num_attention_heads
    if pruned_hidden_size_div != 1:
        export_config["hidden_size"] = pruned_hidden_size
    if pruned_num_layers_div != 1:
        export_config["num_layers"] = pruned_num_layers

    config = {
        "scores_path": ckpt_path,
        "skip_sorting": skip_sorting,
    }
    if skip_sorting:
        assert ckpt_path is None
    else:
        config["forward_loop"] = forward_loop
    model, pruning_scores = prune_minitron(model, export_config, config, channel_divisor)
    if not skip_sorting:
        assert pruning_scores["layer_scores"]
        assert pruning_scores["activations_per_rank"]

        # TODO: Simplify it: this unit test is too long,
        # hard to read (the same set of assertions across different test cases with if-else).

        assert len(pruning_scores["activations_per_rank"]) == size
        activations = pruning_scores["activations_per_rank"][rank]

        # Test case 1: MHA - pruned ffn/4 (num_attention_heads=8, num_query_groups=8, ffn_div=4)
        if size == 1 and pruned_ffn_div == 4:
            # Layer scores
            _assert_approx(pruning_scores["layer_scores"], {1: 0.028923, 2: 0.046508})

            # Validate decoder.layers.0.mlp activations
            mlp_0_acts = activations["decoder.layers.0.mlp"]
            _assert_approx(mlp_0_acts.min().item(), 0.000026)
            _assert_approx(mlp_0_acts.max().item(), 0.000729)
            _assert_approx(mlp_0_acts.mean().item(), 0.000201)

            # Validate decoder.layers.1.mlp activations
            mlp_1_acts = activations["decoder.layers.1.mlp"]
            _assert_approx(mlp_1_acts.min().item(), 0.000022)
            _assert_approx(mlp_1_acts.max().item(), 0.000762)
            _assert_approx(mlp_1_acts.mean().item(), 0.000162)

        # Test case 2: GQA - pruned attention/2 (num_attention_heads=8, num_query_groups=4, attention_div=2)
        elif size == 1 and pruned_num_attention_heads_div == 2 and pruned_ffn_div == 1:
            # Layer scores
            _assert_approx(pruning_scores["layer_scores"], {1: 0.028056, 2: 0.038353})

            # Validate decoder.layers.0.self_attention activations
            attn_0_acts = activations["decoder.layers.0.self_attention"]
            assert attn_0_acts.shape == torch.Size([hidden_size])
            _assert_approx(attn_0_acts.min().item(), 0.010091)
            _assert_approx(attn_0_acts.max().item(), 0.023826)
            _assert_approx(attn_0_acts.mean().item(), 0.014548)

            # Validate decoder.layers.1.self_attention activations
            attn_1_acts = activations["decoder.layers.1.self_attention"]
            assert attn_1_acts.shape == torch.Size([hidden_size])
            _assert_approx(attn_1_acts.min().item(), 0.009982)
            _assert_approx(attn_1_acts.max().item(), 0.035644)
            _assert_approx(attn_1_acts.mean().item(), 0.020140)

    # Assert weights are pruned correctly
    for layer in model.decoder.layers:
        assert layer.mlp.linear_fc1.weight.shape == (
            pruned_ffn * (2 if activation_func == "swiglu" else 1),
            pruned_hidden_size,
        )
        assert layer.mlp.linear_fc2.weight.shape == (pruned_hidden_size, pruned_ffn)
        assert layer.self_attention.linear_qkv.weight.shape == (
            (pruned_num_heads_per_group + 2) * num_query_groups * model.config.kv_channels,
            pruned_hidden_size,
        )
        assert layer.self_attention.linear_proj.weight.shape == (
            pruned_hidden_size,
            pruned_num_heads_per_group * num_query_groups * model.config.kv_channels,
        )

    # Assert model.config is updated for correct save/restoring
    assert model.config.ffn_hidden_size == pruned_ffn
    assert model.config.num_attention_heads == pruned_num_attention_heads
    assert model.config.num_query_groups == num_query_groups
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.num_layers == pruned_num_layers

    # Assert forward pass works on the pruned model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    output = run_mcore_inference(model, prompt_tokens, pruned_hidden_size)

    # Assert re-pruning from scores_path works without running the forward loop again
    if ckpt_path:
        model_rerun = _get_model(initialize_megatron=False)
        model_rerun.load_state_dict(sd)
        model_rerun, pruning_scores = prune_minitron(
            model_rerun, export_config, {"scores_path": ckpt_path}, channel_divisor
        )

        output_rerun = run_mcore_inference(model_rerun, prompt_tokens, pruned_hidden_size)
        assert torch.allclose(output, output_rerun, atol=1e-5)


@pytest.mark.parametrize(
    (
        "num_attention_heads",
        "num_query_groups",
        "activation_func",
        "normalization",
        "ffn_div",
        "num_attention_heads_div",
        "hidden_size_div",
        "num_layers_div",
        "uneven_pp",
        "position_embedding_type",
        "skip_sorting",
        "test_ckpt",
    ),
    [
        # MHA - pruned ffn/4
        (8, 8, "squared_relu", "LayerNorm", 4, 1, 1, 1, False, "rope", False, False),
        # GQA - pruned attention/2
        (8, 4, "squared_relu", "RMSNorm", 1, 2, 1, 1, False, "rope", False, False),
        # GQA - pruned hidden_size/4
        (8, 4, "swiglu", "RMSNorm", 1, 1, 4, 1, False, "rope", True, False),
        # MHA - pruned num_layers/2
        (8, 8, "swiglu", "LayerNorm", 1, 1, 1, 2, False, "rope", False, False),
        # GQA - pruned all/2, uneven pp
        (8, 4, "swiglu", "RMSNorm", 2, 2, 2, 2, True, "yarn", False, True),
    ],
)
def test_mcore_gpt_pruning(
    tmp_path,
    num_attention_heads,
    num_query_groups,
    activation_func,
    normalization,
    ffn_div,
    num_attention_heads_div,
    hidden_size_div,
    num_layers_div,
    uneven_pp,
    position_embedding_type,
    skip_sorting,
    test_ckpt,
):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_mcore_gpt_pruning,
            num_attention_heads,
            num_query_groups,
            activation_func,
            normalization,
            ffn_div,
            num_attention_heads_div,
            hidden_size_div,
            num_layers_div,
            uneven_pp,
            position_embedding_type,
            skip_sorting,
            tmp_path / "minitron_scores.pth" if test_ckpt else None,
        ),
        backend="nccl",
    )


def _test_mcore_gpt_moe_parameter_sorting(rank, size):
    # Use relatively bigger model here for more accurate test for sorting
    channel_divisor = 64

    num_layers = min(size * 2, 8)
    hidden_size = channel_divisor * 4
    num_attention_heads = 8
    num_query_groups = 4
    moe_ffn_hidden_size = channel_divisor * 2
    num_moe_experts = 4
    moe_shared_expert_intermediate_size = channel_divisor * 4
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
    dynamic_space = _convert_model_to_dynamic_space(
        model, get_mcore_minitron_config(channel_divisor)
    )
    registry = ImportanceEstimatorRegistry(model)  # register imp estimators and forward hooks

    # Compute activations for sorting
    for _ in range(10):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    mtn.utils.sort_parameters(model)
    registry.cleanup()

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


def test_mcore_gpt_moe_parameter_sorting():
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=_test_mcore_gpt_moe_parameter_sorting,
        backend="nccl",
    )


def _test_mcore_gpt_pruning_moe(ckpt_path, rank, size):
    channel_divisor = 4

    num_layers = size
    hidden_size = channel_divisor * 4
    moe_ffn_hidden_size = channel_divisor * 2
    num_moe_experts = 4
    moe_shared_expert_intermediate_size = channel_divisor * 4
    max_sequence_length = 8
    vocab_size = 16
    batch_size = 2

    def _get_model(initialize_megatron=True):
        model = get_mcore_gpt_model(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=size,
            initialize_megatron=initialize_megatron,
            num_layers=num_layers,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size,
            activation_func="squared_relu",
            num_moe_experts=num_moe_experts,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        ).cuda()
        return model

    model = _get_model()
    sd = model.state_dict()

    def forward_loop(m):
        for _ in range(5):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

    pruned_hidden_size = hidden_size // 2
    pruned_moe_ffn = moe_ffn_hidden_size // 2
    pruned_moe_shared_ffn = moe_shared_expert_intermediate_size // 2
    pruned_num_moe_experts = num_moe_experts // 2

    export_config = {
        "hidden_size": pruned_hidden_size,
        "moe_ffn_hidden_size": pruned_moe_ffn,
        "moe_shared_expert_intermediate_size": pruned_moe_shared_ffn,
        "num_moe_experts": pruned_num_moe_experts,
    }

    prune_minitron(
        model,
        export_config,
        {"scores_path": ckpt_path, "forward_loop": forward_loop},
        channel_divisor,
    )

    # Assert weights are pruned correctly
    for layer in model.decoder.layers:
        moe = layer.mlp
        assert moe.router.num_experts == pruned_num_moe_experts
        assert moe.router.expert_bias.shape == (pruned_num_moe_experts,)
        assert moe.router.weight.shape == (pruned_num_moe_experts, pruned_hidden_size)
        assert moe.experts.num_local_experts == pruned_num_moe_experts
        assert len(moe.experts.local_experts) == pruned_num_moe_experts
        for expert in moe.experts.local_experts:
            assert expert.linear_fc1.weight.shape == (pruned_moe_ffn, pruned_hidden_size)
            assert expert.linear_fc2.weight.shape == (pruned_hidden_size, pruned_moe_ffn)
        assert moe.shared_experts.linear_fc1.weight.shape == (
            pruned_moe_shared_ffn,
            pruned_hidden_size,
        )
        assert moe.shared_experts.linear_fc2.weight.shape == (
            pruned_hidden_size,
            pruned_moe_shared_ffn,
        )

    # Assert model.config is updated for correct save/restoring
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.moe_ffn_hidden_size == pruned_moe_ffn
    assert model.config.num_moe_experts == pruned_num_moe_experts
    assert model.config.moe_shared_expert_intermediate_size == pruned_moe_shared_ffn

    # Assert forward pass works on the pruned model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    output = run_mcore_inference(model, prompt_tokens, pruned_hidden_size)

    # Assert re-pruning from scores_path works without running the forward loop again
    model_rerun = _get_model(initialize_megatron=False)
    model_rerun.load_state_dict(sd)
    prune_minitron(model_rerun, export_config, {"scores_path": ckpt_path}, channel_divisor)

    output_rerun = run_mcore_inference(model_rerun, prompt_tokens, pruned_hidden_size)
    assert torch.allclose(output, output_rerun, atol=1e-5)


def test_mcore_gpt_pruning_moe(tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_gpt_pruning_moe, tmp_path / "minitron_scores.pth"),
        backend="nccl",
    )
