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

import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True, mamba_required=True)

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_mamba_hybrid_model
from _test_utils.torch.megatron.utils import (
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)
from _test_utils.torch.misc import compare_outputs, set_seed
from _test_utils.torch.nas_prune.minitron_common import prune_minitron
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.identity_op import IdentityOp

import modelopt.torch.nas as mtn
from modelopt.torch.prune.plugins.mcore_minitron import (
    ImportanceEstimatorRegistry,
    _convert_model_to_dynamic_space,
    get_mcore_minitron_config,
)

SEED = 1234


def _test_mcore_mamba_parameter_sorting(rank, size):
    # Use relatively bigger model here for more accurate test for sorting
    channel_divisor = 64

    num_layers = size
    hybrid_override_pattern = "M" * size
    hidden_size = channel_divisor * 4
    mamba_state_dim = channel_divisor
    mamba_head_dim = 16
    mamba_num_groups = 2
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    model = get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hybrid_override_pattern=hybrid_override_pattern,
        hidden_size=hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        bf16=False,
    ).cuda()

    # Randomize norm weights instead of all zeros or ones
    for n, m in model.named_modules():
        if "norm" in n and not isinstance(m, IdentityOp):
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

    # check if all mamba_num_heads, mamba_head_dim, hidden_size have been sorted
    sortable_per_pp = [
        n for n, hp in dynamic_space.named_hparams(configurable=True) if hp.importance is not None
    ]
    # 2 mamba hps per layer + 1 for hidden_size (num_layers is not sorted!)
    assert len(sortable_per_pp) == 2 * num_layers // size + 1

    # sanity check if the model functionality is preserved after sorting
    y2 = run_mcore_inference(model, prompt_tokens)

    # check if the inference results after sorting is the same
    compare_outputs(y1, y2, rtol=1e-5, atol=1e-3)


def test_mcore_mamba_parameter_sorting():
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=_test_mcore_mamba_parameter_sorting,
        backend="nccl",
    )


def _test_mcore_mamba_hybrid_pruning(ckpt_path, rank, size):
    channel_divisor = 4

    num_layers = min(size * 2, 8)
    hidden_size = channel_divisor * 8
    ffn_hidden_size = channel_divisor * 2
    num_attention_heads = 8
    num_query_groups = 4
    mamba_state_dim = channel_divisor * 2
    mamba_head_dim = channel_divisor * 2
    mamba_num_groups = 2
    num_moe_experts = 8
    vocab_size = 32
    batch_size = 2

    def _get_model(initialize_megatron=True):
        model = get_mcore_mamba_hybrid_model(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=size,
            initialize_megatron=initialize_megatron,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            ffn_hidden_size=ffn_hidden_size,
            mamba_state_dim=mamba_state_dim,
            mamba_head_dim=mamba_head_dim,
            mamba_num_groups=mamba_num_groups,
            moe_ffn_hidden_size=ffn_hidden_size,
            moe_shared_expert_intermediate_size=ffn_hidden_size,
            num_moe_experts=num_moe_experts,
            vocab_size=vocab_size,
        ).cuda()
        return model

    model = _get_model()

    mamba_layer = None
    for layer in model.decoder.layers:
        if isinstance(layer, MambaLayer):
            mamba_layer = layer
            break
    assert mamba_layer is not None, f"No MambaLayer found in the model PP rank {rank}!"
    mamba_num_heads = mamba_layer.mixer.nheads

    def forward_loop(m):
        for _ in range(5):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

    # Traditional GPT pruning parameters
    pruned_ffn_hidden_size = ffn_hidden_size // 2
    pruned_num_attention_heads = num_attention_heads // 2
    pruned_hidden_size = hidden_size // 2
    pruned_num_moe_experts = num_moe_experts // 2

    # Mamba-specific pruning parameters
    pruned_mamba_num_heads = mamba_num_heads // 2
    pruned_mamba_head_dim = mamba_head_dim // 2

    # Base export config with GPT/Attention parameters
    export_config = {
        "ffn_hidden_size": pruned_ffn_hidden_size,
        "num_attention_heads": pruned_num_attention_heads,
        "hidden_size": pruned_hidden_size,
        "mamba_num_heads": pruned_mamba_num_heads,
        "mamba_head_dim": pruned_mamba_head_dim,
        "moe_ffn_hidden_size": pruned_ffn_hidden_size,
        "moe_shared_expert_intermediate_size": pruned_ffn_hidden_size,
        "num_moe_experts": pruned_num_moe_experts,
    }
    prune_minitron(
        model,
        export_config,
        {"forward_loop": forward_loop, "scores_path": ckpt_path},
        channel_divisor,
    )

    # Assert weights are pruned correctly
    mixer = mamba_layer.mixer
    bc = 2 * mixer.ngroups * mixer.d_state
    assert mixer.nheads == pruned_mamba_num_heads
    assert mixer.headdim == pruned_mamba_head_dim
    assert mixer.in_proj.input_size == pruned_hidden_size
    assert mixer.d_inner == pruned_mamba_num_heads * pruned_mamba_head_dim
    assert mixer.in_proj.output_size == 2 * mixer.d_inner + bc + pruned_mamba_num_heads
    assert mixer.out_proj.input_size == mixer.d_inner
    assert mixer.out_proj.output_size == pruned_hidden_size
    assert mixer.conv1d.in_channels == mixer.conv1d.out_channels == mixer.d_inner + bc

    # Assert model.config is updated for correct save/restoring
    assert model.config.ffn_hidden_size == pruned_ffn_hidden_size
    assert model.config.num_attention_heads == pruned_num_attention_heads
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.mamba_num_heads == pruned_mamba_num_heads
    assert model.config.mamba_head_dim == pruned_mamba_head_dim
    assert model.config.moe_ffn_hidden_size == pruned_ffn_hidden_size
    assert model.config.moe_shared_expert_intermediate_size == pruned_ffn_hidden_size
    assert model.config.num_moe_experts == pruned_num_moe_experts

    # Assert forward pass works on the pruned model
    run_mcore_inference_with_dummy_input(model, batch_size, pruned_hidden_size)

    # Assert re-pruning from scores_path works without running the forward loop again
    model = _get_model(initialize_megatron=False)
    prune_minitron(model, export_config, {"scores_path": ckpt_path}, channel_divisor)


def test_mcore_mamba_hybrid_pruning(tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_mamba_hybrid_pruning, tmp_path / "modelopt_minitron_scores.pth"),
        backend="nccl",
    )
