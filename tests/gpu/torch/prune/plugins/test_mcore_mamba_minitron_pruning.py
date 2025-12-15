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
from _test_utils.torch.megatron.utils import run_mcore_inference_with_dummy_input
from megatron.core.ssm.mamba_layer import MambaLayer

import modelopt.torch.prune as mtp


def _test_mcore_mamba_hybrid_pruning(ckpt_path, rank, size):
    num_layers = min(size * 2, 8)
    hidden_size = 256
    ffn_hidden_size = 128
    num_attention_heads = 8
    num_query_groups = 4
    mamba_state_dim = 64
    mamba_head_dim = 16
    mamba_num_groups = 2
    num_moe_experts = 8
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
    mtp.prune(
        model,
        mode="mcore_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"forward_loop": forward_loop, "scores_path": ckpt_path},
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
    mtp.prune(
        model,
        mode="mcore_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"scores_path": ckpt_path},
    )


def test_mcore_mamba_hybrid_pruning(tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_mamba_hybrid_pruning, tmp_path / "modelopt_minitron_scores.pth"),
        backend="nccl",
    )
