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
# mypy: ignore-errors

"""Calculate memory usage and parameter counts for neural network subblocks.

This module provides utilities to compute memory footprints and parameter counts
for different subblock types (FFN, Attention, Mamba, MoE) in large language models,
considering various data types, batch sizes, and sequence lengths.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    FFNConfig,
    MambaConfig,
)
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMMoe
from modelopt.torch.puzzletron.utils.utils import (
    calculate_kv_dim,
    raise_unknown_subblock_config_error,
    sizeof_dtype,
)


def calculate_subblock_memory(
    subblock_config: FFNConfig | AttentionConfig,
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    prefill_queue_size: int,
    n_embd: int,
    n_head: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    allocate_prefill_query: bool,
) -> float | dict[str, float]:
    if subblock_config.no_op:
        return 0
    if subblock_config.replace_with_linear:
        return calculate_linear_memory(n_embd, weights_dtype)
    if isinstance(subblock_config, FFNConfig):
        return calculate_ffn_memory(subblock_config, n_embd, weights_dtype)
    if isinstance(subblock_config, AttentionConfig):
        if subblock_config.is_mamba:
            return calculate_mamba_memory(
                subblock_config.mamba, n_embd, batch_size, weights_dtype, kv_cache_dtype
            )
        else:
            return calculate_attention_memory(
                subblock_config,
                batch_size,
                prefill_seq_len,
                generation_seq_len,
                prefill_queue_size,
                n_embd,
                n_head,
                weights_dtype,
                kv_cache_dtype,
                allocate_prefill_query,
            )
    raise_unknown_subblock_config_error(subblock_config)


def calculate_subblock_params(
    subblock_config: FFNConfig | AttentionConfig,
    n_embd: int,
    n_head: int,
) -> int:
    if subblock_config.no_op:
        return 0
    if subblock_config.replace_with_linear:
        return calculate_linear_params(n_embd)
    if isinstance(subblock_config, FFNConfig):
        return calculate_ffn_params(subblock_config, n_embd)
    if isinstance(subblock_config, AttentionConfig):
        if subblock_config.is_mamba:
            return calculate_mamba_params(subblock_config.mamba, n_embd)
        else:
            return calculate_attention_params(subblock_config, n_embd, n_head)
    raise_unknown_subblock_config_error(subblock_config)


def calc_subblock_active_params(
    subblock_config: FFNConfig | AttentionConfig,
    n_embd: int,
    n_head: int,
    moe_stats_file: str,
    batch_size: int,
    block_idx: int,
) -> int:
    if not (isinstance(subblock_config, FFNConfig) and subblock_config.is_moe):
        return calculate_subblock_params(subblock_config, n_embd, n_head)
    else:
        return estimate_moe_active_params(
            subblock_config, n_embd, moe_stats_file, batch_size, block_idx
        )


def load_moe_stats(stats_file: str) -> dict:
    with open(stats_file) as f:
        stats = json.load(f)
    return [np.array(l) / np.sum(l) if len(l) > 0 else 0 for l in stats]


def estimate_num_active_experts(
    dist_over_experts: np.ndarray, batch_size: int, num_experts: int
) -> int:
    # cut the tail and renormalize
    dist_over_experts = np.sort(dist_over_experts)[::-1][:num_experts]
    dist_over_experts = dist_over_experts / (dist_over_experts.sum())
    # calculate the probability of at least one expert being active
    # (expectation on indicators is the expected number of active experts)
    return (1 - (1 - dist_over_experts) ** batch_size).sum()


def estimate_moe_active_params(
    subblock_config: FFNConfig,
    n_embd: int,
    moe_stats_file: Path | str,
    batch_size: int,
    block_idx: int,
) -> int:
    assert Path(moe_stats_file).exists()
    # if not Path(moe_stats_file).exists(): # if path is not provided, should we assume uniform distribution?
    #     return calculate_subblock_params(subblock_config, n_embd, n_head=None)
    moe_stats = load_moe_stats(moe_stats_file)
    dist_over_experts = moe_stats[block_idx]
    num_experts = subblock_config.moe.num_local_experts

    expected_num_active_experts = estimate_num_active_experts(
        dist_over_experts, batch_size, num_experts
    )
    expert_dim = subblock_config.moe.expert_intermediate_dim
    shared_expert_dim = subblock_config.moe.shared_expert_intermediate_dim
    num_linear_layers = 3  # all moe experts have 3 linear layers

    router_num_params = n_embd * num_experts
    expected_num_active_experts_params = (
        num_linear_layers * expert_dim * n_embd * expected_num_active_experts
    )
    shared_expert_num_params = num_linear_layers * shared_expert_dim * n_embd

    expected_total_params = (
        router_num_params + expected_num_active_experts_params + shared_expert_num_params
    )
    return expected_total_params


def calculate_attention_memory(
    attention_config: AttentionConfig,
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    prefill_queue_size: int,
    n_embd: int,
    n_head: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    allocate_prefill_query: bool,
) -> dict[str, float]:
    """allocate_prefill_query: infery-llm style.
    Infery used a unified Wqkv matrix, so before extracting the kv-cache,
    the query also had to be kept in-memory, once per layer.
    """
    seq_len = prefill_seq_len + generation_seq_len
    if (
        attention_config.is_llama4
        and (attention_chunk_size := attention_config.llama4.attention_chunk_size) is not None
    ):
        seq_len = min(seq_len, attention_chunk_size)

    kv_dim = calculate_kv_dim(attention_config.n_heads_in_group, n_head, n_embd)
    total_num_tokens = seq_len * (batch_size + prefill_queue_size)
    kv_cache_size = total_num_tokens * kv_dim
    query_prefill_size = seq_len * n_embd if allocate_prefill_query else 0
    num_params = calculate_attention_params(attention_config, n_embd, n_head)
    total_memory = (
        kv_cache_size * sizeof_dtype(kv_cache_dtype)
        + query_prefill_size * sizeof_dtype(weights_dtype)
        + num_params * sizeof_dtype(weights_dtype)
    ) / 2**20
    kv_cache_memory = kv_cache_size * sizeof_dtype(kv_cache_dtype) / 2**20
    return {"memory_mib": total_memory, "kv_cache_memory_mib": kv_cache_memory}


def calculate_attention_params(
    attention_config: AttentionConfig,
    n_embd: int,
    n_head: int,
) -> int:
    kv_dim = calculate_kv_dim(attention_config.n_heads_in_group, n_head, n_embd)
    return (
        n_embd * n_embd * 2  # Wq + Wo
        + n_embd * kv_dim  # Wk + Wv
        + n_embd  # rms norm
    )


def calculate_mamba_memory(
    mamba_config: MambaConfig,
    n_embd: int,
    batch_size: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
) -> int:
    return (
        calculate_mamba_params(mamba_config, n_embd) * sizeof_dtype(weights_dtype)
        + calculate_mamba_state_size(mamba_config, batch_size) * sizeof_dtype(kv_cache_dtype)
    ) / 2**20


def calculate_mamba_params(
    mamba_config: MambaConfig,
    n_embd: int,
) -> int:
    d_inner, in_proj_dim, conv_dim, kernel_size = _calculate_mamba_intermediates(mamba_config)
    param_shapes = {
        "A_log": (mamba_config.num_heads,),
        "D": (mamba_config.num_heads,),
        "conv1d.bias": (conv_dim,),
        "conv1d.weight": (conv_dim, 1, kernel_size),
        "dt_bias": (mamba_config.num_heads,),
        "in_proj.weight": (in_proj_dim, n_embd),
        "norm.weight": (d_inner,),
        "out_proj.weight": (n_embd, d_inner),
    }
    mamba_mixer_params = sum([math.prod(shape) for shape in param_shapes.values()])
    rms_norm_params = n_embd
    return mamba_mixer_params + rms_norm_params


def calculate_mamba_state_size(
    mamba_config: MambaConfig,
    batch_size: int,
) -> int:
    d_inner, in_proj_dim, conv_dim, kernel_size = _calculate_mamba_intermediates(mamba_config)
    conv_state_size = math.prod((batch_size, conv_dim, kernel_size))
    ssm_state_size = math.prod(
        (batch_size, mamba_config.num_heads, mamba_config.head_dim, mamba_config.state_dim)
    )
    return conv_state_size + ssm_state_size


def _calculate_mamba_intermediates(mamba_config: MambaConfig) -> tuple[int, ...]:
    d_inner = mamba_config.num_heads * mamba_config.head_dim
    in_proj_dim = (
        d_inner * 2 + 2 * mamba_config.num_groups * mamba_config.state_dim + mamba_config.num_heads
    )
    conv_dim = d_inner + 2 * mamba_config.num_groups * mamba_config.state_dim
    kernel_size = 4
    return d_inner, in_proj_dim, conv_dim, kernel_size


def calculate_linear_memory(
    n_embd: int,
    weights_dtype: torch.dtype,
) -> float:
    return calculate_linear_params(n_embd) * sizeof_dtype(weights_dtype) / 2**20


def calculate_linear_params(
    n_embd: int,
) -> int:
    return n_embd**2 + n_embd


def calculate_ffn_memory(
    ffn_config: FFNConfig,
    n_embd: int,
    weights_dtype: torch.dtype,
) -> float:
    num_params = calculate_ffn_params(ffn_config, n_embd)
    return num_params * sizeof_dtype(weights_dtype) / 2**20


def calculate_ffn_params(
    ffn_config: FFNConfig,
    n_embd: int,
) -> float:
    if ffn_config.is_moe:
        return calculate_moe_params(ffn_config, n_embd)
    else:
        return calculate_dense_ffn_params(ffn_config, n_embd)


def calculate_dense_ffn_params(
    ffn_config: FFNConfig,
    n_embd: int,
) -> int:
    intermediate_size = ffn_config.intermediate_size
    num_linear_layers = 3 if getattr(ffn_config, "gated", True) else 2
    rms_norm_params = n_embd
    return n_embd * intermediate_size * num_linear_layers + rms_norm_params


def calculate_moe_params(
    ffn_config: FFNConfig,
    n_embd: int,
) -> int:
    with torch.device("meta"):
        config = DeciLMConfig(hidden_size=n_embd)
        moe = DeciLMMoe(config, ffn_config)
    moe_params = sum(p.numel() for p in moe.parameters())
    layernorm_params = n_embd
    return moe_params + layernorm_params


def calculate_non_block_memory(
    n_embd: int,
    vocab_size: int,
    weight_dtype: torch.dtype,
) -> float:
    return calculate_non_block_params(n_embd, vocab_size) * sizeof_dtype(weight_dtype) / 2**20


def calculate_non_block_params(
    n_embd: int,
    vocab_size: int,
) -> int:
    return vocab_size * n_embd * 2 + n_embd
