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

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import PretrainedConfig

from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    maybe_cast_block_configs,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import init_model_from_config
from modelopt.torch.puzzletron.utils.misc import (
    EmptyInitOnDevice,
    calculate_kv_dim,
    raise_unknown_subblock_config_error,
    sizeof_dtype,
)

__all__ = [
    "calc_subblock_active_params",
    "calculate_ffn_memory",
    "calculate_mamba_memory",
    "calculate_mamba_state_size",
    "calculate_non_block_memory",
    "calculate_non_block_params",
    "calculate_subblock_memory",
    "calculate_subblock_params",
    "estimate_num_active_experts",
    "load_moe_stats",
]


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
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
) -> float | dict[str, float]:
    """Calculate the memory usage of a single subblock (FFN or Attention).

    Given its configuration and runtime dimensions, returns bytes or a detailed dict.

    Args:
        subblock_config (FFNConfig | AttentionConfig): Subblock configuration dataclass.
        batch_size (int): Batch size for memory estimate.
        prefill_seq_len (int): Sequence length for prefill phase.
        generation_seq_len (int): Sequence length for generation phase (token-by-token).
        prefill_queue_size (int): Token queue size for prefill attention memory allocation.
        n_embd (int): Embedding (hidden) dimension.
        n_head (int): Number of attention heads (used for non-FFN).
        weights_dtype (torch.dtype): PyTorch dtype for model weights.
        kv_cache_dtype (torch.dtype): PyTorch dtype for KV cache.
        allocate_prefill_query (bool): Whether to allocate query cache for prefill tokens.
        model_config (PretrainedConfig): HuggingFace-style config instance describing the model.
        descriptor (type[ModelDescriptor]): Model descriptor type (for puzzletron model types).

    Returns:
        float | dict[str, float]: Memory usage in bytes (float), or a dictionary by memory type.
    """
    if subblock_config.no_op:
        return 0
    if isinstance(subblock_config, FFNConfig):
        return calculate_ffn_memory(
            subblock_config,
            model_config,
            descriptor,
            weights_dtype,
        )
    if isinstance(subblock_config, AttentionConfig):
        if subblock_config.is_mamba:
            return calculate_mamba_memory(
                subblock_config,
                model_config,
                descriptor,
                batch_size,
                weights_dtype,
                kv_cache_dtype,
            )
        else:
            return calculate_attention_memory(
                subblock_config,
                model_config,
                descriptor,
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
    config: PretrainedConfig,
    layer_config: BlockConfig | FFNConfig | AttentionConfig,
    descriptor: type[ModelDescriptor],
) -> int:
    """Count parameters on one meta decoder layer.

    The caller is responsible for adjusting per-layer config fields (e.g.
    ``hybrid_override_pattern``) before passing ``config``; see
    ``ModelDescriptor.truncate_pattern_for_subblock``.
    """
    if isinstance(layer_config, (FFNConfig, AttentionConfig)):
        block_config = layer_config.to_blockconfig()
    else:
        block_config = layer_config

    ffn = block_config.ffn
    attn = block_config.attention
    ffn_no_op = ffn is None or ffn.no_op
    attn_no_op = attn is None or attn.no_op
    if not (ffn_no_op or attn_no_op):
        raise AssertionError(
            "One of ffn or attention must be no-op for sublayer param calculation "
            "(single subblock at a time)."
        )
    if ffn_no_op and attn_no_op:
        return 0

    _config = copy.deepcopy(config)
    lm_config = descriptor.get_language_model_config(_config)
    lm_config.num_hidden_layers = 1

    block_configs = maybe_cast_block_configs([block_config])
    _config.block_configs = block_configs
    if lm_config is not _config:
        lm_config.block_configs = block_configs

    # Replaced earlier pattern:
    #   with EmptyInitOnDevice("meta"), deci_x_patcher(..., block_configs=block_configs):
    #       model = init_model_from_config(_config, ...)
    #
    # That fails on GPT-OSS with recent Transformers: ``deci_x_patcher`` runs
    # ``attn_no_op_post_init`` / ``mlp_no_op_post_init`` inside ``DecoderLayer.__init__``, so norms
    # / attn / mlp are swapped for placeholders before ``GptOssModel.__init__`` finishes. At the end
    # of ``GptOssModel.__init__`` the stack calls ``self.post_init()`` — inherited from
    # ``PreTrainedModel`` — which then raises
    # ``ValueError`` (e.g. ``post_attention_layernorm`` in ``_keep_in_fp32_modules`` no longer matches
    # the tree). Below we merge per-layer fields manually, init without the patcher, then call the
    # same descriptor no-op hooks on the built layer (equivalent param count for
    # ``num_hidden_layers == 1``).

    # ``block_config_to_layer_overrides`` may include keys with value ``None``; we omit those so
    # ``lm_config.update`` does not overwrite existing fields with ``None`` (same rule as
    # ``override_config_with_block_configs`` inside ``deci_x_patcher``).
    layer_overrides = descriptor.block_config_to_layer_overrides(block_configs[0])
    lm_config.update({k: v for k, v in layer_overrides.items() if v is not None})

    with EmptyInitOnDevice("meta"):
        model = init_model_from_config(
            _config,
            trust_remote_code=descriptor.requires_trust_remote_code(),
        )

    decoder_layer = model.get_submodule(descriptor.layer_block_name(index=0))
    if attn_no_op:
        descriptor.attn_no_op_post_init(decoder_layer)
    if ffn_no_op:
        descriptor.mlp_no_op_post_init(decoder_layer)
    return sum(p.numel() for p in decoder_layer.parameters())


def calc_subblock_active_params(
    sublayer_config: FFNConfig | AttentionConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    n_embd: int,
    moe_stats_file: str,
    batch_size: int,
    block_idx: int,
) -> int:
    """Calculate the number of "active" parameters for a subblock (FFN, Attention, or MoE).

    For non-MoE subblocks, simply calls `calculate_subblock_params` to count all parameters.
    For MoE (Mixture-of-Experts) FFN subblocks, estimates the expected number of active parameters
    per batch by leveraging expert activation statistics (from a given stats file) and calculating
    the expected number of active experts, then multiplies by the number of parameters per expert.

    Args:
        sublayer_config: The subblock configuration (either FFNConfig or AttentionConfig).
        model_config: The Hugging Face model configuration.
        descriptor: The ModelDescriptor class corresponding to this model family.
        n_embd: The embedding size (hidden dimension).
        moe_stats_file: Path to file containing expert activation probabilities.
        batch_size: The batch size used for the estimate.
        block_idx: The index of the block/subblock within the network, used to index into the stats.

    Returns:
        int: The expected number of "active" parameters for the given subblock.
    """
    if not (isinstance(sublayer_config, FFNConfig) and sublayer_config.is_moe):
        return calculate_subblock_params(model_config, sublayer_config, descriptor)
    return estimate_moe_active_params(
        sublayer_config, n_embd, moe_stats_file, batch_size, block_idx
    )


def load_moe_stats(stats_file: str) -> dict:
    """Load MoE (Mixture-of-Experts) routing statistics from a file.

    This function reads a JSON file containing expert activation probabilities or counts for each MoE block.
    It returns the normalized probability distributions over experts for each block, as a list of numpy arrays.

    Args:
        stats_file (str): Path to the JSON file containing expert routing statistics for each block.

    Returns:
        list[np.ndarray]: A list where each element is a numpy array containing the normalized probability
            distribution over experts for the corresponding block. If a block's expert list is empty,
            its entry is 0.
    """
    with open(stats_file) as f:
        stats = json.load(f)
    return [
        np.array(expert_probs) / np.sum(expert_probs) if len(expert_probs) > 0 else 0
        for expert_probs in stats
    ]


def estimate_num_active_experts(
    dist_over_experts: np.ndarray, batch_size: int, num_experts: int
) -> int:
    """Estimate the expected number of active experts in a Mixture-of-Experts (MoE) layer.

    This function computes the expected number of unique experts that are selected at least once when performing
    inference with a given batch size. It assumes, for each input in the batch, an expert is chosen with probability
    given by `dist_over_experts` (typically a vector of probabilities for each expert). For a batch of size B, the
    expected number of active (i.e., selected at least once) experts is computed.

    Args:
        dist_over_experts (np.ndarray): A 1D array of probabilities for each expert.
        batch_size (int): The number of samples in the batch.
        num_experts (int): The maximum number of experts to consider (fewer if `dist_over_experts` is shorter).

    Returns:
        int: The expected number of experts selected at least once across the batch.
    """
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
    """Estimate the expected number of active (used) parameters for a Mixture-of-Experts (MoE) FFN subblock.

    Args:
        subblock_config (FFNConfig): The FFNConfig for the MoE subblock (with .moe field configured).
        n_embd (int): The embedding dimension (input and output size per expert).
        moe_stats_file (Path | str): Path to the JSON file containing routing/selection probabilities for experts.
        batch_size (int): Batch size to simulate/extrapolate expected expert use.
        block_idx (int): The index of the block/layer whose expert routing statistics should be used.

    Returns:
        int: Estimated number of parameters actively used for the current batch and expert selection statistics.
    """
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
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
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

    kv_dim = calculate_kv_dim(attention_config.num_key_value_heads, n_head, n_embd)
    total_num_tokens = seq_len * (batch_size + prefill_queue_size)
    kv_cache_size = total_num_tokens * kv_dim
    query_prefill_size = seq_len * n_embd if allocate_prefill_query else 0
    num_params = calculate_subblock_params(model_config, attention_config, descriptor)
    total_memory = (
        kv_cache_size * sizeof_dtype(kv_cache_dtype)
        + query_prefill_size * sizeof_dtype(weights_dtype)
        + num_params * sizeof_dtype(weights_dtype)
    ) / 2**20
    kv_cache_memory = kv_cache_size * sizeof_dtype(kv_cache_dtype) / 2**20
    return {"memory_mib": total_memory, "kv_cache_memory_mib": kv_cache_memory}


def calculate_mamba_memory(
    attention_config: AttentionConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    batch_size: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
) -> int:
    """Calculate memory usage (MiB) for a Mamba attention subblock.

    Args:
        attention_config (AttentionConfig): Mamba attention configuration,
            including Mamba-specific settings.
        model_config (PretrainedConfig): Model configuration.
        descriptor (type[ModelDescriptor]): Model descriptor class.
        batch_size (int): Batch size for memory estimate.
        weights_dtype (torch.dtype): Data type for model weights.
        kv_cache_dtype (torch.dtype): Data type for state/kv-cache.

    Returns:
        int: Estimated memory usage in mebibytes (MiB) for the Mamba subblock.
    """
    assert attention_config.mamba is not None
    mamba_config = attention_config.mamba
    num_params = calculate_subblock_params(model_config, attention_config, descriptor)
    return (
        num_params * sizeof_dtype(weights_dtype)
        + calculate_mamba_state_size(mamba_config, batch_size) * sizeof_dtype(kv_cache_dtype)
    ) / 2**20


def calculate_mamba_state_size(
    mamba_config: MambaConfig,
    batch_size: int,
) -> int:
    """Calculate the total state size for a Mamba attention subblock.

    Args:
        mamba_config (MambaConfig): Configuration object containing Mamba subblock parameters.
        batch_size (int): Batch size to estimate the memory/state requirements for.

    Returns:
        int: Total state size (number of elements) required for the Mamba subblock, including convolution and SSM state.
    """
    _, _, conv_dim, kernel_size = _calculate_mamba_intermediates(mamba_config)
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


def calculate_ffn_memory(
    ffn_config: FFNConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    weights_dtype: torch.dtype | str,
    experts_dtype: torch.dtype | str | None = None,
) -> float:
    """Estimate the memory usage in MiB of a feed-forward network (FFN) subblock.

    Args:
        ffn_config (FFNConfig): FFN configuration for the block.
        model_config (PretrainedConfig): The parent model configuration.
        descriptor (type[ModelDescriptor]): Model descriptor class.
        weights_dtype (torch.dtype | str): Data type for FFN weights.
        experts_dtype (torch.dtype | str | None, optional): Data type for expert weights
            (for MoE layers, if present). Defaults to None.

    Returns:
        float: Estimated FFN memory usage in mebibytes (MiB).
    """
    # TODO: How to separate between expert weights and the rest for any model (same as puzzletron).
    num_params = calculate_subblock_params(model_config, ffn_config, descriptor)
    return num_params * sizeof_dtype(weights_dtype) / 2**20


def calculate_non_block_memory(
    n_embd: int,
    vocab_size: int,
    weight_dtype: torch.dtype,
) -> float:
    """Estimate the memory usage in MiB of non-subblock components (e.g., embeddings, output projection).

    Args:
        n_embd (int): Embedding dimension (hidden size).
        vocab_size (int): Vocabulary size.
        weight_dtype (torch.dtype): Data type for model weights.

    Returns:
        float: Estimated non-subblock memory usage in mebibytes (MiB).
    """
    return calculate_non_block_params(n_embd, vocab_size) * sizeof_dtype(weight_dtype) / 2**20


def calculate_non_block_params(
    n_embd: int,
    vocab_size: int,
) -> int:
    """Calculate the number of parameters for non-subblock components (e.g., embeddings, output projection).

    Args:
        n_embd (int): Embedding dimension (hidden size).
        vocab_size (int): Vocabulary size.

    Returns:
        int: Estimated non-subblock parameter count.
    """
    return vocab_size * n_embd * 2 + n_embd
