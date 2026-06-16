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
import math

import torch
from transformers import PretrainedConfig

from ..anymodel.model_descriptor import ModelDescriptor
from ..block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    maybe_cast_block_configs,
)
from ..tools.checkpoint_utils_hf import init_model_from_config
from ..utils.misc import (
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
]


def calculate_subblock_memory(
    subblock_config: FFNConfig | AttentionConfig,
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    n_embd: int,
    n_head: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
) -> float | dict[str, float]:
    """Calculate the memory usage of a single subblock (FFN or Attention).

    Given its configuration and runtime dimensions, returns bytes or a detailed dict.

    Args:
        subblock_config: Subblock configuration dataclass.
        batch_size: Batch size for memory estimate.
        prefill_seq_len: Sequence length for prefill phase.
        generation_seq_len: Sequence length for generation phase (token-by-token).
        n_embd: Embedding (hidden) dimension.
        n_head: Number of attention heads (used for non-FFN).
        weights_dtype: PyTorch dtype for model weights.
        kv_cache_dtype: PyTorch dtype for KV cache.
        model_config: HuggingFace-style config instance describing the model.
        descriptor: Model descriptor type (for puzzletron model types).

    Returns:
        Memory usage in bytes (float), or a dictionary by memory type.
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
                n_embd,
                n_head,
                weights_dtype,
                kv_cache_dtype,
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
) -> int:
    """Calculate the number of "active" parameters for a subblock (FFN, Attention, or MoE).

    For non-MoE subblocks, simply calls `calculate_subblock_params` to count all parameters.
    For MoE (Mixture-of-Experts) FFN subblocks, the active parameter count is deterministic:
    the router selects a fixed ``num_experts_per_tok`` experts per token, so it is the router
    plus the always-on shared expert plus ``num_experts_per_tok`` routed experts.

    Args:
        sublayer_config: The subblock configuration (either FFNConfig or AttentionConfig).
        model_config: The Hugging Face model configuration.
        descriptor: The ModelDescriptor class corresponding to this model family.
        n_embd: The embedding size (hidden dimension).

    Returns:
        The number of "active" parameters for the given subblock.
    """
    if not (isinstance(sublayer_config, FFNConfig) and sublayer_config.is_moe):
        return calculate_subblock_params(model_config, sublayer_config, descriptor)
    return estimate_moe_active_params(sublayer_config, n_embd)


def estimate_moe_active_params(subblock_config: FFNConfig, n_embd: int) -> int:
    """Compute the number of active parameters for a Mixture-of-Experts (MoE) FFN subblock.

    Active experts per token are fixed by the router's top-k (``num_experts_per_tok``), so the
    active parameter count is deterministic: the router, the always-on shared expert, and
    ``num_experts_per_tok`` routed experts.

    Args:
        subblock_config: The FFNConfig for the MoE subblock (with .moe field configured).
        n_embd: The embedding dimension (input and output size per expert).

    Returns:
        Number of parameters actively used per token.
    """
    num_experts = subblock_config.moe.num_local_experts
    num_active_experts = subblock_config.moe.num_experts_per_tok
    expert_dim = subblock_config.moe.expert_intermediate_dim
    shared_expert_dim = subblock_config.moe.shared_expert_intermediate_dim
    num_linear_layers = 3  # all moe experts have 3 linear layers

    router_num_params = n_embd * num_experts
    active_expert_num_params = num_linear_layers * expert_dim * n_embd * num_active_experts
    shared_expert_num_params = num_linear_layers * shared_expert_dim * n_embd
    return router_num_params + active_expert_num_params + shared_expert_num_params


def calculate_attention_memory(
    attention_config: AttentionConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    n_embd: int,
    n_head: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
) -> dict[str, float]:
    """Estimate attention subblock memory (KV cache + weights) in MiB."""
    seq_len = prefill_seq_len + generation_seq_len
    if (
        attention_config.is_llama4
        and (attention_chunk_size := attention_config.llama4.attention_chunk_size) is not None
    ):
        seq_len = min(seq_len, attention_chunk_size)

    kv_dim = calculate_kv_dim(attention_config.num_key_value_heads, n_head, n_embd)
    total_num_tokens = seq_len * batch_size
    kv_cache_size = total_num_tokens * kv_dim
    num_params = calculate_subblock_params(model_config, attention_config, descriptor)
    total_memory = (
        kv_cache_size * sizeof_dtype(kv_cache_dtype) + num_params * sizeof_dtype(weights_dtype)
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
        attention_config: Mamba attention configuration, including Mamba-specific settings.
        model_config: Model configuration.
        descriptor: Model descriptor class.
        batch_size: Batch size for memory estimate.
        weights_dtype: Data type for model weights.
        kv_cache_dtype: Data type for state/kv-cache.

    Returns:
        Estimated memory usage in mebibytes (MiB) for the Mamba subblock.
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
        mamba_config: Configuration object containing Mamba subblock parameters.
        batch_size: Batch size to estimate the memory/state requirements for.

    Returns:
        Total state size (number of elements) required for the Mamba subblock, including convolution and SSM state.
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
    kernel_size = mamba_config.conv_kernel_size
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
        ffn_config: FFN configuration for the block.
        model_config: The parent model configuration.
        descriptor: Model descriptor class.
        weights_dtype: Data type for FFN weights.
        experts_dtype: Data type for expert weights (for MoE layers, if present).

    Returns:
        Estimated FFN memory usage in mebibytes (MiB).
    """
    # TODO: How to separate between expert weights and the rest for any model (same as puzzletron).
    num_params = calculate_subblock_params(model_config, ffn_config, descriptor)
    return num_params * sizeof_dtype(weights_dtype) / 2**20


def calculate_non_block_memory(
    n_embd: int,
    vocab_size: int,
    weight_dtype: torch.dtype,
) -> float:
    """Estimate the memory usage in MiB of non-subblock components (e.g., embeddings, output projection)."""
    return calculate_non_block_params(n_embd, vocab_size) * sizeof_dtype(weight_dtype) / 2**20


def calculate_non_block_params(
    n_embd: int,
    vocab_size: int,
) -> int:
    """Calculate the number of parameters for non-subblock components (e.g., embeddings, output projection)."""
    return vocab_size * n_embd * 2 + n_embd
