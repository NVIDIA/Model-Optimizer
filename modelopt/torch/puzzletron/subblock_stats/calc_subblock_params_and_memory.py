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
from dataclasses import replace

import torch
from transformers import PretrainedConfig

from ..anymodel.model_descriptor import ModelDescriptor
from ..block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MLAConfig,
    MoEConfig,
    SubblockConfig,
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
    "calculate_additive_metrics",
    "calculate_ffn_memory",
    "calculate_mamba_memory",
    "calculate_mamba_state_size",
    "calculate_non_block_memory",
    "calculate_non_block_params",
    "calculate_subblock_memory",
    "calculate_subblock_params",
]

_ATTENTION_LIKE_KINDS = frozenset(("attention", "mla", "mamba"))
_FFN_LIKE_KINDS = frozenset(("ffn", "moe"))


def _causal_attention_pairs(num_tokens: int, window: int | None) -> int:
    if num_tokens <= 0:
        return 0
    if window is None:
        return num_tokens * (num_tokens + 1) // 2
    return sum(min(position, window) for position in range(1, num_tokens + 1))


def calculate_additive_metrics(
    subblock_config: SubblockConfig,
    *,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    n_embd: int,
    n_head: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    num_params: int | None = None,
    active_params: int | None = None,
) -> dict[str, float | int | dict[str, str]]:
    """Return deterministic additive bytes and phase FLOPs for one subblock."""
    if num_params is None:
        num_params = calculate_subblock_params(model_config, subblock_config, descriptor)
    if active_params is None:
        active_params = calc_subblock_active_params(
            subblock_config, model_config, descriptor, n_embd, num_params=num_params
        )
    if subblock_config.no_op:
        num_params = active_params = 0

    weight_memory_mib = num_params * sizeof_dtype(weights_dtype) / 2**20
    kv_cache_bytes_per_token = 0
    state_cache_bytes_per_sequence = 0
    prefill_tokens = prefill_seq_len + (1 if generation_seq_len > 0 else 0)
    decode_tokens = max(0, generation_seq_len - 1)
    prefill_flops = 2 * active_params * batch_size * prefill_tokens
    decode_flops = 2 * active_params * batch_size * decode_tokens

    if isinstance(subblock_config, AttentionConfig) and not subblock_config.no_op:
        kv_dim = calculate_kv_dim(subblock_config.num_kv_heads, n_head, n_embd)
        kv_cache_bytes_per_token = kv_dim * sizeof_dtype(kv_cache_dtype)
        query_heads = int(subblock_config.num_query_heads or n_head)
        head_dim = int(subblock_config.qk_head_dim or (n_embd // n_head))
        window = (
            int(subblock_config.sliding_window_size)
            if isinstance(subblock_config.sliding_window_size, int)
            else None
        )
        prefill_pairs = _causal_attention_pairs(prefill_seq_len, window)
        if generation_seq_len > 0:
            prefill_pairs += min(prefill_seq_len + 1, window or prefill_seq_len + 1)
        decode_pairs = sum(
            min(prefill_seq_len + output_index, window or prefill_seq_len + output_index)
            for output_index in range(2, generation_seq_len + 1)
        )
        prefill_flops += 4 * batch_size * query_heads * head_dim * prefill_pairs
        decode_flops += 4 * batch_size * query_heads * head_dim * decode_pairs
    elif isinstance(subblock_config, MLAConfig) and not subblock_config.no_op:
        lm_config = descriptor.get_language_model_config(model_config)
        rope_dim = int(getattr(lm_config, "qk_rope_head_dim", 0) or 0)
        kv_cache_bytes_per_token = (
            int(subblock_config.kv_lora_rank or 0) + rope_dim
        ) * sizeof_dtype(kv_cache_dtype)
        query_heads = int(subblock_config.num_heads or n_head)
        head_dim = max(1, n_embd // max(1, n_head))
        prefill_pairs = _causal_attention_pairs(prefill_seq_len, None)
        if generation_seq_len > 0:
            prefill_pairs += prefill_seq_len + 1
        decode_pairs = sum(
            prefill_seq_len + output_index
            for output_index in range(2, generation_seq_len + 1)
        )
        prefill_flops += 4 * batch_size * query_heads * head_dim * prefill_pairs
        decode_flops += 4 * batch_size * query_heads * head_dim * decode_pairs
    elif isinstance(subblock_config, MambaConfig) and not subblock_config.no_op:
        state_cache_bytes_per_sequence = calculate_mamba_state_size(
            subblock_config, 1
        ) * sizeof_dtype(kv_cache_dtype)
        _, _, conv_dim, kernel_size = _calculate_mamba_intermediates(subblock_config)
        scan_flops_per_token = 2 * (
            subblock_config.num_heads
            * subblock_config.head_dim
            * subblock_config.state_dim
            + conv_dim * kernel_size
        )
        prefill_flops += scan_flops_per_token * batch_size * prefill_tokens
        decode_flops += scan_flops_per_token * batch_size * decode_tokens

    return {
        "weight_memory_mib": weight_memory_mib,
        "kv_cache_bytes_per_token": kv_cache_bytes_per_token,
        "state_cache_bytes_per_sequence": state_cache_bytes_per_sequence,
        "prefill_flops": int(prefill_flops),
        "decode_flops": int(decode_flops),
        "additive_metric_provenance": {
            "weight_memory_mib": "exact_parameter_formula",
            "kv_cache_bytes_per_token": "typed_formula",
            "state_cache_bytes_per_sequence": "typed_formula",
            "prefill_flops": "typed_formula",
            "decode_flops": "typed_formula",
        },
    }


def _language_model_attr(config: PretrainedConfig, descriptor: type[ModelDescriptor], name: str, default=None):
    return getattr(descriptor.get_language_model_config(config), name, getattr(config, name, default))


def _configured_subblock(
    config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    kinds: frozenset[str],
) -> SubblockConfig | None:
    lm_config = descriptor.get_language_model_config(config)
    raw_block_configs = getattr(lm_config, "block_configs", None) or getattr(
        config, "block_configs", None
    )
    if not raw_block_configs:
        return None
    for block_config in maybe_cast_block_configs(raw_block_configs):
        for subblock in block_config.subblock_configs:
            if subblock.kind in kinds:
                return subblock
    return None


def _candidate_layer_module_names(
    descriptor: type[ModelDescriptor],
    index: int,
) -> tuple[str, ...]:
    """Return live-module aliases for a descriptor layer path.

    Some families intentionally use checkpoint-key paths in their descriptor
    because sorted-teacher surgery operates on safetensors keys.  Remote-code
    HF modules can expose the same logical layer under a different live module
    prefix (Nemotron3: checkpoint ``backbone.layers`` vs module ``model.layers``).
    Parameter/memory estimation works on a meta-initialized module tree, so try
    both forms here instead of forcing the descriptor to pick one globally.
    """
    primary = descriptor.layer_block_name(index=index)
    candidates = [primary]
    aliases = (
        ("backbone.layers.", "model.layers."),
        ("model.layers.", "backbone.layers."),
        ("model.language_model.layers.", "model.layers."),
        ("model.layers.", "model.language_model.layers."),
    )
    for src, dst in aliases:
        if src in primary:
            candidates.append(primary.replace(src, dst, 1))
    return tuple(dict.fromkeys(candidates))


def _get_decoder_layer_module(model: torch.nn.Module, descriptor: type[ModelDescriptor], index: int):
    errors = []
    for name in _candidate_layer_module_names(descriptor, index):
        try:
            return model.get_submodule(name)
        except AttributeError as exc:
            errors.append(f"{name}: {exc}")
    raise AttributeError(
        "Could not resolve decoder layer module for parameter counting. Tried: "
        + "; ".join(errors)
    )


def _checkpoint_tensor_count(module: torch.nn.Module) -> int:
    """Count parameters and persistent buffers serialized by ``state_dict``."""

    return sum(tensor.numel() for tensor in module.state_dict().values())


def _fallback_attention_config(
    config: PretrainedConfig, descriptor: type[ModelDescriptor], *, no_op: bool
) -> AttentionConfig:
    return AttentionConfig(
        no_op=no_op,
        num_kv_heads=_language_model_attr(config, descriptor, "num_key_value_heads"),
        num_query_heads=_language_model_attr(config, descriptor, "num_attention_heads"),
    )


def _fallback_ffn_config(
    config: PretrainedConfig, descriptor: type[ModelDescriptor], *, no_op: bool
) -> FFNConfig:
    return FFNConfig(
        no_op=no_op,
        intermediate_size=_language_model_attr(config, descriptor, "intermediate_size"),
    )


def _single_subblock_to_block_config(
    config: PretrainedConfig,
    subblock_config: SubblockConfig,
    descriptor: type[ModelDescriptor],
) -> BlockConfig:
    """Complete a single measured subblock with a no-op companion side.

    Some descriptors require a structurally complete block to derive layer
    overrides even when we only want the params for attention or FFN.  Use the
    model's typed block configs for the missing side when possible so MoE/Mamba
    families keep the right layer type, and fall back to dense defaults.
    """
    if subblock_config.kind in _ATTENTION_LIKE_KINDS:
        companion = _configured_subblock(config, descriptor, _FFN_LIKE_KINDS) or _fallback_ffn_config(
            config, descriptor, no_op=False
        )
        return BlockConfig(subblock_configs=(subblock_config, replace(companion, no_op=True)))

    if subblock_config.kind in _FFN_LIKE_KINDS:
        companion = _configured_subblock(
            config, descriptor, _ATTENTION_LIKE_KINDS
        ) or _fallback_attention_config(config, descriptor, no_op=False)
        return BlockConfig(subblock_configs=(replace(companion, no_op=True), subblock_config))

    return BlockConfig(subblock_configs=(subblock_config,))


def calculate_subblock_memory(
    subblock_config: SubblockConfig,
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    n_embd: int,
    n_head: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    num_params: int | None = None,
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
    if isinstance(subblock_config, FFNConfig | MoEConfig):
        return calculate_ffn_memory(
            subblock_config,
            model_config,
            descriptor,
            weights_dtype,
            num_params=num_params,
        )
    if isinstance(subblock_config, AttentionConfig):
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
            num_params=num_params,
        )
    if isinstance(subblock_config, MLAConfig):
        return calculate_mla_memory(
            subblock_config,
            model_config,
            descriptor,
            batch_size,
            prefill_seq_len,
            generation_seq_len,
            weights_dtype,
            kv_cache_dtype,
            num_params=num_params,
        )
    if isinstance(subblock_config, MambaConfig):
        return calculate_mamba_memory(
            subblock_config,
            model_config,
            descriptor,
            batch_size,
            weights_dtype,
            kv_cache_dtype,
            num_params=num_params,
        )
    raise_unknown_subblock_config_error(subblock_config)


def calculate_subblock_params(
    config: PretrainedConfig,
    layer_config: BlockConfig | SubblockConfig,
    descriptor: type[ModelDescriptor],
) -> int:
    """Count parameters on one meta decoder layer."""
    if isinstance(layer_config, SubblockConfig):
        block_config = _single_subblock_to_block_config(config, layer_config, descriptor)
    else:
        block_config = layer_config

    active_attention_like = [
        subblock
        for subblock in block_config.subblock_configs
        if subblock.kind in _ATTENTION_LIKE_KINDS and not subblock.no_op
    ]
    active_ffn_like = [
        subblock
        for subblock in block_config.subblock_configs
        if subblock.kind in _FFN_LIKE_KINDS and not subblock.no_op
    ]
    if active_attention_like and active_ffn_like:
        raise AssertionError(
            "One of the attention-like or FFN-like subblocks must be no-op for sublayer param calculation "
            "(single subblock at a time)."
        )
    if not active_attention_like and not active_ffn_like:
        return 0

    _config = copy.deepcopy(config)
    block_configs = maybe_cast_block_configs([block_config])
    descriptor.set_block_configs(_config, block_configs)
    lm_config = descriptor.get_language_model_config(_config)
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

    decoder_layer = _get_decoder_layer_module(model, descriptor, index=0)
    if not active_attention_like:
        descriptor.attn_no_op_post_init(decoder_layer)
    if not active_ffn_like:
        descriptor.mlp_no_op_post_init(decoder_layer)
    return _checkpoint_tensor_count(decoder_layer)


def calc_subblock_active_params(
    sublayer_config: SubblockConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    n_embd: int,
    num_params: int | None = None,
) -> int:
    """Calculate the number of "active" parameters for a subblock (FFN, Attention, or MoE).

    For non-MoE subblocks, simply calls `calculate_subblock_params` to count all parameters.
    For MoE (Mixture-of-Experts) FFN subblocks, the active parameter count is deterministic:
    the router selects a fixed ``top_k`` experts per token, so it is the router
    plus the always-on shared expert plus ``top_k`` routed experts.

    Args:
        sublayer_config: The subblock configuration (either FFNConfig or AttentionConfig).
        model_config: The Hugging Face model configuration.
        descriptor: The ModelDescriptor class corresponding to this model family.
        n_embd: The embedding size (hidden dimension).

    Returns:
        The number of "active" parameters for the given subblock.
    """
    if sublayer_config.no_op:
        return 0
    if isinstance(sublayer_config, MoEConfig):
        moe_config = replace(
            sublayer_config,
            num_experts=sublayer_config.num_experts
            or _language_model_attr(model_config, descriptor, "n_routed_experts"),
            top_k=sublayer_config.top_k
            or _language_model_attr(model_config, descriptor, "num_experts_per_tok"),
            expert_intermediate_size=sublayer_config.expert_intermediate_size
            or _language_model_attr(model_config, descriptor, "moe_intermediate_size"),
            shared_expert_intermediate_size=sublayer_config.shared_expert_intermediate_size
            or _language_model_attr(
                model_config, descriptor, "moe_shared_expert_intermediate_size"
            ),
            latent_dim=sublayer_config.latent_dim
            or _language_model_attr(model_config, descriptor, "moe_latent_size"),
        )
        return estimate_moe_active_params(moe_config, n_embd)
    if num_params is not None:
        return num_params
    return calculate_subblock_params(model_config, sublayer_config, descriptor)


def estimate_moe_active_params(subblock_config: MoEConfig, n_embd: int) -> int:
    """Compute the number of active parameters for a Mixture-of-Experts (MoE) FFN subblock.

    Active experts per token are fixed by the router's ``top_k``, so the
    active parameter count is deterministic: the router, the always-on shared expert, and
    ``top_k`` routed experts.

    Args:
        subblock_config: The MoE subblock configuration.
        n_embd: The embedding dimension (input and output size per expert).

    Returns:
        Number of parameters actively used per token.
    """
    required = {
        "num_experts": subblock_config.num_experts,
        "top_k": subblock_config.top_k,
        "expert_intermediate_size": subblock_config.expert_intermediate_size,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Cannot estimate MoE active params without {missing}")

    num_experts = subblock_config.num_experts
    num_active_experts = subblock_config.top_k
    expert_dim = subblock_config.expert_intermediate_size
    shared_expert_dim = subblock_config.shared_expert_intermediate_size or 0
    router_num_params = n_embd * num_experts
    if subblock_config.latent_dim is not None:
        # Nemotron latent MoE has shared hidden<->latent projections and two
        # matrices per ReLU^2 routed/shared expert (no separate gate projection).
        latent = subblock_config.latent_dim
        latent_projection_params = 2 * n_embd * latent
        active_expert_num_params = 2 * expert_dim * latent * num_active_experts
        shared_expert_num_params = 2 * shared_expert_dim * n_embd
        return (
            router_num_params
            + latent_projection_params
            + active_expert_num_params
            + shared_expert_num_params
        )

    num_linear_layers = 3  # gated up/gate/down experts
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
    num_params: int | None = None,
) -> dict[str, float]:
    """Estimate attention subblock memory (KV cache + weights) in MiB."""
    seq_len = prefill_seq_len + generation_seq_len
    if (
        attention_config.llama4 is not None
        and (attention_chunk_size := attention_config.llama4.attention_chunk_size) is not None
    ):
        seq_len = min(seq_len, attention_chunk_size)
    sliding_window = attention_config.sliding_window_size
    if isinstance(sliding_window, int):
        seq_len = min(seq_len, sliding_window)

    kv_dim = calculate_kv_dim(attention_config.num_kv_heads, n_head, n_embd)
    total_num_tokens = seq_len * batch_size
    kv_cache_size = total_num_tokens * kv_dim
    if num_params is None:
        num_params = calculate_subblock_params(model_config, attention_config, descriptor)
    total_memory = (
        kv_cache_size * sizeof_dtype(kv_cache_dtype) + num_params * sizeof_dtype(weights_dtype)
    ) / 2**20
    kv_cache_memory = kv_cache_size * sizeof_dtype(kv_cache_dtype) / 2**20
    return {"memory_mib": total_memory, "kv_cache_memory_mib": kv_cache_memory}


def calculate_mla_memory(
    mla_config: MLAConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    num_params: int | None = None,
) -> dict[str, float]:
    """Estimate MLA weights and compressed latent/rope KV cache memory."""

    lm_config = descriptor.get_language_model_config(model_config)
    rope_dim = int(getattr(lm_config, "qk_rope_head_dim", 0) or 0)
    cached_width = int(mla_config.kv_lora_rank or 0) + rope_dim
    total_tokens = (prefill_seq_len + generation_seq_len) * batch_size
    cache_elements = total_tokens * cached_width
    if num_params is None:
        num_params = calculate_subblock_params(model_config, mla_config, descriptor)
    kv_cache_memory = cache_elements * sizeof_dtype(kv_cache_dtype) / 2**20
    return {
        "memory_mib": kv_cache_memory + num_params * sizeof_dtype(weights_dtype) / 2**20,
        "kv_cache_memory_mib": kv_cache_memory,
    }


def calculate_mamba_memory(
    mamba_config: MambaConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    batch_size: int,
    weights_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    num_params: int | None = None,
) -> int:
    """Calculate memory usage (MiB) for a Mamba subblock.

    Args:
        mamba_config: Mamba configuration.
        model_config: Model configuration.
        descriptor: Model descriptor class.
        batch_size: Batch size for memory estimate.
        weights_dtype: Data type for model weights.
        kv_cache_dtype: Data type for state/kv-cache.

    Returns:
        Estimated memory usage in mebibytes (MiB) for the Mamba subblock.
    """
    if num_params is None:
        num_params = calculate_subblock_params(model_config, mamba_config, descriptor)
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
    ffn_config: FFNConfig | MoEConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    weights_dtype: torch.dtype | str,
    experts_dtype: torch.dtype | str | None = None,
    num_params: int | None = None,
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
    if num_params is None:
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
