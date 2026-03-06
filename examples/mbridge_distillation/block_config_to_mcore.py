# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map Model-Opt BlockConfig to Megatron-Core TransformerConfig overrides and no_op flags.

Used by mbridge_gpt_patcher and mbridge_mamba_patcher to inject per-layer config
when building heterogeneous models via Megatron Bridge. Supports MoE (per-layer
num_moe_experts, etc.) and Mamba (mamba_state_dim, mamba_head_dim, etc.).
replace_with_linear is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    BlockConfig,
    maybe_cast_block_configs,
)


@dataclass
class MCoreLayerOverrides:
    """Per-layer overrides and no_op flags for mcore TransformerConfig."""

    config_overrides: Dict[str, Any]
    """Dict of TransformerConfig field names to values (e.g. ffn_hidden_size, num_query_groups, num_moe_experts)."""

    attention_no_op: bool = False
    """If True, the patcher will replace the attention submodule with IdentityOp after init."""

    mlp_no_op: bool = False
    """If True, the patcher will replace the MLP submodule with IdentityOp after init."""


def block_config_to_mcore_overrides(
    block_config: Union[BlockConfig, dict],
    num_attention_heads: int,
    hidden_size: int,
) -> MCoreLayerOverrides:
    """Convert a single BlockConfig (Model-Opt / HF-style) to mcore overrides and no_op flags.

    replace_with_linear is not supported. Supports no_op, GQA, per-layer ffn_hidden_size,
    MoE (num_moe_experts, moe_router_topk, moe_ffn_hidden_size, moe_shared_expert_intermediate_size),
    and Mamba (mamba_state_dim, mamba_head_dim, mamba_num_groups, mamba_num_heads).

    Args:
        block_config: One block config (BlockConfig or dict with "attention" and "ffn" keys).
        num_attention_heads: Base model num_attention_heads (for GQA mapping).
        hidden_size: Base model hidden_size (for ffn_hidden_size when not overridden).

    Returns:
        MCoreLayerOverrides with config_overrides dict and attention_no_op / mlp_no_op flags.
    """
    if isinstance(block_config, dict):
        block_config = BlockConfig(**block_config)

    config_overrides: Dict[str, Any] = {}
    attention_no_op = False
    mlp_no_op = False

    # Attention (no replace_with_linear support)
    attn = block_config.attention
    if attn is not None:
        attention_no_op = getattr(attn, "no_op", False)
        mamba_cfg = getattr(attn, "mamba", None)
        if not attention_no_op:
            # Standard attention: GQA
            if mamba_cfg is None:
                num_kv_heads = getattr(attn, "num_key_value_heads", None)
                if num_kv_heads is not None and num_attention_heads >= num_kv_heads:
                    if num_attention_heads % num_kv_heads != 0:
                        raise ValueError(
                            f"num_attention_heads ({num_attention_heads}) must be divisible by "
                            f"num_key_value_heads ({num_kv_heads})"
                        )
                    config_overrides["num_query_groups"] = num_kv_heads
            # Mamba subblock: attention slot can carry MambaConfig for MambaLayer
            if mamba_cfg is not None:
                config_overrides["mamba_state_dim"] = getattr(mamba_cfg, "state_dim", None)
                config_overrides["mamba_head_dim"] = getattr(mamba_cfg, "head_dim", None)
                config_overrides["mamba_num_groups"] = getattr(mamba_cfg, "num_groups", None)
                config_overrides["mamba_num_heads"] = getattr(mamba_cfg, "num_heads", None)
        config_overrides = {k: v for k, v in config_overrides.items() if v is not None}

    # FFN / MLP / MoE (no replace_with_linear support)
    ffn = block_config.ffn
    if ffn is not None:
        mlp_no_op = getattr(ffn, "no_op", False)
        if not mlp_no_op:
            is_moe = getattr(ffn, "is_moe", False) or getattr(ffn, "moe", None) is not None
            if is_moe:
                moe_cfg = getattr(ffn, "moe", None)
                if moe_cfg is not None:
                    config_overrides["num_moe_experts"] = getattr(moe_cfg, "num_local_experts", None)
                    config_overrides["moe_router_topk"] = getattr(moe_cfg, "num_experts_per_tok", None)
                    config_overrides["moe_ffn_hidden_size"] = getattr(
                        moe_cfg, "expert_intermediate_dim", None
                    )
                    config_overrides["moe_shared_expert_intermediate_size"] = getattr(
                        moe_cfg, "shared_expert_intermediate_dim", None
                    )
                    config_overrides = {k: v for k, v in config_overrides.items() if v is not None}
            else:
                intermediate_size = getattr(ffn, "intermediate_size", None)
                if intermediate_size is not None:
                    config_overrides["ffn_hidden_size"] = intermediate_size

    return MCoreLayerOverrides(
        config_overrides=config_overrides,
        attention_no_op=attention_no_op,
        mlp_no_op=mlp_no_op,
    )


def get_overrides_for_layer(
    block_configs: Optional[List[Union[BlockConfig, dict]]],
    layer_number: int,
    num_attention_heads: int,
    hidden_size: int,
    *,
    strict_mamba_slot: bool = False,
) -> Optional[MCoreLayerOverrides]:
    """Get mcore overrides for a given 1-based layer number.

    For Mamba models, each slot is a single subblock (mamba, attention, mlp, or moe).
    Use strict_mamba_slot=True so that a block_config with both attention and ffn as op
    (neither no_op) raises ValueError.

    Args:
        block_configs: List of block configs (one per layer/slot), or None for homogeneous.
        layer_number: 1-based layer index (as passed by TransformerBlock / MambaStack).
        num_attention_heads: Base config num_attention_heads.
        hidden_size: Base config hidden_size.
        strict_mamba_slot: If True, raise ValueError when neither attention nor ffn is no_op
            (invalid for a Mamba slot; each slot must be one subblock type or both no_op).

    Returns:
        MCoreLayerOverrides for that layer, or None if block_configs is None/empty.
    """
    if not block_configs:
        return None
    block_configs = maybe_cast_block_configs(block_configs)
    layer_idx = layer_number - 1
    if layer_idx < 0 or layer_idx >= len(block_configs):
        return None
    overrides = block_config_to_mcore_overrides(
        block_configs[layer_idx],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
    )
    if strict_mamba_slot and not overrides.attention_no_op and not overrides.mlp_no_op:
        raise ValueError(
            "Mamba slot must have at least one of attention or ffn as no_op "
            "(each slot is a single subblock type: mamba, attention, mlp, or moe)."
        )
    return overrides
