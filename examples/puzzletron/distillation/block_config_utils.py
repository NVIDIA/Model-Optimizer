# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block config utilities for heterogeneous AnyModel/Puzzletron models.

Background
----------
An AnyModel (Puzzletron) checkpoint is *heterogeneous*: each decoder layer can have a
different architecture. This per-layer architecture is encoded as a list of BlockConfig
objects — one per decoder layer — stored in the model's HuggingFace config as the
``block_configs`` attribute.

A BlockConfig describes one layer's attention slot (standard attention or Mamba), its FFN
slot (dense MLP or MoE), and whether either slot is disabled (``no_op``).

This module provides two things:

1.  ``load_block_configs(hf_config, converter_name)``
    Loads block_configs from the HF config or generates them via the AnyModel converter.
    The HF config is the canonical, single source of truth — if the checkpoint was saved
    by AnyModel, config.json already contains ``block_configs``. Generating via the
    converter is only needed for models that have not yet been converted / saved.

2.  ``block_config_to_mcore_overrides(block_config, ...)`` / ``get_overrides_for_layer(...)``
    Translates a BlockConfig into ``MCoreLayerOverrides``, which is a plain dict of
    Megatron-Core ``TransformerConfig`` field names → new values for that layer, plus
    boolean flags indicating whether the attention or MLP submodule should be a no-op.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer override container
# ---------------------------------------------------------------------------


@dataclass
class MCoreLayerOverrides:
    """Per-layer TransformerConfig overrides and no-op flags for one decoder layer.

    Attributes:
        config_overrides: Dict mapping TransformerConfig field names to the per-layer value.
            Applied as a shallow copy of the global config before the layer is instantiated.
            Example: ``{"num_query_groups": 4, "num_moe_experts": 16}``.
        attention_no_op: If True the attention submodule should be replaced with a no-op.
        mlp_no_op: If True the MLP/MoE submodule should be replaced with a no-op.
    """

    config_overrides: Dict[str, Any]
    attention_no_op: bool = False
    mlp_no_op: bool = False


# ---------------------------------------------------------------------------
# Loading block_configs from an HF checkpoint
# ---------------------------------------------------------------------------


def load_block_configs(
    hf_config: Any,
    converter_name: str,
) -> Optional[List[dict]]:
    """Load per-layer block_configs for a model.

    Priority order:
        1. ``hf_config.block_configs`` — set by AnyModel when saving the checkpoint.
           This is the canonical source and is the *only* place to look in normal operation.
           It guarantees the block_configs exactly match the weights in the checkpoint.
        2. AnyModel ``ConverterFactory`` — generate default block_configs from the global
           model config. Used when first converting a model that has not yet been saved
           with AnyModel, or for homogeneous models where all layers are identical.

    Args:
        hf_config: A HuggingFace ``PretrainedConfig`` (e.g. from ``AutoConfig.from_pretrained``).
        converter_name: AnyModel converter key (e.g. ``"llama"``, ``"nemotron_h_v2"``).
            Used only if hf_config has no block_configs attribute.

    Returns:
        List of plain dicts (one per decoder layer), or None if block_configs cannot be
        obtained (e.g. ModelOpt not installed and hf_config has no block_configs).
    """
    # --- Primary: read from HF config ---
    raw = getattr(hf_config, "block_configs", None)
    if raw is not None:
        configs = _normalize_block_configs(raw)
        logger.info(
            "Loaded %d block_configs from hf_config.block_configs (model_type=%s)",
            len(configs),
            getattr(hf_config, "model_type", "unknown"),
        )
        return configs

    # --- Fallback: generate via AnyModel ConverterFactory ---
    logger.info(
        "hf_config has no block_configs attribute; generating via AnyModel converter '%s'",
        converter_name,
    )
    try:
        from modelopt.torch.puzzletron.anymodel import ConverterFactory
    except ImportError:
        logger.warning(
            "ModelOpt AnyModel is not installed. Cannot generate block_configs. "
            "The model will be built with the global (homogeneous) TransformerConfig."
        )
        return None

    converter_cls = ConverterFactory.get(converter_name)
    if converter_cls is None:
        logger.warning(
            "Unknown AnyModel converter '%s'. Cannot generate block_configs. "
            "Valid converters can be listed with ConverterFactory.list().",
            converter_name,
        )
        return None

    block_configs = converter_cls.create_block_configs_from_main_config(hf_config)
    configs = _normalize_block_configs(block_configs)
    logger.info(
        "Generated %d block_configs via converter '%s'", len(configs), converter_name
    )
    return configs


def _normalize_block_configs(raw) -> List[dict]:
    """Convert a sequence of BlockConfig objects / dicts to a list of plain dicts.

    Handles:
        - Already-plain dicts (no-op).
        - Objects with a ``to_dict()`` method (BlockConfig dataclass variant).
        - Standard Python dataclasses (via ``dataclasses.asdict``).
        - Arbitrary objects (via ``vars``).
    """
    result = []
    for bc in raw:
        if isinstance(bc, dict):
            result.append(bc)
        elif hasattr(bc, "to_dict"):
            result.append(bc.to_dict())
        elif dataclasses.is_dataclass(bc):
            result.append(dataclasses.asdict(bc))
        else:
            result.append(vars(bc))
    return result


# ---------------------------------------------------------------------------
# BlockConfig → MCoreLayerOverrides translation
# ---------------------------------------------------------------------------


def block_config_to_mcore_overrides(
    block_config: Union["BlockConfig", dict],
    num_attention_heads: int,
    hidden_size: int,
    *,
    moe_shared_expert_intermediate_omitted: bool = False,
) -> MCoreLayerOverrides:
    """Translate one BlockConfig into MCoreLayerOverrides for a single decoder layer.

    Supported mappings
    ------------------
    Attention slot (``block_config.attention``):
        - ``attention.no_op`` → ``attention_no_op = True``
        - ``attention.num_key_value_heads`` → ``num_query_groups`` (GQA)
        - ``attention.mamba.*`` → ``mamba_state_dim``, ``mamba_head_dim``,
          ``mamba_num_groups``, ``mamba_num_heads`` (Mamba2 slot inside attention field)

    FFN slot (``block_config.ffn``):
        - ``ffn.no_op`` → ``mlp_no_op = True``
        - ``ffn.intermediate_size`` → ``ffn_hidden_size`` (dense MLP)
        - ``ffn.moe.num_local_experts`` → ``num_moe_experts``
        - ``ffn.moe.num_experts_per_tok`` → ``moe_router_topk``
        - ``ffn.moe.expert_intermediate_dim`` → ``moe_ffn_hidden_size``
        - ``ffn.moe.shared_expert_intermediate_dim`` → ``moe_shared_expert_intermediate_size``
          (``None`` disables shared experts for that layer — required when the global Bridge
          config enables them but the architecture does not, e.g. GPT-OSS.)
        - If the key is **omitted** from a JSON ``block_configs`` entry, ``MoEConfig`` would
          otherwise default it to 8192 after casting. Callers pass
          ``moe_shared_expert_intermediate_omitted=True`` so Megatron clears shared experts.

    Not supported:
        - ``replace_with_linear`` (requires structural changes, not just config overrides).

    Args:
        block_config: One layer's BlockConfig object or dict.
        num_attention_heads: Global model ``num_attention_heads`` (used to validate GQA).
        hidden_size: Global model ``hidden_size`` (reserved for future use).
        moe_shared_expert_intermediate_omitted: True when the raw HF dict had no
            ``ffn.moe.shared_expert_intermediate_dim`` key (so we must not trust the
            dataclass default 8192).

    Returns:
        MCoreLayerOverrides with a config_overrides dict and no_op flags.

    Raises:
        ValueError: If GQA ``num_key_value_heads`` does not evenly divide ``num_attention_heads``.
    """
    if isinstance(block_config, dict):
        # Import BlockConfig lazily so this module can be imported without ModelOpt.
        from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig
        block_config = BlockConfig(**block_config)

    config_overrides: Dict[str, Any] = {}
    attention_no_op = False
    mlp_no_op = False

    # --- Attention / Mamba slot ---
    attn = block_config.attention
    if attn is not None:
        attention_no_op = bool(getattr(attn, "no_op", False))
        mamba_cfg = getattr(attn, "mamba", None)

        if not attention_no_op:
            if mamba_cfg is not None:
                # Mamba slot: the "attention" field carries Mamba2 hyperparameters.
                _set_if_not_none(config_overrides, "mamba_state_dim", getattr(mamba_cfg, "state_dim", None))
                _set_if_not_none(config_overrides, "mamba_head_dim", getattr(mamba_cfg, "head_dim", None))
                _set_if_not_none(config_overrides, "mamba_num_groups", getattr(mamba_cfg, "num_groups", None))
                _set_if_not_none(config_overrides, "mamba_num_heads", getattr(mamba_cfg, "num_heads", None))
            else:
                # Standard attention slot: configure GQA if num_key_value_heads is specified.
                num_kv_heads = getattr(attn, "num_key_value_heads", None)
                if num_kv_heads is not None:
                    if num_attention_heads % num_kv_heads != 0:
                        raise ValueError(
                            f"num_attention_heads ({num_attention_heads}) must be divisible by "
                            f"num_key_value_heads ({num_kv_heads}) — required for GQA"
                        )
                    config_overrides["num_query_groups"] = num_kv_heads

    # --- FFN / MLP / MoE slot ---
    ffn = block_config.ffn
    if ffn is not None:
        mlp_no_op = bool(getattr(ffn, "no_op", False))

        if not mlp_no_op:
            is_moe = getattr(ffn, "is_moe", False) or getattr(ffn, "moe", None) is not None
            if is_moe:
                moe_cfg = getattr(ffn, "moe", None)
                if moe_cfg is not None:
                    _set_if_not_none(config_overrides, "num_moe_experts",
                                     getattr(moe_cfg, "num_local_experts", None))
                    _set_if_not_none(config_overrides, "moe_router_topk",
                                     getattr(moe_cfg, "num_experts_per_tok", None))
                    _set_if_not_none(config_overrides, "moe_ffn_hidden_size",
                                     getattr(moe_cfg, "expert_intermediate_dim", None))
                    _set_if_not_none(config_overrides, "moe_shared_expert_intermediate_size",
                                     getattr(moe_cfg, "shared_expert_intermediate_dim", None))
            else:
                # Dense MLP: override per-layer FFN hidden size if specified.
                intermediate_size = getattr(ffn, "intermediate_size", None)
                if intermediate_size is not None:
                    config_overrides["ffn_hidden_size"] = intermediate_size

    return MCoreLayerOverrides(
        config_overrides=config_overrides,
        attention_no_op=attention_no_op,
        mlp_no_op=mlp_no_op,
    )


def get_overrides_for_layer(
    block_configs: List[Union["BlockConfig", dict]],
    layer_number: int,
    num_attention_heads: int,
    hidden_size: int,
    *,
    strict_mamba_slot: bool = False,
) -> Optional[MCoreLayerOverrides]:
    """Return MCoreLayerOverrides for the given 1-based global layer number.

    Args:
        block_configs: List of block configs, one per decoder layer/slot.
        layer_number: 1-based global layer index. MCore sets this as ``i + 1`` (plus any
            pipeline-parallel offset) when constructing TransformerBlock or MambaStack layers.
        num_attention_heads: Global model num_attention_heads.
        hidden_size: Global model hidden_size.
        strict_mamba_slot: If True, raise ValueError when a slot's block_config has neither
            attention nor ffn as no_op. Used for MambaStack where each slot must be a single
            dedicated subblock type (mamba, attention, mlp, or moe) — not a combined layer.

    Returns:
        MCoreLayerOverrides for that layer, or None if layer_number is out of range.
    """
    if not block_configs:
        return None

    layer_idx = layer_number - 1
    if layer_idx < 0 or layer_idx >= len(block_configs):
        logger.debug(
            "layer_number %d out of block_configs range (%d entries); no overrides applied",
            layer_number,
            len(block_configs),
        )
        return None

    # Cast to typed BlockConfig objects for consistent attribute access.
    from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
    typed = maybe_cast_block_configs(block_configs)

    overrides = block_config_to_mcore_overrides(
        typed[layer_idx],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
    )

    if strict_mamba_slot and not overrides.attention_no_op and not overrides.mlp_no_op:
        raise ValueError(
            f"Layer {layer_number}: Mamba slot block_config has neither attention nor ffn "
            "as no_op. Each MambaStack slot must be a single subblock type "
            "(mamba, attention, mlp, or moe). Check block_configs[%d]." % layer_idx
        )

    return overrides


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_if_not_none(d: dict, key: str, value: Any) -> None:
    """Add key → value to dict only if value is not None."""
    if value is not None:
        d[key] = value
