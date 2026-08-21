# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Puzzletron typed block configs -> vLLM/AnyModel config adapter.

Puzzletron stores canonical checkpoints with a dense ``block_configs`` list of
typed ``subblock_configs``. The vLLM AnyModel fork consumes the HuggingFace
heterogeneity schema: a sparse ``per_layer_config`` dict mapping
``layer_idx -> {flat HF keys + optional "skip" list}``.

This module rewrites the typed Puzzletron schema in-place so vLLM only sees
``per_layer_config``. When both forms are present, they must be equivalent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# (num_experts_field, moe_intermediate_size_field) per base architecture.
# The typed MoEConfig uses normalized Puzzletron names; the adapter rewrites
# them into the base HF config fields that vLLM's AnyModel fork reads.
_MOE_FIELDS_BY_ARCH: dict[str, tuple[str, str]] = {
    "Qwen2MoeForCausalLM": ("num_experts", "moe_intermediate_size"),
    "Qwen3MoeForCausalLM": ("num_experts", "moe_intermediate_size"),
    "Qwen3_5MoeForCausalLM": ("num_experts", "moe_intermediate_size"),
    "Qwen3_5MoeForConditionalGeneration": ("num_experts", "moe_intermediate_size"),
    "Qwen3_6MoeForCausalLM": ("num_experts", "moe_intermediate_size"),
    "Qwen3_6MoeForConditionalGeneration": ("num_experts", "moe_intermediate_size"),
    "MixtralForCausalLM": ("num_local_experts", "intermediate_size"),
    "GptOssForCausalLM": ("num_local_experts", "intermediate_size"),
    "NemotronHForCausalLM": ("n_routed_experts", "moe_intermediate_size"),
    "DeepseekV3ForCausalLM": ("n_routed_experts", "moe_intermediate_size"),
    "DeepseekV2ForCausalLM": ("n_routed_experts", "moe_intermediate_size"),
}

_DEFAULT_MOE_FIELDS: tuple[str, str] = ("num_local_experts", "intermediate_size")

_MOE_TOPK_FIELD_BY_ARCH: dict[str, str] = {
    "Qwen2MoeForCausalLM": "num_experts_per_tok",
    "Qwen3MoeForCausalLM": "num_experts_per_tok",
    "Qwen3_5MoeForCausalLM": "num_experts_per_tok",
    "Qwen3_5MoeForConditionalGeneration": "num_experts_per_tok",
    "Qwen3_6MoeForCausalLM": "num_experts_per_tok",
    "Qwen3_6MoeForConditionalGeneration": "num_experts_per_tok",
    "MixtralForCausalLM": "num_experts_per_tok",
    "GptOssForCausalLM": "num_experts_per_tok",
    "NemotronHForCausalLM": "num_experts_per_tok",
    "NemotronHV2ForCausalLM": "num_experts_per_tok",
    "DeepseekV3ForCausalLM": "num_experts_per_tok",
    "DeepseekV2ForCausalLM": "num_experts_per_tok",
}

_MOE_SHARED_FIELD_BY_ARCH: dict[str, str] = {
    "Qwen2MoeForCausalLM": "shared_expert_intermediate_size",
    "Qwen3MoeForCausalLM": "shared_expert_intermediate_size",
    "Qwen3_5MoeForCausalLM": "shared_expert_intermediate_size",
    "Qwen3_5MoeForConditionalGeneration": "shared_expert_intermediate_size",
    "Qwen3_6MoeForCausalLM": "shared_expert_intermediate_size",
    "Qwen3_6MoeForConditionalGeneration": "shared_expert_intermediate_size",
    "NemotronHForCausalLM": "moe_shared_expert_intermediate_size",
    "NemotronHV2ForCausalLM": "moe_shared_expert_intermediate_size",
    "DeepseekV3ForCausalLM": "moe_shared_expert_intermediate_size",
    "DeepseekV2ForCausalLM": "moe_shared_expert_intermediate_size",
}

_MOE_LATENT_FIELD_BY_ARCH: dict[str, str] = {
    "NemotronHForCausalLM": "moe_latent_size",
    "NemotronHV2ForCausalLM": "moe_latent_size",
}

_MAMBA_FIELDS_BY_ARCH: dict[str, dict[str, str]] = {
    "NemotronHForCausalLM": {
        "num_heads": "mamba_num_heads",
        "head_dim": "mamba_head_dim",
        "state_dim": "ssm_state_size",
        "num_groups": "n_groups",
        "conv_kernel_size": "conv_kernel",
    },
    "NemotronHV2ForCausalLM": {
        "num_heads": "mamba_num_heads",
        "head_dim": "mamba_head_dim",
        "state_dim": "ssm_state_size",
        "num_groups": "n_groups",
        "conv_kernel_size": "conv_kernel",
    },
    "Qwen3_5ForConditionalGeneration": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_5ForCausalLM": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_6ForConditionalGeneration": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_6ForCausalLM": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_5MoeForConditionalGeneration": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_5MoeForCausalLM": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_6MoeForConditionalGeneration": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
    "Qwen3_6MoeForCausalLM": {
        "num_heads": "linear_num_value_heads",
        "head_dim": "linear_value_head_dim",
        "state_dim": "linear_key_head_dim",
        "num_groups": "linear_num_key_heads",
        "conv_kernel_size": "linear_conv_kernel_dim",
    },
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _subblock(block: Any, kind: str) -> Any:
    subblocks = _get(block, "subblock_configs") or []
    for subblock in subblocks:
        if _get(subblock, "kind") == kind:
            return subblock
    return None


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _delete(obj: Any, key: str) -> None:
    if isinstance(obj, dict):
        obj.pop(key, None)
        return
    try:
        delattr(obj, key)
    except AttributeError:
        pass


def _get_text_config(hf_config: Any) -> Any:
    if hasattr(hf_config, "get_text_config"):
        return hf_config.get_text_config()
    return _get(hf_config, "text_config", hf_config)


def configure_anymodel_metadata(hf_config: Any, descriptor: Any) -> bool:
    """Attach the vLLM AnyModel wrapper contract to a realized HF config.

    ``model_type`` remains unchanged, so Transformers and native AutoModel
    loaders still resolve the canonical family.  vLLM selects its heterogeneous
    wrapper from ``architectures`` and reconstructs each layer from the sparse
    ``per_layer_config`` emitted below.
    """
    architectures = list(_get(hf_config, "architectures", None) or [])
    base_architecture = _get(hf_config, "base_architecture")
    if not base_architecture:
        base_architecture = next((name for name in architectures if name != "AnyModel"), None)
    if not base_architecture:
        return False

    _set(hf_config, "architectures", ["AnyModel"])
    _set(hf_config, "base_architecture", base_architecture)
    _set(hf_config, "anymodel_arch_info", dict(descriptor.anymodel_arch_info()))
    return True


def refresh_realized_checkpoint_config(
    checkpoint_dir: str | Path,
    *,
    trust_remote_code: bool = False,
) -> Path:
    """Rebuild vLLM interchange fields without trusting checkpoint code by default."""
    from transformers import AutoConfig

    from ..anymodel.registry import resolve_descriptor

    checkpoint_dir = Path(checkpoint_dir)
    config = AutoConfig.from_pretrained(
        checkpoint_dir,
        trust_remote_code=trust_remote_code,
    )
    descriptor = resolve_descriptor(config).descriptor
    text_config = _get_text_config(config)

    # This operation is intentionally a re-derivation from canonical
    # ``block_configs``; it also repairs artifacts written by older adapters
    # that exported only a subset of the heterogeneous fields.
    _delete(text_config, "per_layer_config")
    if not configure_anymodel_metadata(config, descriptor):
        raise ValueError(f"Cannot infer base architecture for {checkpoint_dir}")
    if not convert_block_configs_to_per_layer_config(config, keep_block_configs=True):
        raise ValueError(f"Checkpoint has no block_configs: {checkpoint_dir}")

    config_path = checkpoint_dir / "config.json"
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(config.to_json_string(use_diff=False), encoding="utf-8")
    temporary.replace(config_path)
    return config_path


def _is_qwen3p5_config(text_config: Any, base_architecture: str) -> bool:
    model_type = str(_get(text_config, "model_type", ""))
    return model_type.startswith(("qwen3_5", "qwen3_6")) or base_architecture.startswith(
        ("Qwen3_5", "Qwen3_6")
    )


def _is_mamba_block(block: Any) -> bool:
    return _subblock(block, "mamba") is not None


def _update_qwen3p5_layer_types(
    text_config: Any, block_configs: list[Any], base_architecture: str
) -> bool:
    if not _is_qwen3p5_config(text_config, base_architecture):
        return False

    layer_types = []
    for block in block_configs:
        layer_types.append("linear_attention" if _is_mamba_block(block) else "full_attention")

    if _get(text_config, "layer_types") == layer_types:
        return False

    _set(text_config, "layer_types", layer_types)
    return True


def _convert_block_entry(
    block: Any,
    *,
    layer_idx: int,
    global_layer_types: list[str],
    global_sliding_window: int | None,
    global_q: int | None,
    global_kv: int | None,
    global_global_kv: int | None,
    global_head_dim: int | None,
    global_global_head_dim: int | None,
    global_isize: int | None,
    ffn_constructor_divisor: int,
    global_hact: str | None,
    global_moe_num: int | None,
    global_moe_size: int | None,
    global_moe_top_k: int | None,
    global_shared_moe_size: int | None,
    global_moe_latent_size: int | None,
    global_mamba_values: dict[str, Any],
    global_q_lora_rank: int | None,
    global_kv_lora_rank: int | None,
    moe_num_field: str,
    moe_size_field: str,
    moe_top_k_field: str,
    moe_shared_field: str | None,
    moe_latent_field: str | None,
    mamba_fields: dict[str, str],
) -> dict[str, Any]:
    """Translate one typed ``block_configs`` entry into a flat per-layer dict."""
    attn = _subblock(block, "attention") or {}
    ffn = _subblock(block, "ffn") or {}
    moe = _subblock(block, "moe")
    mamba = _subblock(block, "mamba")
    mla = _subblock(block, "mla")
    a_noop = bool(_get(attn, "no_op", False))
    f_noop = bool(_get(ffn, "no_op", False)) or bool(_get(moe, "no_op", False))
    a_is_mamba = mamba is not None

    entry: dict[str, Any] = {}
    skip: list[str] = []
    if a_noop or bool(_get(mamba, "no_op", False)):
        skip.append("attention")
    if f_noop:
        skip.append("mlp")
    if skip:
        entry["skip"] = skip

    if not a_noop and not a_is_mamba:
        q = _get(attn, "num_query_heads")
        if q is not None and q != global_q:
            entry["num_attention_heads"] = q

        k_eq_v = bool(_get(attn, "k_eq_v", False))
        kv = _get(attn, "num_kv_heads")
        kv_field = "num_global_key_value_heads" if k_eq_v else "num_key_value_heads"
        current_kv = global_global_kv if k_eq_v else global_kv
        if kv is not None and kv != current_kv:
            entry[kv_field] = kv

        qk_head_dim = _get(attn, "qk_head_dim")
        if qk_head_dim is not None:
            is_full = _get(attn, "sliding_window_size") == "full"
            head_dim_field = "global_head_dim" if (is_full or k_eq_v) else "head_dim"
            current_head_dim = (
                global_global_head_dim if head_dim_field == "global_head_dim" else global_head_dim
            )
            if qk_head_dim != current_head_dim:
                entry[head_dim_field] = qk_head_dim

        window = _get(attn, "sliding_window_size")
        if window is not None:
            desired_type = "full_attention" if window == "full" else "sliding_attention"
            current_type = (
                global_layer_types[layer_idx] if layer_idx < len(global_layer_types) else None
            )
            if global_layer_types and current_type != desired_type:
                layer_types = list(global_layer_types)
                layer_types[layer_idx] = desired_type
                entry["layer_types"] = layer_types
            current_window = (
                int(global_sliding_window)
                if global_sliding_window is not None
                and (not global_layer_types or current_type == "sliding_attention")
                else None
            )
            desired_window = None if window == "full" else int(window)
            if desired_window != current_window:
                entry["sliding_window"] = desired_window

    if mla is not None and not bool(_get(mla, "no_op", False)):
        mla_heads = _get(mla, "num_heads")
        if mla_heads is not None and mla_heads != global_q:
            entry["num_attention_heads"] = mla_heads
        q_lora_rank = _get(mla, "q_lora_rank")
        if q_lora_rank is not None and q_lora_rank != global_q_lora_rank:
            entry["q_lora_rank"] = q_lora_rank
        kv_lora_rank = _get(mla, "kv_lora_rank")
        if kv_lora_rank is not None and kv_lora_rank != global_kv_lora_rank:
            entry["kv_lora_rank"] = kv_lora_rank

    if not f_noop:
        isize = _get(ffn, "intermediate_size")
        if isize is not None and ffn_constructor_divisor != 1:
            if int(isize) % ffn_constructor_divisor:
                raise ValueError(
                    f"layer {layer_idx} FFN width {isize} is not divisible by "
                    f"constructor divisor {ffn_constructor_divisor}"
                )
            isize = int(isize) // ffn_constructor_divisor
        if isize is not None and isize != global_isize:
            entry["intermediate_size"] = isize

        hact = _get(ffn, "hidden_act")
        if hact is not None and hact != global_hact:
            entry["hidden_act"] = hact

        if moe:
            n_exp = _get(moe, "num_experts")
            if n_exp is not None and n_exp != global_moe_num:
                entry[moe_num_field] = n_exp

            exp_size = _get(moe, "expert_intermediate_size")
            if exp_size is not None and exp_size != global_moe_size:
                entry[moe_size_field] = exp_size

            top_k = _get(moe, "top_k")
            if top_k is not None and top_k != global_moe_top_k:
                entry[moe_top_k_field] = top_k

            shared_size = _get(moe, "shared_expert_intermediate_size")
            if (
                moe_shared_field is not None
                and shared_size is not None
                and shared_size != global_shared_moe_size
            ):
                entry[moe_shared_field] = shared_size

            latent_size = _get(moe, "latent_dim")
            if (
                moe_latent_field is not None
                and latent_size is not None
                and latent_size != global_moe_latent_size
            ):
                entry[moe_latent_field] = latent_size

    if mamba is not None and not bool(_get(mamba, "no_op", False)):
        for typed_field, hf_field in mamba_fields.items():
            value = _get(mamba, typed_field)
            if value is not None and value != global_mamba_values.get(hf_field):
                entry[hf_field] = value

    return entry


def _derive_per_layer_config(
    *,
    text_config: Any,
    block_configs: list[Any],
    base_architecture: str,
) -> dict[str, dict[str, Any]]:
    moe_num_field, moe_size_field = _MOE_FIELDS_BY_ARCH.get(base_architecture, _DEFAULT_MOE_FIELDS)
    moe_top_k_field = _MOE_TOPK_FIELD_BY_ARCH.get(base_architecture, "num_experts_per_tok")
    moe_shared_field = _MOE_SHARED_FIELD_BY_ARCH.get(base_architecture)
    moe_latent_field = _MOE_LATENT_FIELD_BY_ARCH.get(base_architecture)
    mamba_fields = _MAMBA_FIELDS_BY_ARCH.get(base_architecture, {})

    global_q = _get(text_config, "num_attention_heads")
    global_kv = _get(text_config, "num_key_value_heads")
    global_global_kv = _get(text_config, "num_global_key_value_heads")
    global_head_dim = _get(text_config, "head_dim")
    global_global_head_dim = _get(text_config, "global_head_dim")
    global_isize = _get(text_config, "intermediate_size")
    global_hact = _get(text_config, "hidden_act")
    global_moe_num = _get(text_config, moe_num_field)
    global_moe_size = _get(text_config, moe_size_field)
    global_moe_top_k = _get(text_config, moe_top_k_field)
    global_shared_moe_size = (
        _get(text_config, moe_shared_field) if moe_shared_field is not None else None
    )
    global_moe_latent_size = (
        _get(text_config, moe_latent_field) if moe_latent_field is not None else None
    )
    global_mamba_values = {
        hf_field: _get(text_config, hf_field) for hf_field in mamba_fields.values()
    }
    global_q_lora_rank = _get(text_config, "q_lora_rank")
    global_kv_lora_rank = _get(text_config, "kv_lora_rank")
    global_layer_types = list(_get(text_config, "layer_types") or ())
    global_sliding_window = _get(text_config, "sliding_window")
    num_hidden_layers = int(_get(text_config, "num_hidden_layers") or len(block_configs))
    num_kv_shared_layers = int(_get(text_config, "num_kv_shared_layers") or 0)
    first_shared_layer = num_hidden_layers - num_kv_shared_layers
    use_double_wide_mlp = bool(_get(text_config, "use_double_wide_mlp") or False)

    per_layer_config: dict[str, dict[str, Any]] = {}
    for idx, block in enumerate(block_configs):
        ffn_constructor_divisor = (
            2
            if use_double_wide_mlp
            and num_kv_shared_layers > 0
            and first_shared_layer > 0
            and idx >= first_shared_layer
            else 1
        )
        entry = _convert_block_entry(
            block,
            layer_idx=idx,
            global_layer_types=global_layer_types,
            global_sliding_window=global_sliding_window,
            global_q=global_q,
            global_kv=global_kv,
            global_global_kv=global_global_kv,
            global_head_dim=global_head_dim,
            global_global_head_dim=global_global_head_dim,
            global_isize=global_isize,
            ffn_constructor_divisor=ffn_constructor_divisor,
            global_hact=global_hact,
            global_moe_num=global_moe_num,
            global_moe_size=global_moe_size,
            global_moe_top_k=global_moe_top_k,
            global_shared_moe_size=global_shared_moe_size,
            global_moe_latent_size=global_moe_latent_size,
            global_mamba_values=global_mamba_values,
            global_q_lora_rank=global_q_lora_rank,
            global_kv_lora_rank=global_kv_lora_rank,
            moe_num_field=moe_num_field,
            moe_size_field=moe_size_field,
            moe_top_k_field=moe_top_k_field,
            moe_shared_field=moe_shared_field,
            moe_latent_field=moe_latent_field,
            mamba_fields=mamba_fields,
        )
        if entry:
            per_layer_config[str(idx)] = entry
    return per_layer_config


def convert_block_configs_to_per_layer_config(
    hf_config: Any, *, keep_block_configs: bool = False
) -> bool:
    """In-place: convert typed ``block_configs`` to ``per_layer_config``.

    Returns ``True`` if a conversion happened, ``False`` if there was
    nothing to convert. If both representations exist, they must match.

    ``keep_block_configs`` is intended for canonical realized checkpoints that
    must be consumable by both Puzzletron's patched HF model constructors and
    vLLM AnyModel.  The default remains an export-only conversion that removes
    the Puzzletron representation after deriving vLLM's representation.
    """
    block_configs = _get(hf_config, "block_configs")
    if not block_configs:
        return False

    text_config = _get_text_config(hf_config)
    base_architecture = _get(hf_config, "base_architecture", "") or ""
    layer_types_updated = _update_qwen3p5_layer_types(text_config, block_configs, base_architecture)

    per_layer_config = _derive_per_layer_config(
        text_config=text_config,
        block_configs=block_configs,
        base_architecture=base_architecture,
    )

    existing = _get(text_config, "per_layer_config")
    if existing:
        if existing != per_layer_config:
            raise ValueError(
                "AnyModel config carries both block_configs and per_layer_config, "
                "but they are not equivalent. Keep only one canonical Puzzletron "
                "representation before exporting to vLLM."
            )
        if not keep_block_configs:
            _delete(hf_config, "block_configs")
        return layer_types_updated

    n_layers = _get(text_config, "num_hidden_layers")
    if n_layers is not None and len(block_configs) != n_layers:
        logger.warning(
            "block_configs length (%d) does not match num_hidden_layers "
            "(%d); converted entries beyond num_hidden_layers will fail "
            "AnyModel validation.",
            len(block_configs),
            n_layers,
        )

    _set(text_config, "per_layer_config", per_layer_config)
    if not keep_block_configs:
        _delete(hf_config, "block_configs")

    logger.info(
        "Converted ModelOpt block_configs (%d entries) to AnyModel "
        "per_layer_config (%d non-empty entries) for base_architecture=%r.",
        len(block_configs),
        len(per_layer_config),
        base_architecture or "<unknown>",
    )
    return True
