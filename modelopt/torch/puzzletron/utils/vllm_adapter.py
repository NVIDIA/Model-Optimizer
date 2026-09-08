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
"""ModelOpt/AnyModel -> vLLM/AnyModel config adapter.

ModelOpt/AnyModel checkpoints describe per-layer overrides via a dense
``block_configs`` list with nested ``attention`` / ``ffn`` sub-sections.
AnyModel in vLLM now consumes the HuggingFace heterogeneity schema: a sparse
``per_layer_config`` dict mapping ``layer_idx -> {flat HF keys + optional
"skip" list}``.

This module rewrites the Puzzletron schema in-place so vLLM only
ever sees ``per_layer_config``. It is invoked from
``AnyModelConfig.verify_and_update_model_config`` before the arch
convertor or layer-patching runs.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


# (num_experts_field, moe_intermediate_size_field) per base architecture.
# ModelOpt always writes ``moe.num_local_experts`` and
# ``moe.expert_intermediate_{size,dim}``; the adapter rewrites them into the
# field names the base HF config actually reads.
_MOE_FIELDS_BY_ARCH: dict[str, tuple[str, str]] = {
    "Qwen2MoeForCausalLM": ("num_experts", "moe_intermediate_size"),
    "Qwen3MoeForCausalLM": ("num_experts", "moe_intermediate_size"),
    "MixtralForCausalLM": ("num_local_experts", "intermediate_size"),
    "GptOssForCausalLM": ("num_local_experts", "intermediate_size"),
    "NemotronHForCausalLM": ("n_routed_experts", "moe_intermediate_size"),
    "DeepseekV3ForCausalLM": ("n_routed_experts", "moe_intermediate_size"),
    "DeepseekV2ForCausalLM": ("n_routed_experts", "moe_intermediate_size"),
}

_DEFAULT_MOE_FIELDS: tuple[str, str] = ("num_local_experts", "intermediate_size")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _convert_block_entry(
    block: Any,
    *,
    global_kv: int | None,
    global_isize: int | None,
    global_hact: str | None,
    global_moe_num: int | None,
    global_moe_size: int | None,
    moe_num_field: str,
    moe_size_field: str,
) -> dict[str, Any]:
    """Translate a single ModelOpt ``block_configs`` entry into a flat
    per-layer override dict. Only attributes that differ from the global
    config are emitted; sub-module no-ops become a ``"skip"`` list."""
    attn = _get(block, "attention") or {}
    ffn = _get(block, "ffn") or {}
    a_noop = bool(_get(attn, "no_op", False))
    f_noop = bool(_get(ffn, "no_op", False))

    entry: dict[str, Any] = {}
    skip: list[str] = []
    if a_noop:
        skip.append("attention")
    if f_noop:
        skip.append("mlp")
    if skip:
        entry["skip"] = skip

    if not a_noop:
        kv = _get(attn, "num_key_value_heads")
        if kv is not None and kv != global_kv:
            entry["num_key_value_heads"] = kv

    if not f_noop:
        isize = _get(ffn, "intermediate_size")
        if isize is not None and isize != global_isize:
            entry["intermediate_size"] = isize

        hact = _get(ffn, "hidden_act")
        if hact is not None and hact != global_hact:
            entry["hidden_act"] = hact

        moe = _get(ffn, "moe")
        if moe:
            n_exp = _get(moe, "num_local_experts")
            if n_exp is None:
                n_exp = _get(moe, "num_experts")
            if n_exp is None:
                n_exp = _get(moe, "n_routed_experts")
            if n_exp is not None and n_exp != global_moe_num:
                entry[moe_num_field] = n_exp

            exp_size = _get(
                moe,
                "expert_intermediate_size",
                _get(moe, "expert_intermediate_dim"),
            )
            if exp_size is not None and exp_size != global_moe_size:
                entry[moe_size_field] = exp_size

    return entry


def convert_block_configs_to_per_layer_config(hf_config: Any) -> bool:
    """In-place: convert legacy ``block_configs`` on ``hf_config`` to
    ``per_layer_config`` on its text config.

    Returns ``True`` if a conversion happened, ``False`` if there was
    nothing to convert. If ``per_layer_config`` is already set, the legacy
    field is dropped and a warning emitted (the new schema wins).
    """
    block_configs = getattr(hf_config, "block_configs", None)
    if not block_configs:
        return False

    text_config = (
        hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
    )

    existing = getattr(text_config, "per_layer_config", None)
    if existing:
        logger.warning_once(
            "AnyModel config has both legacy 'block_configs' and new "
            "'per_layer_config'; using per_layer_config and ignoring "
            "block_configs."
        )
        if hasattr(hf_config, "block_configs"):
            try:
                delattr(hf_config, "block_configs")
            except AttributeError:
                pass
        return False

    base_architecture = getattr(hf_config, "base_architecture", None) or ""
    moe_num_field, moe_size_field = _MOE_FIELDS_BY_ARCH.get(base_architecture, _DEFAULT_MOE_FIELDS)

    global_kv = getattr(text_config, "num_key_value_heads", None)
    global_isize = getattr(text_config, "intermediate_size", None)
    global_hact = getattr(text_config, "hidden_act", None)
    global_moe_num = getattr(text_config, moe_num_field, None)
    global_moe_size = getattr(text_config, moe_size_field, None)

    per_layer_config: dict[str, dict[str, Any]] = {}
    for idx, block in enumerate(block_configs):
        entry = _convert_block_entry(
            block,
            global_kv=global_kv,
            global_isize=global_isize,
            global_hact=global_hact,
            global_moe_num=global_moe_num,
            global_moe_size=global_moe_size,
            moe_num_field=moe_num_field,
            moe_size_field=moe_size_field,
        )
        if entry:
            per_layer_config[str(idx)] = entry

    n_layers = getattr(text_config, "num_hidden_layers", None)
    if n_layers is not None and len(block_configs) != n_layers:
        logger.warning(
            "block_configs length (%d) does not match num_hidden_layers "
            "(%d); converted entries beyond num_hidden_layers will fail "
            "AnyModel validation.",
            len(block_configs),
            n_layers,
        )

    setattr(text_config, "per_layer_config", per_layer_config)
    try:
        delattr(hf_config, "block_configs")
    except AttributeError:
        pass

    logger.info(
        "Converted ModelOpt block_configs (%d entries) to AnyModel "
        "per_layer_config (%d non-empty entries) for base_architecture=%r.",
        len(block_configs),
        len(per_layer_config),
        base_architecture or "<unknown>",
    )
    return True
