# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-backed trainable-parameter selection for local distillation."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from transformers import PreTrainedModel

from ..anymodel.model_descriptor import ModelDescriptor
from ..block_config import maybe_cast_block_configs
from .bypass_utils import normalize_keys_to_learn

__all__ = ["set_keys_to_learn"]


def _param_names_for_subblock_key(
    model: PreTrainedModel,
    descriptor: type[ModelDescriptor] | ModelDescriptor,
    subblock_key: str,
) -> set[str]:
    lm_config = descriptor.get_language_model_config(model.config)
    state_keys = [
        name
        for name in model.state_dict()
        if "._fp32_params." not in name
        and "._fp32_buffers." not in name
        and not name.startswith("mtp.")
        and ".mtp." not in name
    ]
    weight_groups = descriptor.get_weight_groups(state_keys, lm_config.num_hidden_layers)
    attention_groups = [name for name in weight_groups if name.endswith("_attention")]
    ffn_groups = [name for name in weight_groups if name.endswith("_ffn")]
    if subblock_key == "subblock_attention":
        group_names = attention_groups
    elif subblock_key == "subblock_ffn":
        group_names = ffn_groups
    elif subblock_key == "subblock_mamba":
        group_names = attention_groups
    elif subblock_key == "entire_block":
        group_names = attention_groups + ffn_groups
    else:
        raise ValueError(f"Unsupported subblock key: {subblock_key!r}")

    block_configs = getattr(model.config, "block_configs", None) or getattr(
        lm_config, "block_configs", None
    )
    if block_configs is not None:
        block_configs = maybe_cast_block_configs(block_configs)
    if subblock_key == "subblock_mamba" and block_configs is None:
        raise ValueError("keys_to_learn='subblock_mamba' requires model config block_configs")

    selected: list[str] = []
    for group_name in group_names:
        if block_configs is not None:
            match = re.match(r"block_(\d+)_attention", group_name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx < len(block_configs):
                    is_mamba = block_configs[layer_idx].get_subblock("mamba") is not None
                    if subblock_key == "subblock_attention" and is_mamba:
                        continue
                    if subblock_key == "subblock_mamba" and not is_mamba:
                        continue
        selected.extend(weight_groups[group_name])
    return set(selected)


def set_keys_to_learn(
    model: PreTrainedModel,
    descriptor: type[ModelDescriptor] | ModelDescriptor,
    keys_to_learn: str | Sequence[str],
) -> tuple[str, ...]:
    """Enable gradients for the selected descriptor groups and return local names."""
    normalized = normalize_keys_to_learn(keys_to_learn)
    selected: set[str] = set()
    for subblock_key in normalized["subblocks"]:
        selected.update(_param_names_for_subblock_key(model, descriptor, subblock_key))

    enabled: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = name in selected and torch.is_floating_point(parameter)
        parameter.requires_grad_(trainable)
        if trainable:
            enabled.append(name)
    return tuple(sorted(enabled))

