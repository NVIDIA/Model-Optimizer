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

"""Resolver: transforms validated RecipeConfig into mtq-compatible config dicts.

The resolver is the core translation layer between human-readable YAML
and the internal config dicts that mtq.quantize() and mtq.auto_quantize() accept.
"""

from __future__ import annotations

import copy
from typing import Any

from .formats import get_format, get_kv_format
from .models import (
    AlgorithmConfig,
    AutoQuantizeSection,
    OverrideEntry,
    QuantizationSection,
    QuantizerSpec,
    RecipeConfig,
)
from .presets import get_preset

# Fallback disabled quantizer patterns — used when PR #1000's load_config is not available.
# Matches PR #1000's configs/ptq/base.yml (torch.nn. prefix, not nn.).
_FALLBACK_DISABLED_QUANTIZER_CFG: dict[str, Any] = {
    "torch.nn.BatchNorm1d": {"*": {"enable": False}},
    "torch.nn.BatchNorm2d": {"*": {"enable": False}},
    "torch.nn.BatchNorm3d": {"*": {"enable": False}},
    "torch.nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},
    "*router*": {"enable": False},
    "*mlp.gate.*": {"enable": False},
    "*mlp.shared_expert_gate.*": {"enable": False},
    "*linear_attn.conv1d*": {"enable": False},
    "*mixer.conv1d*": {"enable": False},
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},
}


def _load_disabled_quantizer_cfg() -> dict[str, Any]:
    """Load disabled quantizer config: prefer PR #1000's base.yml, fall back to inline."""
    from modelopt.torch.recipes.utils import try_import_load_config

    load_config = try_import_load_config()
    if load_config is not None:
        try:
            cfg = load_config("configs/ptq/base")
            return cfg["quant_cfg"]
        except (ValueError, KeyError, TypeError):
            pass
    return _FALLBACK_DISABLED_QUANTIZER_CFG


_DEFAULT_DISABLED_QUANTIZER_CFG: dict[str, Any] = _load_disabled_quantizer_cfg()


def _update_quant_cfg_with_kv_cache(
    quant_cfg: dict[str, Any], kv_cache_quant_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Merge KV cache quantizer patterns into the main config.

    Equivalent to modelopt.torch.quantization.utils.update_quant_cfg_with_kv_cache_quant().

    Uses dict.update() so the user's explicit kv_cache section in the recipe
    always wins over any KV patterns baked into the preset (e.g., from Tier 1
    YAML presets that include kv_quant.yml).
    """
    quant_cfg["quant_cfg"] = quant_cfg.get("quant_cfg", {"default": {"enable": False}})
    quant_cfg["quant_cfg"].update(kv_cache_quant_cfg)
    if not quant_cfg.get("algorithm"):
        quant_cfg["algorithm"] = "max"
    return quant_cfg


def resolve_recipe(recipe: RecipeConfig) -> dict[str, Any]:
    """Resolve a RecipeConfig into output dict(s) for mtq APIs.

    Returns a dict with keys:
      - "quantize_config": config dict for mtq.quantize() (if quantization section present)
      - "auto_quantize_kwargs": kwargs dict for mtq.auto_quantize() (if auto_quantize section)
      - "calibration": calibration params dict (if specified)
      - "export": export params dict (if specified)
    """
    result: dict[str, Any] = {}

    if recipe.quantization:
        result["quantize_config"] = _resolve_quantization(recipe.quantization)
        if recipe.quantization.calibration:
            result["calibration"] = recipe.quantization.calibration.model_dump()

    if recipe.auto_quantize:
        result["auto_quantize_kwargs"] = _resolve_auto_quantize(recipe.auto_quantize)
        if recipe.auto_quantize.calibration:
            result["calibration"] = recipe.auto_quantize.calibration.model_dump()

    if recipe.export:
        result["export"] = recipe.export.model_dump()

    return result


def _resolve_quantization(section: QuantizationSection) -> dict[str, Any]:
    """Produce the config dict for mtq.quantize()."""
    # Step 1: Start from preset or build from scratch
    if section.preset:
        config = get_preset(section.preset)
    else:
        quant_cfg: dict[str, Any] = {}
        if section.weights:
            quant_cfg["*weight_quantizer"] = _resolve_quantizer_spec(section.weights, "weight")
        if section.activations:
            quant_cfg["*input_quantizer"] = _resolve_quantizer_spec(
                section.activations, "activation"
            )
        quant_cfg.update(_DEFAULT_DISABLED_QUANTIZER_CFG)
        config = {"quant_cfg": quant_cfg, "algorithm": "max"}

    # Step 2: Apply algorithm override
    if section.algorithm is not None:
        if isinstance(section.algorithm, str):
            config["algorithm"] = section.algorithm
        elif isinstance(section.algorithm, AlgorithmConfig):
            algo_dict = section.algorithm.model_dump(exclude_none=True)
            config["algorithm"] = algo_dict

    # Step 3: Apply overrides
    for override in section.overrides:
        _apply_override(config["quant_cfg"], override)

    # Step 4: Apply disabled_patterns
    for pattern in section.disabled_patterns:
        config["quant_cfg"][pattern] = {"enable": False}

    # Step 5: Merge KV cache
    if section.kv_cache:
        kv_cfg = copy.deepcopy(get_kv_format(section.kv_cache.format))
        config = _update_quant_cfg_with_kv_cache(config, kv_cfg)

    return config


def _resolve_quantizer_spec(spec: QuantizerSpec, target: str) -> dict[str, Any] | list[dict]:
    """Convert a QuantizerSpec to quantizer attribute dict(s).

    Args:
        spec: The quantizer specification from the recipe YAML.
        target: "weight" or "activation" — used to pick format defaults.

    Returns:
        A dict of quantizer attributes, or a list of dicts for staged quantization.
    """
    if spec.stages:
        return [_resolve_single_quantizer(stage, target) for stage in spec.stages]
    return _resolve_single_quantizer(spec, target)


def _resolve_single_quantizer(spec: QuantizerSpec, target: str) -> dict[str, Any]:
    """Resolve a single (non-staged) quantizer spec to attribute dict."""
    result: dict[str, Any] = {}

    if spec.format:
        fmt = get_format(spec.format)
        result.update(copy.deepcopy(fmt[target]))

    # Expert-mode overrides
    if spec.num_bits is not None:
        result["num_bits"] = spec.num_bits
    if spec.axis is not None:
        result["axis"] = spec.axis
    if spec.block_sizes is not None:
        result["block_sizes"] = _resolve_block_sizes(spec.block_sizes)
    if not spec.enable:
        result["enable"] = False

    return result


def _resolve_block_sizes(bs: dict[str, Any]) -> dict:
    """Convert block_sizes from YAML-friendly format to internal format.

    YAML uses string keys like "last_dim"; internal uses integer keys like -1.
    Also supports passing through raw dicts with integer keys directly.
    """
    result: dict = {}
    key_map = {"last_dim": -1, "second_last_dim": -2}

    for k, v in bs.items():
        if k in key_map:
            result[key_map[k]] = v
        elif k == "scale_bits":
            result["scale_bits"] = v
        else:
            # Pass through: "type", "scale_bits" (tuple), integer keys, etc.
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                result[k] = v

    return result


def _apply_override(quant_cfg: dict, override: OverrideEntry) -> None:
    """Apply a single override entry to the quant_cfg dict.

    Pattern overrides merge into existing entries (preserving preset values).
    Module-class overrides also merge to avoid dropping defaults like disabled BatchNorm.
    """
    if override.pattern:
        # Start from existing entry if present (to preserve preset values for merging)
        entry: dict[str, Any] = copy.deepcopy(quant_cfg.get(override.pattern, {}))
        if override.enable is not None:
            entry["enable"] = override.enable
        if override.format:
            fmt = get_format(override.format)
            entry.update(copy.deepcopy(fmt["weight"]))
        if override.weights:
            entry.update(_resolve_single_quantizer(override.weights, "weight"))
        if override.activations:
            entry.update(_resolve_single_quantizer(override.activations, "activation"))
        if override.scale_type:
            # Merge scale_type into block_sizes.type, preserving existing block_sizes
            bs = entry.get("block_sizes", {})
            bs["type"] = override.scale_type
            entry["block_sizes"] = bs
        if override.num_bits is not None:
            entry["num_bits"] = override.num_bits
        if override.axis is not None:
            entry["axis"] = override.axis
        quant_cfg[override.pattern] = entry

    elif override.module_class:
        # Merge into existing entry to preserve defaults (e.g., disabled BatchNorm)
        mc_cfg: dict[str, Any] = copy.deepcopy(quant_cfg.get(override.module_class, {}))
        if override.weights:
            mc_cfg["*weight_quantizer"] = _resolve_quantizer_spec(override.weights, "weight")
        if override.activations:
            mc_cfg["*input_quantizer"] = _resolve_quantizer_spec(override.activations, "activation")
        if override.enable is not None and not override.weights and not override.activations:
            mc_cfg = {"*": {"enable": override.enable}}
        quant_cfg[override.module_class] = mc_cfg


def _resolve_auto_quantize(section: AutoQuantizeSection) -> dict[str, Any]:
    """Produce kwargs dict for mtq.auto_quantize()."""
    format_configs = [get_preset(fmt_entry.preset) for fmt_entry in section.formats]

    kwargs: dict[str, Any] = {
        "constraints": {"effective_bits": section.effective_bits},
        "quantization_formats": format_configs,
        "num_calib_steps": section.num_calib_steps,
        "num_score_steps": section.num_score_steps,
        "method": section.method,
    }

    if section.disabled_patterns:
        kwargs["disabled_layers"] = section.disabled_patterns

    if section.kv_cache:
        kv_cfg = copy.deepcopy(get_kv_format(section.kv_cache.format))
        # Apply KV cache quantization to each candidate format
        for fmt_cfg in format_configs:
            _update_quant_cfg_with_kv_cache(fmt_cfg, kv_cfg)

    return kwargs
