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

"""Preset registry with tiered resolution.

Tier 1 (preferred): Load from modelopt_recipes/ YAML fragments
    - Uses the YAML fragment library from PR #1000's modelopt_recipes package
    - Resolves __base__ inheritance and merges fragments into complete configs
    - Requires modelopt_recipes package installed

Tier 2 (fallback): Live import from modelopt.torch.quantization.config
    - Gets preset dicts from Python constants (deprecated — team removing these)
    - Requires nvidia-modelopt[torch] installed

Tier 3 (last resort): Bundled snapshot dicts
    - Hardcoded copies of the preset dicts
    - Works on Mac without GPU, in CI, in dry-run mode
    - Must be periodically synced with upstream via scripts/sync_presets.py
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PRESET_REGISTRY: dict[str, dict[str, Any]] | None = None
_PRESET_SOURCE: str = "unknown"

# Mapping from preset name to the composed recipe directory in modelopt_recipes/.
# Each entry is a directory under general/ptq/ containing model_quant.yml + kv_quant.yml.
# The model_quant.yml uses __base__ to compose atomic fragments (base + quantizer + algorithm).
_PRESET_YAML_MAP: dict[str, str] = {
    # Core formats
    "fp8": "general/ptq/fp8_default-fp8_kv",
    "fp8_pc_pt": "general/ptq/fp8_per_channel_per_token-fp8_kv",
    "fp8_pb_wo": "general/ptq/fp8_2d_blockwise_weight_only-fp8_kv",
    "int8": "general/ptq/int8_default-fp8_kv",
    "int8_sq": "general/ptq/int8_smoothquant-fp8_kv",
    "int8_wo": "general/ptq/int8_weight_only-fp8_kv",
    "int4": "general/ptq/int4_blockwise_weight_only-fp8_kv",
    "int4_awq": "general/ptq/int4_awq-fp8_kv",
    # NVFP4 family
    "nvfp4": "general/ptq/nvfp4_default-fp8_kv",
    "nvfp4_awq": "general/ptq/nvfp4_awq_lite-fp8_kv",
    "nvfp4_awq_lite": "general/ptq/nvfp4_awq_lite-fp8_kv",
    "nvfp4_awq_clip": "general/ptq/nvfp4_awq_clip-fp8_kv",
    "nvfp4_awq_full": "general/ptq/nvfp4_awq_full-fp8_kv",
    "nvfp4_mse": "general/ptq/nvfp4_w4a4_weight_mse_fp8_sweep-fp8_kv",
    "nvfp4_local_hessian": "general/ptq/nvfp4_w4a4_weight_local_hessian-fp8_kv",
    "nvfp4_fp8_mha": "general/ptq/nvfp4_fp8_mha-fp8_kv",
    "nvfp4_svdquant": "general/ptq/nvfp4_svdquant_default-fp8_kv",
    "nvfp4_mlp_only": "general/ptq/nvfp4_mlp_only-fp8_kv",
    "nvfp4_mlp_wo": "general/ptq/nvfp4_mlp_weight_only-fp8_kv",
    # W4A8 variants
    "w4a8_awq": "general/ptq/w4a8_awq_beta-fp8_kv",
    "w4a8_nvfp4_fp8": "general/ptq/w4a8_nvfp4_fp8-fp8_kv",
    "w4a8_mxfp4_fp8": "general/ptq/w4a8_mxfp4_fp8-fp8_kv",
    # MX formats
    "mxfp8": "general/ptq/mxfp8_default-fp8_kv",
    "mxfp6": "general/ptq/mxfp6_default-fp8_kv",
    "mxfp4": "general/ptq/mxfp4_default-fp8_kv",
    "mxint8": "general/ptq/mxint8_default-fp8_kv",
    "mxfp4_mlp_wo": "general/ptq/mxfp4_mlp_weight_only-fp8_kv",
    # Mamba MOE
    "mamba_moe_fp8_conservative": "general/ptq/mamba_moe_fp8_conservative-fp8_kv",
    "mamba_moe_nvfp4_aggressive": "general/ptq/mamba_moe_nvfp4_aggressive-fp8_kv",
    "mamba_moe_nvfp4_conservative": "general/ptq/mamba_moe_nvfp4_conservative-fp8_kv",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (like OmegaConf.merge but lightweight)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_with_bases(yaml_path: Path, recipes_root: Path) -> dict[str, Any]:
    """Load a YAML file resolving __base__ inheritance.

    Implements the same __base__ merging as PR #1000's load_config():
    reads __base__ list, recursively loads each base, merges in order.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    bases = data.pop("__base__", [])
    if not bases:
        return data

    # Resolve each base file (path without .yml extension)
    merged: dict[str, Any] = {}
    for base_ref in bases:
        base_path = recipes_root / f"{base_ref}.yml"
        if not base_path.is_file():
            base_path = recipes_root / f"{base_ref}.yaml"
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config not found: {base_ref} (tried .yml and .yaml)")
        base_data = _load_yaml_with_bases(base_path, recipes_root)
        merged = _deep_merge(merged, base_data)

    # Current file overrides bases
    merged = _deep_merge(merged, data)
    return merged


def _normalize_yaml_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML-loaded config to match Python constant format.

    Converts list values (e.g., num_bits: [4, 3]) to tuples to match
    the format returned by Python *_CFG constants.
    """
    if isinstance(config, dict):
        return {k: _normalize_yaml_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        # num_bits and scale_bits are stored as tuples in Python constants
        return tuple(_normalize_yaml_config(x) for x in config)
    return config


def _load_recipe_from_yaml(recipe_dir: str, recipes_root: Path) -> dict[str, Any]:
    """Load a composed recipe directory into a preset config dict.

    A recipe directory contains:
    - model_quant.yml: main quantizer config (__base__ inheritance)
    - kv_quant.yml (optional): KV cache config (__base__ inheritance)
    - recipe.yml: metadata (recipe_type, description)

    Returns a dict matching the format of Python *_CFG constants:
    {"quant_cfg": {...}, "algorithm": "..."}
    """
    recipe_path = recipes_root / recipe_dir

    # Load model quantizer config
    model_quant_path = recipe_path / "model_quant.yml"
    if not model_quant_path.is_file():
        raise FileNotFoundError(f"model_quant.yml not found in {recipe_path}")
    config = _load_yaml_with_bases(model_quant_path, recipes_root)

    # Load KV cache config if present and merge
    kv_quant_path = recipe_path / "kv_quant.yml"
    if kv_quant_path.is_file():
        kv_config = _load_yaml_with_bases(kv_quant_path, recipes_root)
        if "quant_cfg" in kv_config:
            config.setdefault("quant_cfg", {}).update(kv_config["quant_cfg"])

    return _normalize_yaml_config(config)


def _try_load_yaml_registry() -> dict[str, dict[str, Any]] | None:
    """Attempt to load all presets from modelopt_recipes/ YAML fragments.

    Returns the complete registry dict, or None if modelopt_recipes is not available.
    """
    try:
        from importlib.resources import files

        recipes_pkg = files("modelopt_recipes")
    except (ModuleNotFoundError, TypeError):
        return None

    # Convert Traversable to Path for consistent file operations
    # importlib.resources.files() may return a Traversable that isn't a Path
    recipes_root = Path(str(recipes_pkg))
    if not recipes_root.is_dir():
        return None

    registry: dict[str, dict[str, Any]] = {}
    for preset_name, recipe_dir in _PRESET_YAML_MAP.items():
        config = _try_load_single_yaml_preset(preset_name, recipe_dir, recipes_root)
        if config is None:
            return None  # Partial load is worse than no load — fall through to next tier
        registry[preset_name] = config

    return registry


def _try_load_single_yaml_preset(
    preset_name: str, recipe_dir: str, recipes_root: Path
) -> dict[str, Any] | None:
    """Load a single preset from YAML, returning None on failure."""
    try:
        return _load_recipe_from_yaml(recipe_dir, recipes_root)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        logger.debug("Failed to load YAML preset '%s': %s", preset_name, exc)
        return None


def _try_load_python_registry() -> dict[str, dict[str, Any]] | None:
    """Attempt to load presets from Python constants (deprecated).

    Returns the registry dict, or None if the constants are not available.
    This tier will be removed when the team removes *_CFG constants.
    """
    try:
        import modelopt.torch.quantization.config as _cfg_mod
    except ModuleNotFoundError:
        return None

    # Build registry from Python constants. If any attribute is missing
    # (e.g., team has started removing constants), fall through gracefully.
    try:
        return {
            # Core formats
            "fp8": _cfg_mod.FP8_DEFAULT_CFG,
            "fp8_pc_pt": _cfg_mod.FP8_PER_CHANNEL_PER_TOKEN_CFG,
            "fp8_pb_wo": _cfg_mod.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
            "int8": _cfg_mod.INT8_DEFAULT_CFG,
            "int8_sq": _cfg_mod.INT8_SMOOTHQUANT_CFG,
            "int8_wo": _cfg_mod.INT8_WEIGHT_ONLY_CFG,
            "int4": _cfg_mod.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            "int4_awq": _cfg_mod.INT4_AWQ_CFG,
            # NVFP4 family
            "nvfp4": _cfg_mod.NVFP4_DEFAULT_CFG,
            "nvfp4_awq": _cfg_mod.NVFP4_AWQ_LITE_CFG,
            "nvfp4_awq_lite": _cfg_mod.NVFP4_AWQ_LITE_CFG,
            "nvfp4_awq_clip": _cfg_mod.NVFP4_AWQ_CLIP_CFG,
            "nvfp4_awq_full": _cfg_mod.NVFP4_AWQ_FULL_CFG,
            "nvfp4_mse": _cfg_mod.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG,
            "nvfp4_local_hessian": _cfg_mod.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
            "nvfp4_fp8_mha": _cfg_mod.NVFP4_FP8_MHA_CONFIG,
            "nvfp4_svdquant": _cfg_mod.NVFP4_SVDQUANT_DEFAULT_CFG,
            "nvfp4_mlp_only": _cfg_mod.NVFP4_MLP_ONLY_CFG,
            "nvfp4_mlp_wo": _cfg_mod.NVFP4_MLP_WEIGHT_ONLY_CFG,
            # W4A8 variants
            "w4a8_awq": _cfg_mod.W4A8_AWQ_BETA_CFG,
            "w4a8_nvfp4_fp8": _cfg_mod.W4A8_NVFP4_FP8_CFG,
            "w4a8_mxfp4_fp8": _cfg_mod.W4A8_MXFP4_FP8_CFG,
            # MX formats
            "mxfp8": _cfg_mod.MXFP8_DEFAULT_CFG,
            "mxfp6": _cfg_mod.MXFP6_DEFAULT_CFG,
            "mxfp4": _cfg_mod.MXFP4_DEFAULT_CFG,
            "mxint8": _cfg_mod.MXINT8_DEFAULT_CFG,
            "mxfp4_mlp_wo": _cfg_mod.MXFP4_MLP_WEIGHT_ONLY_CFG,
            # Mamba MOE
            "mamba_moe_fp8_aggressive": _cfg_mod.MAMBA_MOE_FP8_AGGRESSIVE_CFG,
            "mamba_moe_fp8_conservative": _cfg_mod.MAMBA_MOE_FP8_CONSERVATIVE_CFG,
            "mamba_moe_nvfp4_aggressive": _cfg_mod.MAMBA_MOE_NVFP4_AGGRESSIVE_CFG,
            "mamba_moe_nvfp4_conservative": _cfg_mod.MAMBA_MOE_NVFP4_CONSERVATIVE_CFG,
        }
    except AttributeError:
        # Some constants have been removed — this tier is no longer usable
        return None


def _load_registry() -> dict[str, dict[str, Any]]:
    """Lazily load preset configs with tiered fallback."""
    global _PRESET_REGISTRY, _PRESET_SOURCE
    if _PRESET_REGISTRY is not None:
        return _PRESET_REGISTRY

    # Tier 1: Load from modelopt_recipes/ YAML fragments (aligned with PR #1000)
    registry = _try_load_yaml_registry()
    if registry is not None:
        _PRESET_REGISTRY = registry
        _PRESET_SOURCE = "yaml"
        logger.debug("Loaded %d presets from modelopt_recipes/ YAML fragments", len(registry))
        return _PRESET_REGISTRY

    # Tier 2: Live import from Python constants (deprecated, will be removed)
    registry = _try_load_python_registry()
    if registry is not None:
        _PRESET_REGISTRY = registry
        _PRESET_SOURCE = "live"
        logger.debug("Loaded %d presets from Python constants (deprecated path)", len(registry))
        return _PRESET_REGISTRY

    # Tier 3: Bundled snapshot
    from ._bundled_presets import BUNDLED_PRESETS

    warnings.warn(
        "Neither modelopt_recipes package nor nvidia-modelopt installed. "
        "Using bundled preset snapshot. Install modelopt_recipes for YAML-based presets, "
        "or nvidia-modelopt[torch] for live presets.",
        UserWarning,
        stacklevel=2,
    )
    _PRESET_REGISTRY = BUNDLED_PRESETS
    _PRESET_SOURCE = "bundled"
    return _PRESET_REGISTRY


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of the preset config dict."""
    registry = _load_registry()
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return copy.deepcopy(registry[name])


def get_preset_source() -> str:
    """Return 'yaml', 'live', or 'bundled' indicating which tier is active."""
    _load_registry()
    return _PRESET_SOURCE


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(_load_registry().keys())
