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

Tier 1a (preferred): Use PR #1000's load_config() when available
    - Canonical OmegaConf-based __base__ resolution
    - Forward-compatible: auto-adopts when PR #1000 merges
    - Falls through gracefully if load_config is not yet available

Tier 1b: Load from modelopt_recipes/ YAML fragments with our own loader
    - Lightweight __base__ resolution (no OmegaConf dependency)
    - Used when modelopt_recipes is installed but load_config is not available

Tier 2 (fallback): Live import from modelopt.torch.quantization.config
    - Gets preset dicts from Python constants (deprecated — team removing these)
    - Both tiers are in the same repo, so at least one is always available
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from modelopt.torch.recipes.utils import load_yaml_with_bases

logger = logging.getLogger(__name__)

_PRESET_REGISTRY: dict[str, dict[str, Any]] | None = None
_PRESET_METADATA: dict[str, dict[str, str]] = {}
_PRESET_SOURCE: str = "unknown"

# Mapping from preset name to the composed recipe directory in modelopt_recipes/.
# Each entry is a directory under general/ptq/ containing model_quant.yml + kv_quant.yml.
# The model_quant.yml uses __base__ to compose atomic fragments (base + quantizer + algorithm).
PRESET_YAML_MAP: dict[str, str] = {
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
    "nvfp4_omlp_only": "general/ptq/nvfp4_omlp_only-fp8_kv",
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
    # Mamba MOE — mamba_moe_fp8_aggressive is Tier-2-only (no YAML directory in PR #1000)
    "mamba_moe_fp8_aggressive": "general/ptq/mamba_moe_fp8_aggressive-fp8_kv",
    "mamba_moe_fp8_conservative": "general/ptq/mamba_moe_fp8_conservative-fp8_kv",
    "mamba_moe_nvfp4_aggressive": "general/ptq/mamba_moe_nvfp4_aggressive-fp8_kv",
    "mamba_moe_nvfp4_conservative": "general/ptq/mamba_moe_nvfp4_conservative-fp8_kv",
}

# Presets that only exist as Python constants (no YAML directory in PR #1000).
# These are skipped during YAML loading and filled from Tier 2 instead.
_TIER2_ONLY_PRESETS: set[str] = {"mamba_moe_fp8_aggressive"}


def _get_load_config():  # pragma: no cover
    """Try to import PR #1000's load_config and verify it works.

    Returns the function or None. Unlike try_import_load_config() (which only
    checks importability), this also verifies the YAML fragments are present.
    """
    from modelopt.torch.recipes.utils import try_import_load_config

    load_config = try_import_load_config()
    if load_config is None:
        return None
    try:
        load_config("configs/ptq/base")
        return load_config
    except (ValueError, AttributeError, TypeError):
        return None


def load_recipe_from_yaml(
    recipe_dir: str, recipes_root: Path
) -> dict[str, Any]:  # pragma: no cover
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
    config = load_yaml_with_bases(model_quant_path, recipes_root)

    # Load KV cache config if present and merge
    kv_quant_path = recipe_path / "kv_quant.yml"
    if kv_quant_path.is_file():
        kv_config = load_yaml_with_bases(kv_quant_path, recipes_root)
        if "quant_cfg" in kv_config:
            config.setdefault("quant_cfg", {}).update(kv_config["quant_cfg"])

    return config


def _load_recipe_metadata(
    recipe_dir: str, recipes_root: Path
) -> dict[str, str] | None:  # pragma: no cover
    """Load recipe.yml metadata from a composed recipe directory."""
    recipe_yml = recipes_root / recipe_dir / "recipe.yml"
    if not recipe_yml.is_file():
        return None
    try:
        with open(recipe_yml) as f:
            data = yaml.safe_load(f) or {}
        return {
            "description": data.get("description", ""),
            "recipe_type": data.get("recipe_type", "ptq"),
        }
    except (yaml.YAMLError, OSError):
        return None


def _try_load_yaml_registry_via_load_config(
    load_config_fn,
) -> dict[str, dict[str, Any]] | None:  # pragma: no cover
    """Tier 1a: Load presets using PR #1000's load_config (canonical OmegaConf merge)."""
    registry: dict[str, dict[str, Any]] = {}
    for preset_name, recipe_dir in PRESET_YAML_MAP.items():
        if preset_name in _TIER2_ONLY_PRESETS:
            continue
        try:
            config = load_config_fn(f"{recipe_dir}/model_quant")
        except (ValueError, FileNotFoundError):
            logger.debug("load_config failed for preset '%s' — aborting Tier 1a", preset_name)
            return None

        # Load KV quant if present
        try:
            kv_config = load_config_fn(f"{recipe_dir}/kv_quant")
            if "quant_cfg" in kv_config:
                config.setdefault("quant_cfg", {}).update(kv_config["quant_cfg"])
        except (ValueError, FileNotFoundError):
            pass

        # Load recipe.yml metadata
        try:
            meta = load_config_fn(f"{recipe_dir}/recipe")
            _PRESET_METADATA[preset_name] = {
                "description": meta.get("description", ""),
                "recipe_type": meta.get("recipe_type", "ptq"),
            }
        except (ValueError, FileNotFoundError):
            pass

        registry[preset_name] = config

    return registry


def _try_load_yaml_registry() -> dict[str, dict[str, Any]] | None:  # pragma: no cover
    """Attempt to load presets from YAML fragments.

    Tries PR #1000's load_config first (Tier 1a), then our own loader (Tier 1b).
    Returns the complete registry dict, or None if neither approach works.
    """
    # Tier 1a: Use PR #1000's load_config (canonical, OmegaConf merge)
    load_config_fn = _get_load_config()
    if load_config_fn is not None:
        registry = _try_load_yaml_registry_via_load_config(load_config_fn)
        if registry is not None:
            return registry

    # Tier 1b: Our lightweight YAML loader
    try:
        from importlib.resources import files

        recipes_pkg = files("modelopt_recipes")
    except (ModuleNotFoundError, TypeError):
        return None

    recipes_root = Path(str(recipes_pkg))
    if not recipes_root.is_dir():
        return None

    yaml_registry: dict[str, dict[str, Any]] = {}
    for preset_name, recipe_dir in PRESET_YAML_MAP.items():
        if preset_name in _TIER2_ONLY_PRESETS:
            continue
        config = _try_load_single_yaml_preset(preset_name, recipe_dir, recipes_root)
        if config is None:
            return None  # Partial load is worse than no load — fall through to next tier

        # Load metadata
        meta = _load_recipe_metadata(recipe_dir, recipes_root)
        if meta:
            _PRESET_METADATA[preset_name] = meta

        yaml_registry[preset_name] = config

    return yaml_registry


def _try_load_single_yaml_preset(
    preset_name: str, recipe_dir: str, recipes_root: Path
) -> dict[str, Any] | None:  # pragma: no cover
    """Load a single preset from YAML, returning None on failure."""
    try:
        return load_recipe_from_yaml(recipe_dir, recipes_root)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        logger.debug("Failed to load YAML preset '%s': %s", preset_name, exc)
        return None


def _fill_tier2_only_presets(registry: dict[str, dict[str, Any]]) -> None:  # pragma: no cover
    """Load Tier-2-only presets from Python constants."""
    try:
        import modelopt.torch.quantization.config as _cfg_mod
    except (ImportError, ModuleNotFoundError):
        return

    _tier2_attr_map: dict[str, str] = {
        "mamba_moe_fp8_aggressive": "MAMBA_MOE_FP8_AGGRESSIVE_CFG",
    }
    for name, attr in _tier2_attr_map.items():
        if name not in registry:
            try:
                registry[name] = getattr(_cfg_mod, attr)
            except AttributeError:
                logger.debug("Tier-2-only preset '%s' not found as %s", name, attr)


def _try_load_python_registry() -> dict[str, dict[str, Any]] | None:  # pragma: no cover
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
            "nvfp4_omlp_only": _cfg_mod.NVFP4_OMLP_ONLY_CFG,
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


def _load_registry() -> dict[str, dict[str, Any]]:  # pragma: no cover
    """Lazily load preset configs with tiered fallback."""
    global _PRESET_REGISTRY, _PRESET_SOURCE
    if _PRESET_REGISTRY is not None:
        return _PRESET_REGISTRY

    # Tier 1: Load from YAML fragments (1a: load_config, 1b: our own loader)
    registry = _try_load_yaml_registry()
    if registry is not None:
        # Fill in Tier-2-only presets from Python constants
        _fill_tier2_only_presets(registry)
        _PRESET_REGISTRY = registry
        _PRESET_SOURCE = "yaml"
        logger.debug("Loaded %d presets from YAML fragments", len(registry))
        return _PRESET_REGISTRY

    # Tier 2: Live import from Python constants (deprecated, will be removed)
    registry = _try_load_python_registry()
    if registry is not None:
        _PRESET_REGISTRY = registry
        _PRESET_SOURCE = "live"
        logger.debug("Loaded %d presets from Python constants (deprecated path)", len(registry))
        return _PRESET_REGISTRY

    raise RuntimeError(
        "Cannot load preset registry. Neither modelopt_recipes YAML fragments "
        "nor modelopt.torch.quantization.config Python constants are available. "
        "Run 'pip install -e .' from the Model-Optimizer repo root."
    )


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of the preset config dict."""
    registry = _load_registry()
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return copy.deepcopy(registry[name])


def get_preset_info(name: str) -> dict[str, str]:
    """Return metadata for a preset (description, recipe_type).

    Metadata is loaded from recipe.yml in the composed recipe directory.
    Returns empty dict if no metadata is available.
    """
    _load_registry()  # Ensure metadata is loaded
    return _PRESET_METADATA.get(name, {})


def get_preset_source() -> str:
    """Return 'yaml' or 'live' indicating which tier is active."""
    _load_registry()
    return _PRESET_SOURCE


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(_load_registry().keys())
