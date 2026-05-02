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

"""Format registry: maps human-readable format names to quantizer attribute kwargs.

Each entry has separate "weight" and "activation" defaults since they sometimes differ
(e.g., int8 weights use axis=0, activations use axis=None).

When PR #1000's load_config() is available, registries are loaded from YAML fragments
for automatic forward compatibility. Otherwise, falls back to inline definitions.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mapping from our format names to PR #1000's YAML fragment paths (without extension).
_FORMAT_YAML_MAP: dict[str, str] = {
    "fp8": "configs/ptq/w8a8_fp8_fp8",
    "nvfp4": "configs/ptq/w4a4_nvfp4_nvfp4",
    "int8": "configs/ptq/w8a8_int8_per_channel_int8",
    "int4": "configs/ptq/w4_int4_blockwise",
    "mxfp8": "configs/ptq/w8a8_mxfp8_mxfp8",
    "mxfp6": "configs/ptq/w6a6_mxfp6_mxfp6",
    "mxfp4": "configs/ptq/w4a4_mxfp4_mxfp4",
}

_KV_FORMAT_YAML_MAP: dict[str, str] = {
    "fp8": "configs/ptq/kv_fp8",
    "nvfp4": "configs/ptq/kv_nvfp4",
    "fp8_affine": "configs/ptq/kv_fp8_affine",
    "nvfp4_affine": "configs/ptq/kv_nvfp4_affine",
    "nvfp4_rotate": "configs/ptq/kv_nvfp4_rotate",
}

# Fallback values when PR #1000's load_config is not available.
# Uses lists (not tuples) to match PR #1000's OmegaConf output convention.
_FALLBACK_FORMAT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "fp8": {
        "weight": {"num_bits": [4, 3], "axis": None},
        "activation": {"num_bits": [4, 3], "axis": None},
    },
    "nvfp4": {
        "weight": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": [4, 3]},
            "axis": None,
            "enable": True,
        },
        "activation": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": [4, 3]},
            "axis": None,
            "enable": True,
        },
    },
    "int8": {
        "weight": {"num_bits": 8, "axis": 0},
        "activation": {"num_bits": 8, "axis": None},
    },
    "int4": {
        "weight": {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
        "activation": {"enable": False},
    },
    "mxfp8": {
        "weight": {
            "num_bits": [4, 3],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": [8, 0]},
            "enable": True,
        },
        "activation": {
            "num_bits": [4, 3],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": [8, 0]},
            "enable": True,
        },
    },
    "mxfp6": {
        "weight": {
            "num_bits": [3, 2],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": [8, 0]},
            "enable": True,
        },
        "activation": {
            "num_bits": [3, 2],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": [8, 0]},
            "enable": True,
        },
    },
    "mxfp4": {
        "weight": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": [8, 0]},
            "enable": True,
        },
        "activation": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": [8, 0]},
            "enable": True,
        },
    },
}

_FALLBACK_KV_FORMAT_REGISTRY: dict[str, dict[str, Any]] = {
    "fp8": {
        "*[kv]_bmm_quantizer": {"num_bits": [4, 3], "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "nvfp4": {
        "*[kv]_bmm_quantizer": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": [4, 3]},
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
    "fp8_affine": {
        "*[kv]_bmm_quantizer": {
            "num_bits": [4, 3],
            "axis": None,
            "enable": True,
            "bias": {-2: None, -4: None, "type": "static"},
        },
        "default": {"enable": False},
    },
    "nvfp4_affine": {
        "*[kv]_bmm_quantizer": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": [4, 3]},
            "axis": None,
            "enable": True,
            "bias": {-2: None, -4: None, "type": "static"},
        },
        "default": {"enable": False},
    },
    "nvfp4_rotate": {
        "*q_bmm_quantizer": {"enable": False, "rotate": True},
        "*k_bmm_quantizer": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": [4, 3]},
            "axis": None,
            "enable": True,
            "rotate": True,
        },
        "*v_bmm_quantizer": {
            "num_bits": [2, 1],
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": [4, 3]},
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
}


def _try_load_format_registry_from_yaml() -> dict[str, dict[str, dict[str, Any]]] | None:
    """Try to load FORMAT_REGISTRY from PR #1000's YAML fragments via load_config."""
    from modelopt.torch.recipes.utils import try_import_load_config

    load_config = try_import_load_config()
    if load_config is None:
        return None

    try:
        registry: dict[str, dict[str, dict[str, Any]]] = {}
        for name, yaml_path in _FORMAT_YAML_MAP.items():
            cfg = load_config(yaml_path)
            qcfg = cfg.get("quant_cfg", {})
            registry[name] = {
                "weight": qcfg.get("*weight_quantizer", {}),
                "activation": qcfg.get("*input_quantizer", {}),
            }
        logger.debug("Loaded FORMAT_REGISTRY from %d YAML fragments", len(registry))
        return registry
    except (ValueError, KeyError, TypeError) as exc:
        logger.debug("Failed to load FORMAT_REGISTRY from YAML: %s", exc)
        return None


def _try_load_kv_format_registry_from_yaml() -> dict[str, dict[str, Any]] | None:
    """Try to load KV_FORMAT_REGISTRY from PR #1000's YAML fragments via load_config."""
    from modelopt.torch.recipes.utils import try_import_load_config

    load_config = try_import_load_config()
    if load_config is None:
        return None

    try:
        registry: dict[str, dict[str, Any]] = {}
        for name, yaml_path in _KV_FORMAT_YAML_MAP.items():
            cfg = load_config(yaml_path)
            registry[name] = cfg.get("quant_cfg", cfg)
        logger.debug("Loaded KV_FORMAT_REGISTRY from %d YAML fragments", len(registry))
        return registry
    except (ValueError, KeyError, TypeError) as exc:
        logger.debug("Failed to load KV_FORMAT_REGISTRY from YAML: %s", exc)
        return None


def _build_format_registry() -> dict[str, dict[str, dict[str, Any]]]:
    """Build FORMAT_REGISTRY: prefer YAML fragments, fall back to inline."""
    registry = _try_load_format_registry_from_yaml()
    if registry is not None:
        return registry
    return copy.deepcopy(_FALLBACK_FORMAT_REGISTRY)


def _build_kv_format_registry() -> dict[str, dict[str, Any]]:
    """Build KV_FORMAT_REGISTRY: prefer YAML fragments, fall back to inline."""
    registry = _try_load_kv_format_registry_from_yaml()
    if registry is not None:
        return registry
    return copy.deepcopy(_FALLBACK_KV_FORMAT_REGISTRY)


# Module-level registries — loaded at import time with graceful fallback.
FORMAT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = _build_format_registry()
KV_FORMAT_REGISTRY: dict[str, dict[str, Any]] = _build_kv_format_registry()


def get_format(name: str) -> dict[str, dict[str, Any]]:
    """Look up a format by name. Raises KeyError if not found."""
    if name not in FORMAT_REGISTRY:
        available = sorted(FORMAT_REGISTRY.keys())
        raise KeyError(f"Unknown format '{name}'. Available: {available}")
    return FORMAT_REGISTRY[name]


def get_kv_format(name: str) -> dict[str, Any]:
    """Look up a KV cache format by name. Raises KeyError if not found."""
    if name not in KV_FORMAT_REGISTRY:
        available = sorted(KV_FORMAT_REGISTRY.keys())
        raise KeyError(f"Unknown KV cache format '{name}'. Available: {available}")
    return KV_FORMAT_REGISTRY[name]
