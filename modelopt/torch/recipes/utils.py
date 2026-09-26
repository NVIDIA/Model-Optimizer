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

"""Shared utilities for the recipe system."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (like OmegaConf.merge but lightweight)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_with_bases(yaml_path: Path, recipes_root: Path) -> dict[str, Any]:
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
        base_data = load_yaml_with_bases(base_path, recipes_root)
        merged = deep_merge(merged, base_data)

    # Current file overrides bases
    merged = deep_merge(merged, data)
    return merged


def make_serializable(obj: Any) -> Any:
    """Convert tuples and other non-JSON-safe types for serialization/display.

    Recursively converts dicts (with any key type), lists, and tuples into
    JSON-compatible structures. Dict keys are stringified.
    """
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def try_import_load_config():
    """Try to import PR #1000's load_config function.

    Returns the function if available, None otherwise. This is the forward-compatible
    import point for load_config() from modelopt.torch.opt.config.
    """
    try:
        from modelopt.torch.opt.config import load_config  # type: ignore[attr-defined]

        return load_config
    except (ImportError, ModuleNotFoundError):
        return None
