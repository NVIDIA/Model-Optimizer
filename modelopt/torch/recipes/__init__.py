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

"""Recipe system for NVIDIA Model Optimizer.

Usage:
    from modelopt.torch.recipes import load_recipe

    # Load a recipe YAML file
    result = load_recipe("path/to/recipe.yaml")

    # For quantization recipes:
    config = result["quantize_config"]  # dict for mtq.quantize()
    model = mtq.quantize(model, config, forward_loop=forward_loop)

    # For auto-quantize recipes:
    kwargs = result["auto_quantize_kwargs"]  # kwargs for mtq.auto_quantize()
    model, state = mtq.auto_quantize(model, **kwargs, data_loader=..., ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import (
    FORMAT_REGISTRY,
    KV_FORMAT_REGISTRY,
    RecipeConfig,
    get_preset,
    get_preset_info,
    get_preset_source,
    list_presets,
    resolve_recipe,
)


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Load a YAML recipe and resolve it to mtq-compatible config dicts.

    Args:
        path: Path to the recipe YAML file.

    Returns:
        A dict with keys depending on the recipe type:
        - "quantize_config": config dict for mtq.quantize()
        - "auto_quantize_kwargs": kwargs dict for mtq.auto_quantize()
        - "calibration": calibration params dict (if specified)
        - "export": export params dict (if specified)
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    recipe = RecipeConfig.model_validate(raw)
    return resolve_recipe(recipe)


__all__ = [
    "FORMAT_REGISTRY",
    "KV_FORMAT_REGISTRY",
    "RecipeConfig",
    "get_preset",
    "get_preset_info",
    "get_preset_source",
    "list_presets",
    "load_recipe",
    "resolve_recipe",
]
