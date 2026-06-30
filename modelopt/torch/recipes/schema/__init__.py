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

"""Recipe schema — models, formats, presets, and resolution."""

from .formats import FORMAT_REGISTRY, KV_FORMAT_REGISTRY
from .models import RecipeConfig
from .presets import (
    PRESET_YAML_MAP,
    get_preset,
    get_preset_info,
    get_preset_source,
    list_presets,
    load_recipe_from_yaml,
)
from .resolver import resolve_recipe

__all__ = [
    "FORMAT_REGISTRY",
    "KV_FORMAT_REGISTRY",
    "PRESET_YAML_MAP",
    "RecipeConfig",
    "get_preset",
    "get_preset_info",
    "get_preset_source",
    "list_presets",
    "load_recipe_from_yaml",
    "resolve_recipe",
]
