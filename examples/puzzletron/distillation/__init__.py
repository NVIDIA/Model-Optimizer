# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Heterogeneous AnyModel / Puzzletron models in Megatron Bridge — v2 (clean rewrite).

Public API
----------
From ``block_config_utils``:
    load_block_configs          — Load per-layer block_configs from an HF config.
    MCoreLayerOverrides         — Per-layer TransformerConfig overrides container.
    block_config_to_mcore_overrides — Translate one BlockConfig → MCoreLayerOverrides.
    get_overrides_for_layer     — Look up overrides by 1-based global layer number.

From ``layer_patchers``:
    mbridge_patcher             — Context manager that patches MCore layer construction.
    NoOpWithBias                — Correct no-op replacement for attention/MLP submodules.

From ``provider_patch``:
    apply_patch                 — Patch ModelProviderMixin.provide at class level.
    remove_patch                — Restore ModelProviderMixin.provide.
    set_provider_block_configs  — Attach block_configs to a provider instance.
"""

from block_config_utils import (
    MCoreLayerOverrides,
    block_config_to_mcore_overrides,
    get_overrides_for_layer,
    load_block_configs,
)
from layer_patchers import NoOpWithBias, mbridge_patcher
from provider_patch import apply_patch, remove_patch, set_provider_block_configs

__all__ = [
    # block_config_utils
    "MCoreLayerOverrides",
    "block_config_to_mcore_overrides",
    "get_overrides_for_layer",
    "load_block_configs",
    # layer_patchers
    "mbridge_patcher",
    "NoOpWithBias",
    # provider_patch
    "apply_patch",
    "remove_patch",
    "set_provider_block_configs",
]