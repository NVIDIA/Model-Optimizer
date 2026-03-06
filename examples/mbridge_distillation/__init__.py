# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Heterogeneous models in Megatron Bridge via patching (model-opt)."""

from .block_config_to_mcore import (
    MCoreLayerOverrides,
    block_config_to_mcore_overrides,
    get_overrides_for_layer,
)
from .mbridge_gpt_patcher import mbridge_gpt_patcher
from .mbridge_mamba_patcher import mbridge_mamba_patcher
from .patch_mbridge_provider import (
    apply_patch,
    load_block_configs,
    remove_patch,
    set_provider_block_configs,
)

__all__ = [
    "MCoreLayerOverrides",
    "block_config_to_mcore_overrides",
    "get_overrides_for_layer",
    "mbridge_gpt_patcher",
    "mbridge_mamba_patcher",
    "apply_patch",
    "remove_patch",
    "load_block_configs",
    "set_provider_block_configs",
]
