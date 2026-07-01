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

"""Megatron-Bridge integration for Puzzletron AnyModel checkpoints.

This package provides everything needed to run heterogeneous (AnyModel /
Puzzletron) checkpoints through Megatron-Bridge:

- Bridge adapters (``base``, ``llama``, ``qwen3``, ``gpt_oss``) that register
  ``block_configs``-aware bridges with Megatron-Bridge (import side effect).
- The layer/provider patchers (``layer_patchers``, ``provider_patch``) and the
  per-layer ``block_configs`` translation (``block_config_utils``) that inject
  per-layer ``TransformerConfig`` overrides during model construction.
- Reusable distillation helpers (``distill_patches``).

The patcher / translation modules import Megatron-Core lazily, so they remain
importable without Megatron-Bridge installed. The bridge adapters and the
distillation helpers require Megatron-Bridge at import time and are therefore
guarded so a missing Megatron-Bridge does not break the rest of the package.
"""

from modelopt.torch.utils import import_plugin

# Pure logic: safe to import without Megatron-Bridge (Megatron-Core is imported
# lazily inside the patcher functions / context managers).
from .block_config_utils import *
from .layer_patchers import *
from .provider_patch import *

# Bridge adapters + distillation helpers require megatron.bridge at import time.
# Guard so the pure patcher API above stays importable when Megatron-Bridge is
# unavailable.
with import_plugin("megatron.bridge"):
    from .base import *
    from .distill_patches import *
    from .llama import *
    from .qwen3 import *

# The GPT-OSS bridge additionally requires a Megatron-Bridge build that ships
# ``GPTOSSProvider``. Guard it separately so its absence does not prevent the
# other bridge adapters above from registering.
with import_plugin("megatron.bridge gpt_oss"):
    from .gpt_oss import *
