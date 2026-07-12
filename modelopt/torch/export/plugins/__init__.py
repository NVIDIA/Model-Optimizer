# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Export package plugin."""

from typing import Any

from modelopt.torch.utils import import_plugin

with import_plugin("megatron_importer"):
    from .megatron_importer import *

from .hf_spec_export import *

with import_plugin("hf_checkpoint_utils"):
    from .hf_checkpoint_utils import *

if "sanitize_hf_config_for_deployment" not in globals():

    def sanitize_hf_config_for_deployment(config_data: dict[str, Any], model: Any) -> None:
        """No-op fallback when Hugging Face checkpoint utilities are unavailable."""
        return None


from .vllm_fakequant_hf import *

with import_plugin("vllm_fakequant_megatron"):
    from .vllm_fakequant_megatron import *
