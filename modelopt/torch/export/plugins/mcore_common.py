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

"""Custom mapping from Megatron Core models to their Hugging Face counter part."""

import importlib

from .mcore_custom import (
    all_mcore_hf_export_mapping,
    all_mcore_hf_import_mapping,
    all_mcore_hf_vision_passthrough_mapping,
)

__all__ = [
    "all_mcore_hf_export_mapping",
    "all_mcore_hf_import_mapping",
    "all_mcore_hf_vision_passthrough_mapping",
]

# The mappings live with their model in ``modelopt/torch/models/<model_type>/mcore.py`` and register
# themselves on import; add a new model type here.
MCORE_MODEL_TYPES = (
    "deepseek_v2",
    "deepseek_v3",
    "glm5_next",
    "glm_moe_dsa",
    "gpt_oss",
    "llama",
    "llama4",
    "nemotron",
    "nemotron_h",
    "nemotron_h_omni",
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_vl",
)

for _model_type in MCORE_MODEL_TYPES:
    importlib.import_module(f"modelopt.torch.models.{_model_type}.mcore")
