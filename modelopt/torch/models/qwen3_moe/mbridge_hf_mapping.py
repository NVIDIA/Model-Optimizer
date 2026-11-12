# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom mapping from Qwen3-MoE Hugging Face models to Megatron Core models.

Reuses the dense Qwen3 rules.
"""

from modelopt.torch.export.plugins.mcore_custom import register_mcore_hf_mapping

from ..qwen3.mbridge_hf_mapping import qwen3_causal_lm_export, qwen3_causal_lm_import

__all__: list[str] = []

register_mcore_hf_mapping(
    "Qwen3MoeForCausalLM",
    export_mapping=qwen3_causal_lm_export,
    import_mapping=qwen3_causal_lm_import,
)
