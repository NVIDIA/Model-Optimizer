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

"""Custom mapping from Qwen3-VL Hugging Face models to Megatron Core models.

Qwen3-VL nests the language model under ``model.language_model.`` while ``lm_head`` stays at the
root, so the mappings are derived from Qwen3's. The visual encoder is copied verbatim from HF via
``QWEN3VL_VISION_PREFIXES`` rather than mapped. ``Qwen3VLMoeForConditionalGeneration`` is not
supported: its 3-D fused expert weights cannot reuse the dense Qwen3 rules.
"""

from .mcore_custom import with_language_model_prefix
from .mcore_qwen import qwen3_causal_lm_export, qwen3_causal_lm_import

# Vision-tower weights copied straight from the HF checkpoint (never quantized).
QWEN3VL_VISION_PREFIXES = ("model.visual.",)

qwen3vl_causal_lm_import = with_language_model_prefix(qwen3_causal_lm_import)
qwen3vl_causal_lm_export = with_language_model_prefix(qwen3_causal_lm_export)
