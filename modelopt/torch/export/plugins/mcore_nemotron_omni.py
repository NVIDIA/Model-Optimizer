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

"""Custom mapping from the new model (``NemotronH_Omni_Reasoning_V3``) to Megatron Core models.

The new model nests the whole Nemotron-H language model, ``lm_head`` included, under
``language_model.``, so the mapping is derived from Nemotron-H's (MTP included). The vision tower
and projector are copied verbatim from HF via ``NEMOTRON_H_OMNI_VISION_PREFIXES``, not mapped.
"""

from .mcore_custom import with_language_model_prefix
from .mcore_nemotron import nemotron_h_causal_lm_export

# Weights copied straight from the HF checkpoint (never quantized).
NEMOTRON_H_OMNI_VISION_PREFIXES = ("vision_model.", "vision_projector.", "mlp1.")

nemotron_h_omni_causal_lm_export = with_language_model_prefix(
    nemotron_h_causal_lm_export, old_prefix="", new_prefix="language_model."
)
