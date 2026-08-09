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

"""Gemma4 specs (HF model type ``gemma4``).

Gemma4RMSNorm is intentionally absent from ``weight_plus_one_norm_names`` until the
+1 handling is validated on Gemma4.
"""

from ..registry import register
from ..specs import ModelSpec, MoEVariant

# Gemma4 MoE experts are unfused into per-expert nn.Linear layers. The MoE block lives
# in the text model, so ``gemma4_text`` reuses this exact layout -- see
# ``gemma4_text/export.py``, which imports it rather than restating it.
GEMMA4_MOE_VARIANTS = (
    MoEVariant(
        block_names=("Gemma4TextDecoderLayer",),
        expert_linear_names=("gate_proj", "down_proj", "up_proj"),
        gate_up_pair=("gate_proj", "up_proj"),
        has_iterable_experts=True,
    ),
)

register(ModelSpec(model_type="gemma4", moe_variants=GEMMA4_MOE_VARIANTS))
