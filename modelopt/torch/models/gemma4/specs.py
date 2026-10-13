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

from ..specs import ExportSpec, ModelSpec, MoESpec, register

__all__: list[str] = []

# Gemma4 MoE experts are unfused into per-expert nn.Linear layers. The MoE block lives in
# the text model, so ``gemma4_text`` reuses this exact layout rather than restating it --
# the two must not drift apart.
#
# Private despite the cross-directory import: ``gemma4_text`` is a sub-model type of this
# same family, which transformers keeps in one package (its
# SPECIAL_MODEL_TYPE_TO_MODULE_NAME maps ``gemma4_text`` -> ``gemma4``). The directories
# are split here because the registry is keyed on ``config.model_type`` and there are two
# of them, so this is an intra-family detail, not a public API.
_GEMMA4_MOE_SPEC = MoESpec(
    block_names=("Gemma4TextDecoderLayer",),
    expert_linear_names=("gate_proj", "down_proj", "up_proj"),
    gate_up_pair=("gate_proj", "up_proj"),
)

register(
    ModelSpec(
        model_type="gemma4",
        min_transformers_version="5.5",
        export_spec=ExportSpec(grouped_expert_export=True),
        moe_spec=_GEMMA4_MOE_SPEC,
    )
)
