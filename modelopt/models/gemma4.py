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

"""Gemma4 specs (HF model type ``gemma4``)."""

from .registry import register
from .specs import ModelSpec, MoEVariant

# Gemma4 MoE experts are unfused into per-expert nn.Linear layers. The MoE block
# lives in the text model, so the same layout is registered for both the VLM root
# type and the text type (a text-only checkpoint's root model_type is
# ``gemma4_text``, following the gemma3 precedent).
_GEMMA4_MOE_VARIANTS = (
    MoEVariant(
        block_names=("Gemma4TextDecoderLayer",),
        expert_linear_names=("gate_proj", "down_proj", "up_proj"),
        gate_up_pair=("gate_proj", "up_proj"),
        has_iterable_experts=True,
    ),
)

register(ModelSpec(model_type="gemma4", moe_variants=_GEMMA4_MOE_VARIANTS))
register(ModelSpec(model_type="gemma4_text", moe_variants=_GEMMA4_MOE_VARIANTS))
