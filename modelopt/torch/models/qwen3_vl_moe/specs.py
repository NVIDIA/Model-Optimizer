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

"""Qwen3-VL-MoE specs (HF model type ``qwen3_vl_moe``)."""

from ..specs import ModelSpec, MoESpec, register

__all__: list[str] = []

# Qwen3VLMoeTextExperts is fused on every supported transformers: 3-D gate_up_proj and
# down_proj parameters (transposed before 5.12, standard layout from 5.12 on). Before 5.12
# the PTQ wrapper unrolls them into gate_proj/up_proj/down_proj ModuleLists (see
# modeling_ptq.py); that rewrite is not iterable per expert, so no export path groups it.
register(
    ModelSpec(
        model_type="qwen3_vl_moe",
        min_transformers_version="4.57",
        moe_spec=MoESpec(
            block_names=("Qwen3VLMoeTextSparseMoeBlock",),
            expert_linear_names=("gate_up_proj", "down_proj"),
            fused_expert_names=True,
        ),
    )
)
