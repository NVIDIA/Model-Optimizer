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

"""DeepSeek-V3 specs (HF model type ``deepseek_v3``).

Covers the transformers-native V3 family (V3, R1, V3.1). The remote-code
``DeepseekMoE`` block of DeepSeek-MoE is a separate spec under ``deepseek``.
"""

from ..specs import ModelSpec, MoESpec, MoEVariant, register

# DeepseekV3MoE is invisible to the generic MoE detection: its class name does not end
# in "SparseMoeBlock" and it calls its router ``gate``, so neither the name test nor the
# structural router+experts test in is_moe matches. block_names below is what puts it on
# the MoE path at all.
#
# The experts container changed upstream. transformers 4.57 builds an nn.ModuleList of
# DeepseekV3MLP, which is the naming recorded here; 5.x replaces it with a fused, 3-D
# DeepseekV3Experts holding gate_up_proj/down_proj. The fused layout resolves through
# the structural first-projection check in get_expert_linear_names and never consults
# this naming -- the same split Mixtral has across the same releases.
register(
    ModelSpec(
        model_type="deepseek_v3",
        min_transformers_version="4.57",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    block_names=("DeepseekV3MoE",),
                    expert_linear_names=("gate_proj", "down_proj", "up_proj"),
                    gate_up_pair=("gate_proj", "up_proj"),
                    has_iterable_experts=True,
                ),
            ),
        ),
    )
)
