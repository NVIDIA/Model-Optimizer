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

"""DeepSeek-V4 specs (HF model type ``deepseek_v4``)."""

from ..registry import register
from ..specs import ModelSpec, MoESpec, MoEVariant

# DeepseekV4Experts is fused from the start: a single module holding 3-D gate_up_proj
# and down_proj parameters rather than an iterable of per-expert modules. So there is no
# (gate, up) pair left for a serving engine to fuse, and the grouped-export path
# (get_experts_list) does not apply -- both fields keep their defaults.
register(
    ModelSpec(
        model_type="deepseek_v4",
        min_transformers_version="5.8",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    block_names=("DeepseekV4SparseMoeBlock",),
                    expert_linear_names=("gate_up_proj", "down_proj"),
                ),
            ),
        ),
    )
)
