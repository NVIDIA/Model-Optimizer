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

"""DeepSeek-MoE specs (trust-remote-code model type ``deepseek``).

Matches the remote-code ``DeepseekMoE`` block, not the HF-native ``deepseek_v3``
classes.
"""

from ..registry import register
from ..specs import ModelSpec, MoESpec, MoEVariant

# has_iterable_experts stays False until the grouped export path is validated on
# this model.
register(
    ModelSpec(
        model_type="deepseek",
        modeling_source="remote_code",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    block_names=("DeepseekMoE",),
                    expert_linear_names=("gate_proj", "down_proj", "up_proj"),
                    gate_up_pair=("gate_proj", "up_proj"),
                ),
            ),
        ),
    )
)
