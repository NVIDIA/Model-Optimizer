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

"""GPT-OSS specs (HF model type ``gpt_oss``)."""

from ..registry import register
from ..specs import ModelSpec, MoESpec, MoEVariant

register(
    ModelSpec(
        model_type="gpt_oss",
        min_transformers_version="4.57",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    # GPT-OSS fuses gate and up into a single gate_up_proj.
                    # transformers names the block GptOssMLP; GptOssMoE is kept for the
                    # legacy name this data was migrated from.
                    block_names=("GptOssMLP", "GptOssMoE"),
                    expert_linear_names=("gate_up_proj", "down_proj"),
                ),
            ),
        ),
    )
)
