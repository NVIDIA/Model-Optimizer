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

from ..specs import EPSpec, ModelSpec, MoESpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="gpt_oss",
        min_transformers_version="5.5",
        ep_spec=EPSpec(
            unsupported=(
                "GPT-OSS stores its experts transposed, (num_experts, in, out), with gate and up "
                "interleaved in gate_up_proj, and the expert-parallel export cannot write that "
                "layout."
            ),
        ),
        moe_spec=MoESpec(
            # GPT-OSS fuses gate and up into a single gate_up_proj.
            # transformers names the block GptOssMLP; GptOssMoE is kept for the
            # legacy name this data was migrated from.
            block_names=("GptOssMLP", "GptOssMoE"),
            expert_linear_names=("gate_up_proj", "down_proj"),
            fused_expert_names=True,
        ),
    )
)
