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

"""Qwen3.5-MoE specs (HF model type ``qwen3_5_moe``)."""

from .registry import register
from .specs import ModelSpec, MoEVariant

# has_iterable_experts stays False to preserve pre-refactor behavior: the legacy
# get_experts_list keyed off ``type(root_model).__name__.lower()``, and
# "qwen3_5moeforcausallm" matched none of its qwen substrings (the "_5" breaks
# "qwen3moeforcausallm"), so Qwen3.5-MoE raised NotImplementedError there. The
# layout looks identical to qwen3_moe, so flipping this to True is likely correct
# -- but that is a behavior change and belongs in its own PR with validation.
register(
    ModelSpec(
        model_type="qwen3_5_moe",
        moe_variants=(
            MoEVariant(
                block_names=("Qwen3_5MoeSparseMoeBlock",),
                expert_linear_names=("gate_proj", "down_proj", "up_proj"),
                gate_up_pair=("gate_proj", "up_proj"),
            ),
        ),
    )
)
