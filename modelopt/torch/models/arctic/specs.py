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

"""Snowflake Arctic specs (trust-remote-code model type ``arctic``)."""

from ..registry import register
from ..specs import ModelSpec, MoESpec, MoEVariant

register(
    ModelSpec(
        model_type="arctic",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    block_names=("ArcticMoE",),
                    # ArcticMLP experts use Mixtral-style w1/w2/w3 naming.
                    expert_linear_names=("w1", "w2", "w3"),
                    gate_up_pair=("w1", "w3"),
                ),
            ),
        ),
    )
)
