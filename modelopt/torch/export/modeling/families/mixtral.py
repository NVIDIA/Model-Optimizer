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

"""Mixtral family export specs."""

from ..base import ModelSpec
from ..registry import register

# Mixtral with iterable experts uses w1/w2/w3. Fused experts (transformers 5.0+) are
# detected from their per-expert quantizer attributes and need no naming override here.
register(
    ModelSpec(
        name="mixtral",
        moe_block_names=("MixtralSparseMoeBlock",),
        expert_linear_names=("w1", "w2", "w3"),
        has_iterable_experts=True,
    )
)

# Older transformers naming for Mixtral.
register(
    ModelSpec(
        name="mixtral_mcore",
        moe_block_names=("MixtralMoeSparseMoeBlock",),
        expert_linear_names=("linear_fc1", "linear_fc2"),
    )
)
