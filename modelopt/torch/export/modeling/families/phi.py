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

"""Phi family export specs."""

from ..base import ModelSpec
from ..registry import register

register(
    ModelSpec(
        name="phi3",
        decoder_types=("phi3",),
        forced_activation="swiglu",
    )
)

# Phi3Small and the related TLGv4 MLP treat up_proj as the fc projection (not gate).
register(
    ModelSpec(
        name="phi3_small",
        mlp_block_names=("Phi3SmallMLP", "TLGv4MLP"),
        mlp_keyword_roles={"up_proj": "fc"},
    )
)
