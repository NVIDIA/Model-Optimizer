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

"""Mixtral specs (HF model type ``mixtral``)."""

from ..specs import ExportSpec, ModelSpec, MoELayout, MoESpec, register

# Mixtral with per-expert experts uses w1/w2/w3. Fused experts (transformers 5.0+) are
# detected from their per-expert quantizer attributes and need no naming override here.
register(
    ModelSpec(
        model_type="mixtral",
        min_transformers_version="4.57",
        export_spec=ExportSpec(grouped_expert_export=True),
        moe_spec=MoESpec(
            moe_layouts=(
                MoELayout(
                    block_names=("MixtralSparseMoeBlock",),
                    expert_linear_names=("w1", "w2", "w3"),
                    # w1 = gate, w3 = up, w2 = down (Mixtral convention).
                    gate_up_pair=("w1", "w3"),
                ),
            ),
        ),
    )
)
