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

"""Llama4 specs (HF model type ``llama4``)."""

from ..specs import ModelSpec, MoESpec, register

__all__: list[str] = []

# Llama4TextExperts is fused: one module holding 3-D gate_up_proj and down_proj
# parameters, run through torch.bmm. There is no (gate, up) pair left to fuse, and the
# grouped-export path (get_experts_list) does not apply. The block also carries a
# shared_expert MLP, which is an ordinary dense MLP, not part of this layout.
register(
    ModelSpec(
        model_type="llama4",
        min_transformers_version="4.57",
        moe_spec=MoESpec(
            block_names=("Llama4TextMoe",),
            expert_linear_names=("gate_up_proj", "down_proj"),
            fused_expert_names=True,
        ),
    )
)
