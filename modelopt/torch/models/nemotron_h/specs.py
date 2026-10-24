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

"""Nemotron-H specs (HF model type ``nemotron_h``)."""

from ..specs import EPSpec, ExportSpec, ModelSpec, MoESpec, register

__all__: list[str] = []

# Ungated experts, so ``up_proj``/``down_proj`` rather than a gate/up pair. The gate emits logits and
# the MoE block runs its own group-limited top-k, so the plan has no router entry and the experts
# receive global expert ids -- which the expert-parallel all-to-all experts forward routes on. The
# loader maps ``nemotron_experts_ep``, a name no transformers style carries, to that forward.
BASE_MODEL_EP_PLAN = {
    "layers.*.mixer.experts.up_proj": "grouped_gemm",
    "layers.*.mixer.experts.down_proj": "grouped_gemm",
    "layers.*.mixer.experts": "nemotron_experts_ep",
}

register(
    ModelSpec(
        model_type="nemotron_h",
        min_transformers_version="5.5",
        export_spec=ExportSpec(grouped_expert_export=True),
        ep_spec=EPSpec(base_model_ep_plan=BASE_MODEL_EP_PLAN),
        moe_spec=MoESpec(
            # NemotronHMOE experts (NemotronHMLP) use up_proj and down_proj only (no gate).
            block_names=("NemotronHMOE",),
            expert_linear_names=("up_proj", "down_proj"),
        ),
    )
)
