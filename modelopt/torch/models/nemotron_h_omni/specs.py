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


"""NemotronH-Omni specs (HF model type ``nemotron_h_omni``).

The Omni VLM's language sub-model is a native NemotronH, so its MoE facts are NemotronH's.
What differs is where the expert-parallel plan has to be attached: on the nested ``llm_config``,
not the VLM root. transformers aggregates each sub-model's plan up to the root WITH its module
prefix, so a plan on ``llm_config`` surfaces as ``language_model.<base>.layers.*`` and matches the
real parameter paths -- while a plan on the top config reaches no expert at all.
"""

from ..nemotron_h.specs import BASE_MODEL_EP_PLAN
from ..specs import EPSpec, ModelSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="nemotron_h_omni",
        # transformers ships the Omni VLM itself from 5.18. Expert parallelism still runs on the
        # checkpoint's own code (``trust_remote_code=True``), whose sub-model config is the
        # ``llm_config`` the plan below targets; transformers' port names it ``text_config``.
        min_transformers_version="5.18",
        ep_spec=EPSpec(
            base_model_ep_plan=BASE_MODEL_EP_PLAN,
            plan_config_attr="llm_config",
        ),
    )
)
