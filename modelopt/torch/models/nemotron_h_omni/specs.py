# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
What differs is where the expert-parallel plan has to be attached: on the nested language-model
config, not the VLM root. transformers aggregates each sub-model's plan up to the root WITH its
module prefix, so a plan there surfaces as ``language_model.<base>.layers.*`` and matches the real
parameter paths -- while a plan on the top config reaches no expert at all.

Only that plan is registered under the root type, so lookups of the MoE and export sections by
``nemotron_h_omni`` find neither. No export depends on them: on transformers 5 the language
model's experts are one fused container, which the export writes through its own handler, and
every step those lookups gate leaves a fused container as it is.
"""

from ..nemotron_h.specs import BASE_MODEL_EP_PLAN
from ..specs import EPSpec, ModelSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="nemotron_h_omni",
        # transformers ships the Omni VLM itself from 5.18. The checkpoint's own code
        # (``trust_remote_code=True``) names the language-model config ``llm_config``, and
        # transformers' port ``text_config``: the plan goes on whichever the config has, tried in
        # the order the load reads them.
        min_transformers_version="5.18",
        ep_spec=EPSpec(
            # Its own copy: the registry is process-wide, and a spec holding NemotronH's dict would
            # carry any change to one plan into the other.
            base_model_ep_plan=dict(BASE_MODEL_EP_PLAN),
            plan_config_attr=("text_config", "llm_config"),
        ),
    )
)
