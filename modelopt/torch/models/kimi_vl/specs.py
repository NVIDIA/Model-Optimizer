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

"""kimi_vl specs (HF model type ``kimi_vl``).

Kimi-VL-A3B: a remote-code VLM whose text backbone is a Moonlight MoE. Its bundled
modeling builds the language model from per-expert Linears, which the EP machinery cannot shard, so
``ep.py`` swaps in the transformers-native deepseek_v3 language model.

Only the expert-parallel section is registered under ``kimi_vl``: the MoE and export sections stay
``deepseek_v3``'s, so lookups by ``kimi_vl`` find neither yet. Nor do ``is_moe``'s name and structure
fallbacks recognize the native ``DeepseekV3MoE`` blocks ``prepare`` installs, so the HF export's
MoE-block handling skips them. Sharing the sections is left to a follow-up.
"""

from .._kimi import make_importable
from ..deepseek_v3.specs import BASE_MODEL_EP_PLAN
from ..specs import EPSpec, ModelSpec, register
from .ep import prepare

__all__: list[str] = []

register(
    ModelSpec(
        model_type="kimi_vl",
        modeling_source="remote_code",
        # prepare swaps a native deepseek_v3 in as the language model, with deepseek_v3's plan.
        ep_spec=EPSpec(
            prepare=prepare,
            prepared_ep_plan=BASE_MODEL_EP_PLAN,
            plan_config_attr="text_config",
            make_importable=make_importable,
        ),
    )
)
