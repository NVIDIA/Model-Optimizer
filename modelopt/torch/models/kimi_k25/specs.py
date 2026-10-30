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

"""kimi_k25 specs (HF model type ``kimi_k25``).

Kimi-K2.5 (Kimi-K2.6 and Kimi-K2.7-Code share its model type): the compressed-expert sibling of
``kimi_vl`` -- same swap to a native deepseek_v3 language model, plus on-the-fly decompression of
its packed experts at load.

Only the expert-parallel section is registered under ``kimi_k25``: the MoE and export sections stay
``deepseek_v3``'s, so lookups by ``kimi_k25`` find neither yet. Nor do ``is_moe``'s name and structure
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
        model_type="kimi_k25",
        # transformers ships Kimi-K2.5 itself from 5.13, so modeling_source stays "transformers": the
        # registry pairs it with that release, and a "remote_code" claim would be stale from 5.13 on.
        # Expert parallelism still runs on the checkpoint's own code (``trust_remote_code=True``),
        # which ``prepare`` patches, refusing the port. So the version describes transformers' port,
        # not the model this spec's load builds: a MoE or export section added here must name the
        # native deepseek_v3 language model's modules.
        min_transformers_version="5.13",
        # prepare swaps a native deepseek_v3 in as the language model, with deepseek_v3's plan,
        # and decompresses the experts as they load, which needs transformers 5.8.
        ep_spec=EPSpec(
            prepare=prepare,
            prepared_ep_plan=BASE_MODEL_EP_PLAN,
            plan_config_attr="text_config",
            make_importable=make_importable,
            requires_transformers="5.8",
        ),
    )
)
