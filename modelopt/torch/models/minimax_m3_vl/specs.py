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

"""MiniMax-M3 specs (HF model type ``minimax_m3_vl``).

M3's MoE block already follows the sparse-dispatch convention; the only gap is that its experts
hardcode a per-expert loop instead of dispatching through the ExpertsInterface, so ``ep.py``
decorates the class and the generic transport takes over unchanged.
"""

from ..specs import EPSpec, ModelSpec, register
from .ep import BASE_MODEL_EP_PLAN, prepare

__all__: list[str] = []

register(
    ModelSpec(
        model_type="minimax_m3_vl",
        min_transformers_version="5.13",
        ep_spec=EPSpec(
            base_model_ep_plan=BASE_MODEL_EP_PLAN,
            plan_config_attr="text_config",
            requires_transformers="5.13",
            prepare=prepare,
        ),
    )
)
