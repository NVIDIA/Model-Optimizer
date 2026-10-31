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

transformers ships everything M3 needs to run expert-parallel: its MoE block calls the experts with
the router's top-k, its experts dispatch through the experts interface, and its text config carries a
``base_model_ep_plan``. The generic all-to-all transport takes over unchanged.
"""

from ..specs import EPSpec, ModelSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="minimax_m3_vl",
        min_transformers_version="5.13",
        ep_spec=EPSpec(requires_transformers="5.13"),
    )
)
