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

"""Step-3.5 specs (HF model type ``step3p5``)."""

from ..specs import ModelSpec, register

__all__: list[str] = []

# No MoESpec: the routed experts are expert-indexed MoELinear projections on the MoE MLP
# itself, with no `experts` container, and are exported by the MoELinear handler rather
# than through the MoE-block lookups. Declaring the block would make is_moe claim it and
# send AWQ export into get_experts_list, which does not support this layout.
register(ModelSpec(model_type="step3p5", modeling_source="remote_code"))
