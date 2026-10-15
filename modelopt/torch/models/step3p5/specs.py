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

"""Step-3.5 specs (HF model type ``step3p5``)."""

from ..specs import ModelSpec, register

__all__: list[str] = []

# Native in transformers since 5.16, where the step3p7 package also registers this text model
# type; older Step-3.5 checkpoints ship their own modeling code (trust_remote_code).
#
# No MoESpec. The native MoE block (Step3p7SparseMoeBlock, with fused gate_up_proj/down_proj
# experts) is already recognized by is_moe by name and handled by the generic fused-experts
# path. The remote-code layout keeps its routed experts as expert-indexed MoELinear
# projections on the MoE MLP itself, with no `experts` container; declaring that block would
# make is_moe claim it and send AWQ export into get_experts_list, which does not support it.
register(ModelSpec(model_type="step3p5", min_transformers_version="5.16"))
