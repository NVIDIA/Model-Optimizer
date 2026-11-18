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


"""DeepSeek V4.1 text replay and native MoE ownership facts."""

from ..specs import BypassSpec, EPSpec, ModelSpec, MoESpec, register

register(
    ModelSpec(
        model_type="deepseek_v41",
        modeling_source="modelopt",
        min_transformers_version="5.23",
        bypass_spec=BypassSpec(
            decoder_paths=("model.layers", "layers"),
            feedforward_names=("ffn",),
            batch_packed_documents=False,
        ),
        moe_spec=MoESpec(
            block_names=("V41MoE", "DeepseekV41MoE"),
            expert_linear_names=("w1", "w2", "w3"),
            gate_up_pair=("w1", "w3"),
        ),
        ep_spec=EPSpec(requires_transformers="5.23"),
    )
)
