# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Default EAGLE architecture config."""

default_eagle_config = {
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
    "position_embedding_type": "rope",
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "num_hidden_layers": 1,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "initializer_range": 0.01,
    "rms_norm_eps": 1e-05,
    "mlp_bias": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_input_layernorm_in_first_layer": True,
    "use_last_layernorm": False,
    "use_aux_hidden_state": False,
    "eagle_aux_hidden_state_layer_ids": [],
    "use_mtp_layernorm": False,
    "parallel_draft_step": 1,
    "parallel_draft_heads_num_layers": 1,
    "has_lm_head": False,
    "head_dim": 128,
}
