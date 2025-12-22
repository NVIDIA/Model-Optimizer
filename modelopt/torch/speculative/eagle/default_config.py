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

default_kimik2_eagle_config = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "aux_loss_alpha": 0.001,
    "bos_token_id": 163584,
    "eos_token_id": 163586,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "intermediate_size": 18432,
    "kv_lora_rank": 512,
    "max_position_embeddings": 262144,
    "model_type": "kimi_k2",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_group": 1,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 1,
    "num_key_value_heads": 64,
    "num_nextn_predict_layers": 0,
    "num_return_sequences": 1,
    "output_attentions": False,
    "output_hidden_states": False,
    "output_scores": False,
    "pad_token_id": 163839,
    "prefix": None,
    "pretraining_tp": 1,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 0.00001,
    "rope_scaling": {
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 50000.0,
    "routed_scaling_factor": 2.827,
    "scoring_func": "sigmoid",
    "seq_aux": True,
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.3",
    "use_cache": True,
    "v_head_dim": 128,
    "_attn_implementation": "eager",
    "use_input_layernorm_in_first_layer": True,
    "use_last_layernorm": True,
    "use_aux_hidden_state": True,
    "eagle_aux_hidden_state_layer_ids": [],
    "use_mtp_layernorm": False,
    "parallel_draft_step": 1,
    "parallel_draft_heads_num_layers": 1,
    "has_lm_head": False,
}
