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


"""Custom mapping from DeepSeek Hugging Face models to Megatron Core models."""

from .mcore_custom import (
    COL_ETP,
    COL_TP,
    REPLICATE,
    ROW_ETP,
    ROW_TP,
    CustomModuleMapping,
    GatedMLPMerging,
    GatedMLPSlicing,
    NameRemapping,
)

deepseek_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("model.embed_tokens."),
    "final_layernorm": NameRemapping("model.norm."),
    "output_layer": NameRemapping("lm_head."),
    # Multi-Latent Attention (V3 has lora on q as well)
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm."),
    "linear_q_proj": NameRemapping("model.layers.{}.self_attn.q_proj."),
    "linear_q_down_proj": NameRemapping("model.layers.{}.self_attn.q_a_proj."),
    "linear_q_layernorm": NameRemapping("model.layers.{}.self_attn.q_a_layernorm."),
    "linear_q_up_proj": NameRemapping("model.layers.{}.self_attn.q_b_proj."),
    "linear_kv_down_proj": NameRemapping("model.layers.{}.self_attn.kv_a_proj_with_mqa."),
    "linear_kv_layernorm": NameRemapping("model.layers.{}.self_attn.kv_a_layernorm."),
    "linear_kv_up_proj": NameRemapping("model.layers.{}.self_attn.kv_b_proj."),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm."),
    # MLP for dense layers
    "linear_fc1": GatedMLPSlicing("model.layers.{}.mlp."),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj."),
    # MoE shared experts
    "router": NameRemapping(
        "model.layers.{}.mlp.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "shared_experts.linear_fc1": GatedMLPSlicing("model.layers.{}.mlp.shared_experts."),
    "shared_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.shared_experts.down_proj."),
    # MoE local experts
    "local_experts.linear_fc1": GatedMLPSlicing("model.layers.{}.mlp.experts.{}."),
    "local_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.experts.{}.down_proj."),
}


eagle_mtp_deepseek_causal_lm_export: dict[str, CustomModuleMapping] = {
    "fc": NameRemapping("eh_proj."),
    "enorm": NameRemapping("enorm."),
    "hnorm": NameRemapping("hnorm."),
    "input_layernorm": NameRemapping("layers.{}.input_layernorm."),
    "linear_q_proj": NameRemapping("layers.{}.self_attn.q_proj."),
    "linear_q_down_proj": NameRemapping("layers.{}.self_attn.q_a_proj."),
    "linear_q_layernorm": NameRemapping("layers.{}.self_attn.q_a_layernorm."),
    "linear_q_up_proj": NameRemapping("layers.{}.self_attn.q_b_proj."),
    "linear_kv_down_proj": NameRemapping("layers.{}.self_attn.kv_a_proj_with_mqa."),
    "linear_kv_layernorm": NameRemapping("layers.{}.self_attn.kv_a_layernorm."),
    "linear_kv_up_proj": NameRemapping("layers.{}.self_attn.kv_b_proj."),
    "linear_proj": NameRemapping("layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": NameRemapping("layers.{}.post_attention_layernorm."),
    "router": NameRemapping(
        "layers.{}.mlp.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "shared_experts.linear_fc1": GatedMLPSlicing("layers.{}.mlp.shared_experts."),
    "shared_experts.linear_fc2": NameRemapping("layers.{}.mlp.shared_experts.down_proj."),
    "local_experts.linear_fc1": GatedMLPSlicing("layers.{}.mlp.experts.{}."),
    "local_experts.linear_fc2": NameRemapping("layers.{}.mlp.experts.{}.down_proj."),
}


deepseek_causal_lm_import = {
    "word_embeddings": NameRemapping("model.embed_tokens.", COL_TP),
    "final_layernorm": NameRemapping("model.norm.", REPLICATE),
    "output_layer": NameRemapping("lm_head.", COL_TP),
    # Per-layer
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm.", REPLICATE),
    "linear_q_proj": NameRemapping("model.layers.{}.self_attn.q_proj.", COL_TP),
    "linear_q_down_proj": NameRemapping("model.layers.{}.self_attn.q_a_proj.", REPLICATE),
    "linear_q_layernorm": NameRemapping("model.layers.{}.self_attn.q_a_layernorm.", REPLICATE),
    "linear_q_up_proj": NameRemapping("model.layers.{}.self_attn.q_b_proj.", COL_TP),
    "linear_kv_down_proj": NameRemapping(
        "model.layers.{}.self_attn.kv_a_proj_with_mqa.", REPLICATE
    ),
    "linear_kv_layernorm": NameRemapping("model.layers.{}.self_attn.kv_a_layernorm.", REPLICATE),
    "linear_kv_up_proj": NameRemapping("model.layers.{}.self_attn.kv_b_proj.", COL_TP),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj.", ROW_TP),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.", REPLICATE),
    "linear_fc1": GatedMLPMerging("model.layers.{}.mlp.", COL_TP),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj.", ROW_TP),
    # MoE shared experts
    "router": NameRemapping(
        "model.layers.{}.mlp.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "shared_experts.linear_fc1": GatedMLPMerging("model.layers.{}.mlp.shared_experts.", COL_TP),
    "shared_experts.linear_fc2": NameRemapping(
        "model.layers.{}.mlp.shared_experts.down_proj.", ROW_TP
    ),
    # MoE local experts
    "local_experts.linear_fc1": GatedMLPMerging("model.layers.{}.mlp.experts.{}.", COL_ETP),
    "local_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.experts.{}.down_proj.", ROW_ETP),
}


eagle_mtp_deepseek_causal_lm_import: dict[str, CustomModuleMapping] = {
    "fc": NameRemapping("model.layers.{}.eh_proj.", REPLICATE),
    "enorm": NameRemapping("model.layers.{}.enorm.", REPLICATE),
    "hnorm": NameRemapping("model.layers.{}.hnorm.", REPLICATE),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm.", REPLICATE),
    "linear_q_proj": NameRemapping("model.layers.{}.self_attn.q_proj.", COL_TP),
    "linear_q_down_proj": NameRemapping("model.layers.{}.self_attn.q_a_proj.", REPLICATE),
    "linear_q_layernorm": NameRemapping("model.layers.{}.self_attn.q_a_layernorm.", REPLICATE),
    "linear_q_up_proj": NameRemapping("model.layers.{}.self_attn.q_b_proj.", COL_TP),
    "linear_kv_down_proj": NameRemapping(
        "model.layers.{}.self_attn.kv_a_proj_with_mqa.", REPLICATE
    ),
    "linear_kv_layernorm": NameRemapping("model.layers.{}.self_attn.kv_a_layernorm.", REPLICATE),
    "linear_kv_up_proj": NameRemapping("model.layers.{}.self_attn.kv_b_proj.", COL_TP),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj.", ROW_TP),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.", REPLICATE),
    "router": NameRemapping(
        "model.layers.{}.mlp.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "shared_experts.linear_fc1": GatedMLPMerging("model.layers.{}.mlp.shared_experts.", COL_TP),
    "shared_experts.linear_fc2": NameRemapping(
        "model.layers.{}.mlp.shared_experts.down_proj.", ROW_TP
    ),
    "local_experts.linear_fc1": GatedMLPMerging("model.layers.{}.mlp.experts.{}.", COL_ETP),
    "local_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.experts.{}.down_proj.", ROW_ETP),
}
