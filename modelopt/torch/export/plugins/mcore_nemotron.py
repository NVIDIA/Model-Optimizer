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


"""Custom mapping from Nemotron Hugging Face models to Megatron Core models."""

from .mcore_custom import (
    COL_ETP,
    COL_TP,
    REPLICATE,
    ROW_ETP,
    ROW_TP,
    CustomModuleMapping,
    GroupedMLPMerging,
    NameRemapping,
    QKVMerging,
    QKVSlicing,
    SelfAttentionScaling,
)

# Example on adding a new CausalLM.
nemotron_causal_lm_export: dict[str, CustomModuleMapping] = {
    # NemotronForCausalLM is using square-relu where no gated handle is needed.
    "word_embeddings": NameRemapping("model.embed_tokens."),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm."),
    "linear_qkv": QKVSlicing("model.layers.{}.self_attn."),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    "core_attention": SelfAttentionScaling("backbone.layers.{}.mixer."),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm."),
    # NemotronForCausalLM is using square-relu where no gated handle is needed.
    "linear_fc1": NameRemapping("model.layers.{}.mlp.up_proj."),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj."),
    "final_layernorm": NameRemapping("model.norm."),
    "output_layer": NameRemapping("lm_head."),
}


nemotron_h_causal_lm_import: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("backbone.embeddings.", COL_TP),
    "final_norm": NameRemapping("backbone.norm_f.", REPLICATE),
    "output_layer": NameRemapping("lm_head.", COL_TP),
    # Mamba
    "norm": NameRemapping("backbone.layers.{}.norm.", REPLICATE),
    "mixer_norm": NameRemapping("backbone.layers.{}.mixer.norm.", REPLICATE),
    "A_log": NameRemapping("backbone.layers.{}.mixer.A_log", REPLICATE),
    "D": NameRemapping("backbone.layers.{}.mixer.D", REPLICATE),
    "dt_bias": NameRemapping("backbone.layers.{}.mixer.dt_bias", REPLICATE),
    "conv1d": NameRemapping("backbone.layers.{}.mixer.conv1d.", REPLICATE),
    "in_proj": NameRemapping("backbone.layers.{}.mixer.in_proj.", COL_TP),
    "out_proj": NameRemapping("backbone.layers.{}.mixer.out_proj.", ROW_TP),
    # Attention
    "input_layernorm": NameRemapping("backbone.layers.{}.norm.", REPLICATE),
    "linear_qkv": QKVMerging("backbone.layers.{}.mixer.", COL_TP),
    "linear_proj": NameRemapping("backbone.layers.{}.mixer.o_proj.", ROW_TP),
    # MLP
    "pre_mlp_layernorm": NameRemapping("backbone.layers.{}.norm.", REPLICATE),
    "linear_fc1": NameRemapping("backbone.layers.{}.mixer.up_proj.", COL_TP),
    "linear_fc2": NameRemapping("backbone.layers.{}.mixer.down_proj.", ROW_TP),
    # MoE
    "router": NameRemapping(
        "backbone.layers.{}.mixer.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "local_experts.linear_fc1": NameRemapping(
        "backbone.layers.{}.mixer.experts.{}.up_proj.", COL_ETP
    ),
    "local_experts.linear_fc2": NameRemapping(
        "backbone.layers.{}.mixer.experts.{}.down_proj.", ROW_ETP
    ),
    "shared_experts.linear_fc1": NameRemapping(
        "backbone.layers.{}.mixer.shared_experts.up_proj.", COL_TP
    ),
    "shared_experts.linear_fc2": NameRemapping(
        "backbone.layers.{}.mixer.shared_experts.down_proj.", ROW_TP
    ),
    # Latent MoE
    "fc1_latent_proj": NameRemapping("backbone.layers.{}.mixer.fc1_latent_proj.", REPLICATE),
    "fc2_latent_proj": NameRemapping("backbone.layers.{}.mixer.fc2_latent_proj.", REPLICATE),
    # Repeated MTP module
    "mtp.enorm": NameRemapping("mtp.layers.{}.enorm.", {"is_mtp": True}),
    "mtp.hnorm": NameRemapping("mtp.layers.{}.hnorm.", {"is_mtp": True}),
    "mtp.eh_proj": NameRemapping("mtp.layers.{}.eh_proj.", {"is_mtp": True}),
    "mtp.final_layernorm": NameRemapping("mtp.layers.{}.final_layernorm.", {"is_mtp": True}),
    # Grouped local experts in MTP
    "experts.linear_fc1": GroupedMLPMerging(
        "mtp.layers.{}.mixer.experts.{{}}.up_proj", COL_ETP | {"is_mtp": True}
    ),
    "experts.linear_fc2": GroupedMLPMerging(
        "mtp.layers.{}.mixer.experts.{{}}.down_proj", ROW_ETP | {"is_mtp": True}
    ),
}

nemotron_h_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("backbone.embeddings."),
    "final_norm": NameRemapping("backbone.norm_f."),
    "output_layer": NameRemapping("lm_head."),
    # Mamba
    "norm": NameRemapping("backbone.layers.{}.norm."),
    "mixer_norm": NameRemapping("backbone.layers.{}.mixer.norm."),
    "A_log": NameRemapping("backbone.layers.{}.mixer.A_log"),
    "D": NameRemapping("backbone.layers.{}.mixer.D"),
    "dt_bias": NameRemapping("backbone.layers.{}.mixer.dt_bias"),
    "conv1d": NameRemapping("backbone.layers.{}.mixer.conv1d."),
    "in_proj": NameRemapping("backbone.layers.{}.mixer.in_proj."),
    "out_proj": NameRemapping("backbone.layers.{}.mixer.out_proj."),
    # Attention
    "input_layernorm": NameRemapping("backbone.layers.{}.norm."),
    "linear_qkv": QKVSlicing("backbone.layers.{}.mixer."),
    "linear_proj": NameRemapping("backbone.layers.{}.mixer.o_proj."),
    "core_attention": SelfAttentionScaling("backbone.layers.{}.mixer."),
    # MLP
    "pre_mlp_layernorm": NameRemapping("backbone.layers.{}.norm."),
    "linear_fc1": NameRemapping("backbone.layers.{}.mixer.up_proj."),
    "linear_fc2": NameRemapping("backbone.layers.{}.mixer.down_proj."),
    # MoE
    "router": NameRemapping(
        "backbone.layers.{}.mixer.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "local_experts.linear_fc1": NameRemapping("backbone.layers.{}.mixer.experts.{}.up_proj."),
    "local_experts.linear_fc2": NameRemapping("backbone.layers.{}.mixer.experts.{}.down_proj."),
    "shared_experts.linear_fc1": NameRemapping("backbone.layers.{}.mixer.shared_experts.up_proj."),
    "shared_experts.linear_fc2": NameRemapping(
        "backbone.layers.{}.mixer.shared_experts.down_proj."
    ),
    # Latent MoE
    "fc1_latent_proj": NameRemapping("backbone.layers.{}.mixer.fc1_latent_proj."),
    "fc2_latent_proj": NameRemapping("backbone.layers.{}.mixer.fc2_latent_proj."),
    # MTP
    "mtp.enorm": NameRemapping("mtp.layers.{}.enorm."),
    "mtp.hnorm": NameRemapping("mtp.layers.{}.hnorm."),
    "mtp.eh_proj": NameRemapping("mtp.layers.{}.eh_proj."),
    "mtp.final_layernorm": NameRemapping("mtp.layers.{}.final_layernorm."),
}
