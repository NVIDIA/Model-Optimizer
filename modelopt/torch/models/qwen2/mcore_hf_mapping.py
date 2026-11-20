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

"""Custom mapping from Qwen2 Hugging Face models to Megatron Core models."""

from modelopt.torch.export.plugins.mcore_custom import (
    COL_TP,
    REPLICATE,
    ROW_TP,
    CustomModuleMapping,
    GatedMLPMerging,
    GatedMLPSlicing,
    NameRemapping,
    QKVMerging,
    QKVSlicing,
    SelfAttentionScaling,
    register_mcore_hf_mapping,
)

__all__: list[str] = []

qwen25_causal_lm_import: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("model.embed_tokens.", COL_TP),
    "final_layernorm": NameRemapping("model.norm.", REPLICATE),
    "output_layer": NameRemapping("lm_head.", COL_TP),
    # Attention
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm.", REPLICATE),
    "fused_input_layernorm": NameRemapping("model.layers.{}.input_layernorm.weight"),
    "linear_qkv": QKVMerging("model.layers.{}.self_attn.", COL_TP),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj.", ROW_TP),
    # MLP
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.", REPLICATE),
    "fused_pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.weight"),
    "linear_fc1": GatedMLPMerging("model.layers.{}.mlp.", COL_TP),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj.", ROW_TP),
}

qwen25_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("model.embed_tokens."),
    "final_layernorm": NameRemapping("model.norm."),
    "output_layer": NameRemapping("lm_head."),
    # Attention
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm."),
    "fused_input_layernorm": NameRemapping("model.layers.{}.input_layernorm.weight"),
    "linear_qkv": QKVSlicing("model.layers.{}.self_attn."),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    # KV cache quant export
    "core_attention": SelfAttentionScaling("model.layers.{}.self_attn."),
    # MLP
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm."),
    "fused_pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.weight"),
    "linear_fc1": GatedMLPSlicing("model.layers.{}.mlp."),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj."),
}

register_mcore_hf_mapping(
    "Qwen2ForCausalLM",
    export_mapping=qwen25_causal_lm_export,
    import_mapping=qwen25_causal_lm_import,
)
