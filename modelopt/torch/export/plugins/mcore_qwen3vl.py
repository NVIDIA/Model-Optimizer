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

"""Custom mapping from Qwen3-VL Hugging Face models to Megatron Core models.

Qwen3-VL model structure differs from Qwen3:
- Language model weights are under `model.language_model.` prefix
- Visual encoder weights are under `model.visual.` prefix

This module handles the language model conversion for PTQ/QAT workflows.
Visual components are typically kept in full precision.

HuggingFace Qwen3-VL-8B structure:
- model.language_model.embed_tokens.weight
- model.language_model.layers.{L}.input_layernorm.weight
- model.language_model.layers.{L}.self_attn.q_proj.weight
- model.language_model.layers.{L}.self_attn.k_proj.weight
- model.language_model.layers.{L}.self_attn.v_proj.weight
- model.language_model.layers.{L}.self_attn.q_norm.weight
- model.language_model.layers.{L}.self_attn.k_norm.weight
- model.language_model.layers.{L}.self_attn.o_proj.weight
- model.language_model.layers.{L}.post_attention_layernorm.weight
- model.language_model.layers.{L}.mlp.gate_proj.weight
- model.language_model.layers.{L}.mlp.up_proj.weight
- model.language_model.layers.{L}.mlp.down_proj.weight
- model.language_model.norm.weight
- lm_head.weight
"""

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
    QKVMerging,
    QKVSlicing,
)

# Import rules: HuggingFace -> Megatron Core
qwen3vl_causal_lm_import: dict[str, CustomModuleMapping] = {
    # Embeddings - note the language_model prefix
    "word_embeddings": NameRemapping("model.language_model.embed_tokens.", COL_TP),
    # Final layer norm
    "final_layernorm": NameRemapping("model.language_model.norm.", REPLICATE),
    # Output layer (lm_head is at root level, not under language_model)
    "output_layer": NameRemapping("lm_head.", COL_TP),
    # Attention - input layernorm
    "input_layernorm": NameRemapping("model.language_model.layers.{}.input_layernorm.", REPLICATE),
    # Attention - QKV projection (merged)
    "linear_qkv": QKVMerging("model.language_model.layers.{}.self_attn.", COL_TP),
    # Attention - output projection
    "linear_proj": NameRemapping("model.language_model.layers.{}.self_attn.o_proj.", ROW_TP),
    # Attention - Q/K layer norms (Qwen3 uses RMSNorm on Q and K)
    "q_layernorm": NameRemapping("model.language_model.layers.{}.self_attn.q_norm.", REPLICATE),
    "k_layernorm": NameRemapping("model.language_model.layers.{}.self_attn.k_norm.", REPLICATE),
    # MLP - pre-MLP layernorm (post_attention_layernorm in HF)
    "pre_mlp_layernorm": NameRemapping(
        "model.language_model.layers.{}.post_attention_layernorm.", REPLICATE
    ),
    # MLP - gate_proj + up_proj merged into linear_fc1
    "linear_fc1": GatedMLPMerging("model.language_model.layers.{}.mlp.", COL_TP),
    # MLP - down_proj as linear_fc2
    "linear_fc2": NameRemapping("model.language_model.layers.{}.mlp.down_proj.", ROW_TP),
    # MoE support (for Qwen3-VL MoE variants like 30B-A3B)
    "router": NameRemapping("model.language_model.layers.{}.mlp.gate.", REPLICATE),
    "local_experts.linear_fc1": GatedMLPMerging(
        "model.language_model.layers.{}.mlp.experts.{}.", COL_ETP
    ),
    "local_experts.linear_fc2": NameRemapping(
        "model.language_model.layers.{}.mlp.experts.{}.down_proj.", ROW_ETP
    ),
}

# Export rules: Megatron Core -> HuggingFace
qwen3vl_causal_lm_export: dict[str, CustomModuleMapping] = {
    # Embeddings
    "word_embeddings": NameRemapping("model.language_model.embed_tokens."),
    # Final layer norm
    "final_layernorm": NameRemapping("model.language_model.norm."),
    # Output layer
    "output_layer": NameRemapping("lm_head."),
    # Attention - input layernorm
    "input_layernorm": NameRemapping("model.language_model.layers.{}.input_layernorm."),
    # Attention - QKV projection (sliced back to separate q/k/v)
    "linear_qkv": QKVSlicing("model.language_model.layers.{}.self_attn."),
    # Attention - output projection
    "linear_proj": NameRemapping("model.language_model.layers.{}.self_attn.o_proj."),
    # Attention - Q/K layer norms
    "q_layernorm": NameRemapping("model.language_model.layers.{}.self_attn.q_norm."),
    "k_layernorm": NameRemapping("model.language_model.layers.{}.self_attn.k_norm."),
    # MLP - pre-MLP layernorm
    "pre_mlp_layernorm": NameRemapping("model.language_model.layers.{}.post_attention_layernorm."),
    # MLP - linear_fc1 sliced back to gate_proj + up_proj
    "linear_fc1": GatedMLPSlicing("model.language_model.layers.{}.mlp."),
    # MLP - down_proj
    "linear_fc2": NameRemapping("model.language_model.layers.{}.mlp.down_proj."),
    # MoE support
    "router": NameRemapping("model.language_model.layers.{}.mlp.gate."),
    "local_experts.linear_fc1": GatedMLPSlicing("model.language_model.layers.{}.mlp.experts.{}."),
    "local_experts.linear_fc2": NameRemapping(
        "model.language_model.layers.{}.mlp.experts.{}.down_proj."
    ),
}