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

"""Custom mapping from GPT-OSS Hugging Face models to Megatron Core models."""

from .mcore_custom import (
    COL_TP,
    PACK_EP,
    REPLICATE,
    ROW_TP,
    CustomModuleMapping,
    NameRemapping,
    PackNameRemappingGPT,
    QKVMerging,
    QKVSlicing,
    UnpackNameRemappingGPT,
)

gptoss_causal_lm_export: dict[str, CustomModuleMapping | bool] = {
    "word_embeddings": NameRemapping("model.embed_tokens."),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm."),
    "linear_qkv": QKVSlicing("model.layers.{}.self_attn."),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    "softmax_offset": NameRemapping("model.layers.{}.self_attn.sinks"),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm."),
    "use_packed_local_experts": True,
    "local_experts.linear_fc1": PackNameRemappingGPT(
        "model.layers.{}.mlp.experts.gate_up_proj",
        {"layer_type": "linear_fc1"},
    ),
    "local_experts.linear_fc2": PackNameRemappingGPT(
        "model.layers.{}.mlp.experts.down_proj",
        {"layer_type": "linear_fc2"},
    ),
    "router": NameRemapping("model.layers.{}.mlp.router."),
    "final_layernorm": NameRemapping("model.norm."),
    "output_layer": NameRemapping("lm_head."),
}

gptoss_causal_lm_import: dict[str, CustomModuleMapping | bool] = {
    "word_embeddings": NameRemapping("model.embed_tokens.", COL_TP),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm.", REPLICATE),
    "linear_qkv": QKVMerging("model.layers.{}.self_attn.", COL_TP),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj.", ROW_TP),
    "softmax_offset": NameRemapping("model.layers.{}.self_attn.sinks", COL_TP),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.", REPLICATE),
    "router": NameRemapping("model.layers.{}.mlp.router.", REPLICATE),
    "use_packed_local_experts": True,
    "local_experts.linear_fc1_ep": UnpackNameRemappingGPT(
        "model.layers.{}.mlp.experts.gate_up_proj",
        PACK_EP | {"layer_type": "linear_fc1"},
    ),
    "local_experts.linear_fc2_ep": UnpackNameRemappingGPT(
        "model.layers.{}.mlp.experts.down_proj",
        PACK_EP | {"layer_type": "linear_fc2"},
    ),
    "final_layernorm": NameRemapping("model.norm.", REPLICATE),
    "output_layer": NameRemapping("lm_head.", COL_TP),
}
