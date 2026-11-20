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

"""Custom mapping from Llama-4 Hugging Face models to Megatron Core models."""

from modelopt.torch.export.plugins.mcore_custom import (
    COL_TP,
    PACK_COL_ETP,
    PACK_EP,
    PACK_ROW_ETP,
    REPLICATE,
    ROW_TP,
    CustomModuleMapping,
    GatedMLPMerging,
    GatedMLPSlicing,
    NameRemapping,
    PackNameRemapping,
    QKVMerging,
    QKVSlicing,
    UnpackNameRemapping,
    register_mcore_hf_mapping,
)

__all__: list[str] = []

llama4_causal_lm_export: dict[str, CustomModuleMapping | bool] = {
    "word_embeddings": NameRemapping("language_model.model.embed_tokens."),
    "input_layernorm": NameRemapping("language_model.model.layers.{}.input_layernorm."),
    # MoE-only on MLP side, so fused_pre_mlp_layernorm path is unreachable.
    "fused_input_layernorm": NameRemapping("language_model.model.layers.{}.input_layernorm.weight"),
    # self_attn
    "linear_qkv": QKVSlicing("language_model.model.layers.{}.self_attn."),
    "linear_proj": NameRemapping("language_model.model.layers.{}.self_attn.o_proj."),
    # mlp
    "pre_mlp_layernorm": NameRemapping("language_model.model.layers.{}.post_attention_layernorm."),
    "shared_experts.linear_fc1": GatedMLPSlicing(
        "language_model.model.layers.{}.feed_forward.shared_expert.",
    ),
    "shared_experts.linear_fc2": NameRemapping(
        "language_model.model.layers.{}.feed_forward.shared_expert.down_proj.",
    ),
    # moe_layer
    "router": NameRemapping("language_model.model.layers.{}.feed_forward.router."),
    "use_packed_local_experts": True,
    "local_experts.linear_fc1": PackNameRemapping(
        "language_model.model.layers.{}.feed_forward.experts.gate_up_proj",
        {"layer_type": "linear_fc1"},
    ),
    "local_experts.linear_fc2": PackNameRemapping(
        "language_model.model.layers.{}.feed_forward.experts.down_proj",
        {"layer_type": "linear_fc2"},
    ),
    "final_layernorm": NameRemapping("language_model.model.norm."),
    "output_layer": NameRemapping("language_model.lm_head."),
}

llama4_causal_lm_import: dict[str, CustomModuleMapping | bool] = {
    "word_embeddings": NameRemapping("language_model.model.embed_tokens.", COL_TP),
    "input_layernorm": NameRemapping("language_model.model.layers.{}.input_layernorm.", REPLICATE),
    # Fused TE spec (TELayerNormColumnParallelLinear) — see qwen3/mcore_hf_mapping.py for rationale.
    # Llama4 is MoE-only on the MLP side (no layer.mlp.linear_fc1), so the importer's
    # fused_pre_mlp_layernorm path is unreachable; only fused_input_layernorm is wired.
    "fused_input_layernorm": NameRemapping("language_model.model.layers.{}.input_layernorm.weight"),
    "linear_qkv": QKVMerging("language_model.model.layers.{}.self_attn.", COL_TP),
    "linear_proj": NameRemapping("language_model.model.layers.{}.self_attn.o_proj.", ROW_TP),
    "pre_mlp_layernorm": NameRemapping(
        "language_model.model.layers.{}.post_attention_layernorm.", REPLICATE
    ),
    "shared_experts.linear_fc1": GatedMLPMerging(
        "language_model.model.layers.{}.feed_forward.shared_expert.", COL_TP
    ),
    "shared_experts.linear_fc2": NameRemapping(
        "language_model.model.layers.{}.feed_forward.shared_expert.down_proj.", ROW_TP
    ),
    "router": NameRemapping("language_model.model.layers.{}.feed_forward.router.", REPLICATE),
    "use_packed_local_experts": True,
    "local_experts.linear_fc1_etp": UnpackNameRemapping(
        "language_model.model.layers.{}.feed_forward.experts.gate_up_proj",
        PACK_COL_ETP | {"layer_type": "linear_fc1"},
    ),
    "local_experts.linear_fc2_etp": UnpackNameRemapping(
        "language_model.model.layers.{}.feed_forward.experts.down_proj",
        PACK_ROW_ETP | {"layer_type": "linear_fc2"},
    ),
    "local_experts.linear_fc1_ep": UnpackNameRemapping(
        "language_model.model.layers.{}.feed_forward.experts.gate_up_proj",
        PACK_EP | {"layer_type": "linear_fc1"},
    ),
    "local_experts.linear_fc2_ep": UnpackNameRemapping(
        "language_model.model.layers.{}.feed_forward.experts.down_proj",
        PACK_EP | {"layer_type": "linear_fc2"},
    ),
    "final_layernorm": NameRemapping("language_model.model.norm.", REPLICATE),
    "output_layer": NameRemapping("language_model.lm_head.", COL_TP),
}

register_mcore_hf_mapping(
    "Llama4ForConditionalGeneration",
    export_mapping=llama4_causal_lm_export,
    import_mapping=llama4_causal_lm_import,
)
