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

"""Custom mapping from Llama Hugging Face models to Megatron Core models.

Also holds the mappings of the EAGLE draft models exported from them.
"""

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

llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("model.embed_tokens."),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm."),
    "fused_input_layernorm": NameRemapping("model.layers.{}.input_layernorm.weight"),
    "linear_qkv": QKVSlicing("model.layers.{}.self_attn."),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    # KV cache quant export
    "core_attention": SelfAttentionScaling("model.layers.{}.self_attn."),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm."),
    "fused_pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.weight"),
    "linear_fc1": GatedMLPSlicing("model.layers.{}.mlp."),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj."),
    "final_layernorm": NameRemapping("model.norm."),
    "output_layer": NameRemapping("lm_head."),
}

medusa_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    # MedusaForCausalLM support
    "lm_head": NameRemapping(
        "medusa_heads.{}.1."
    ),  # TODO: lm_head is hardcoded to .1 as currently only support using 1 layer in medusa head
    # needs a fix
    "linear": NameRemapping("medusa_heads.{}.{}.linear."),
}

eagle_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("embed_tokens."),
    "enorm": NameRemapping("enorm."),
    "hnorm": NameRemapping("hnorm."),
    "fc": NameRemapping("fc."),
    "first_input_layernorm": NameRemapping("layers.{}.input_layernorm."),
    "input_layernorm": NameRemapping("layers.{}.input_layernorm."),
    "linear_qkv": QKVSlicing("layers.{}.self_attn."),
    "linear_proj": NameRemapping("layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": NameRemapping("layers.{}.post_attention_layernorm."),
    "linear_fc1": GatedMLPSlicing("layers.{}.mlp."),
    "linear_fc2": NameRemapping("layers.{}.mlp.down_proj."),
    "final_layernorm": NameRemapping("norm."),
    "d2t": NameRemapping("d2t"),
    "output_layer": NameRemapping("lm_head."),
    "parallel_draft_heads.medusa_layers": NameRemapping(
        "parallel_draft_heads.{}.medusa_layers.{}.linear."
    ),
    "parallel_draft_heads.lm_head": NameRemapping("parallel_draft_heads.lm_head."),
}

eagle3_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("embed_tokens."),
    "enorm": NameRemapping("midlayer.input_layernorm."),
    "fc": NameRemapping("fc."),
    "first_input_layernorm": NameRemapping("midlayer.hidden_norm."),
    "linear_qkv": QKVSlicing("midlayer.self_attn."),
    "linear_proj": NameRemapping("midlayer.self_attn.o_proj."),
    "pre_mlp_layernorm": NameRemapping("midlayer.post_attention_layernorm."),
    "linear_fc1": GatedMLPSlicing("midlayer.mlp."),
    "linear_fc2": NameRemapping("midlayer.mlp.down_proj."),
    "final_layernorm": NameRemapping("norm."),
    "d2t": NameRemapping("d2t"),
    "output_layer": NameRemapping("lm_head."),
    "parallel_draft_heads.medusa_layers": NameRemapping(
        "parallel_draft_heads.{}.medusa_layers.{}.linear."
    ),
    "parallel_draft_heads.lm_head": NameRemapping("parallel_draft_heads.lm_head."),
}

eagle3_deep_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("embed_tokens."),
    "enorm": NameRemapping("midlayer.0.input_layernorm."),
    "fc": NameRemapping("fc."),
    "first_input_layernorm": NameRemapping("midlayer.0.hidden_norm."),
    "input_layernorm": NameRemapping("midlayer.{}.hidden_norm."),
    "linear_qkv": QKVSlicing("midlayer.{}.self_attn."),
    "linear_proj": NameRemapping("midlayer.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": NameRemapping("midlayer.{}.post_attention_layernorm."),
    "linear_fc1": GatedMLPSlicing("midlayer.{}.mlp."),
    "linear_fc2": NameRemapping("midlayer.{}.mlp.down_proj."),
    "final_layernorm": NameRemapping("norm."),
    "d2t": NameRemapping("d2t"),
    "output_layer": NameRemapping("lm_head."),
    "parallel_draft_heads.medusa_layers": NameRemapping(
        "parallel_draft_heads.{}.medusa_layers.{}.linear."
    ),
    "parallel_draft_heads.lm_head": NameRemapping("parallel_draft_heads.lm_head."),
}

llama_causal_lm_import: dict[str, CustomModuleMapping] = {
    "word_embeddings": NameRemapping("model.embed_tokens.", COL_TP),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm.", REPLICATE),
    # Fused TE spec (TELayerNormColumnParallelLinear) — see qwen3/mcore.py for rationale.
    "fused_input_layernorm": NameRemapping("model.layers.{}.input_layernorm.weight"),
    "linear_qkv": QKVMerging("model.layers.{}.self_attn.", COL_TP),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj.", ROW_TP),
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.", REPLICATE),
    "fused_pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.weight"),
    "linear_fc1": GatedMLPMerging("model.layers.{}.mlp.", COL_TP),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj.", ROW_TP),
    "final_layernorm": NameRemapping("model.norm.", REPLICATE),
    "output_layer": NameRemapping("lm_head.", COL_TP),
}

register_mcore_hf_mapping(
    "LlamaForCausalLM",
    export_mapping=llama_causal_lm_export,
    import_mapping=llama_causal_lm_import,
)
register_mcore_hf_mapping(
    "LlamaForCausalLMEagle",
    export_mapping=eagle_llama_causal_lm_export,
)
register_mcore_hf_mapping(
    "LlamaForCausalLMEagle3",
    export_mapping=eagle3_llama_causal_lm_export,
)
register_mcore_hf_mapping(
    "LlamaForCausalLMEagle3Deep",
    export_mapping=eagle3_deep_llama_causal_lm_export,
)
