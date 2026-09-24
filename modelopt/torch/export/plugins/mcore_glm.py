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

"""Custom mappings from Megatron Core models to GLM-5.x Hugging Face models.

GLM-5 / GLM-5.2 (``glm_moe_dsa``) is DeepSeek-V3-style MLA with a DSA indexer.

For GLM-5.3-Flash (``glm5_next``), Megatron-Bridge builds each HF decoder layer as two physical
layers wrapped in mHC hyper-connections: attention (KDA or NoPE-MLA DSA) at ``2N`` and the dense MLP /
MoE at ``2N+1``. The vision tower is copied from HF.

Both keep the MTP layer at HF index ``num_hidden_layers`` under the decoder's names.
"""

from .mcore_custom import (
    GatedMLPSlicing,
    GroupedGatedMLPSlicing,
    GroupedMLPSlicing,
    KimiDeltaAttentionSlicing,
    NameRemapping,
    SelfAttentionScaling,
    with_language_model_prefix,
)
from .mcore_deepseek import deepseek_causal_lm_export

# Vision-tower weights copied straight from the HF checkpoint (never quantized).
GLM5_NEXT_VISION_PREFIXES = ("model.visual.",)

_glm5_next_causal_lm_export: dict = {
    # Layer-level flags read by the exporter (see the module docstring).
    "fold_attn_mlp_layer_pairs": True,
    "mtp_in_decoder_layers": True,
    "word_embeddings": NameRemapping("model.embed_tokens."),
    "final_norm": NameRemapping("model.norm."),
    "output_layer": NameRemapping("lm_head."),
    # mHC hyper-connections: formatted with (hf_layer_id, "attn" | "ffn").
    "hc_fn": NameRemapping("model.layers.{}.hc_{}_fn"),
    "hc_base": NameRemapping("model.layers.{}.hc_{}_base"),
    "hc_scale": NameRemapping("model.layers.{}.hc_{}_scale"),
    "input_layernorm": NameRemapping("model.layers.{}.input_layernorm."),
    # KDA linear attention (fused q|k|v in_proj and conv1d are split by ``kda``).
    "kda": KimiDeltaAttentionSlicing("model.layers.{}.self_attn."),
    "kda.beta_proj": NameRemapping("model.layers.{}.self_attn.b_proj."),
    "kda.f_a_proj": NameRemapping("model.layers.{}.self_attn.f_a_proj."),
    "kda.f_b_proj": NameRemapping("model.layers.{}.self_attn.f_b_proj."),
    "kda.g_a_proj": NameRemapping("model.layers.{}.self_attn.g_a_proj."),
    "kda.g_b_proj": NameRemapping("model.layers.{}.self_attn.g_b_proj."),
    "kda.A_log": NameRemapping("model.layers.{}.self_attn.A_log"),
    "kda.dt_bias": NameRemapping("model.layers.{}.self_attn.dt_bias"),
    "kda.out_norm": NameRemapping("model.layers.{}.self_attn.o_norm."),
    "kda.out_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    # NoPE MLA with the DSA kpool indexer
    "linear_q_down_proj": NameRemapping("model.layers.{}.self_attn.q_a_proj."),
    "linear_q_layernorm": NameRemapping("model.layers.{}.self_attn.q_a_layernorm."),
    "linear_q_up_proj": NameRemapping("model.layers.{}.self_attn.q_b_proj."),
    "linear_kv_down_proj": NameRemapping("model.layers.{}.self_attn.kv_a_proj_with_mqa."),
    "linear_kv_layernorm": NameRemapping("model.layers.{}.self_attn.kv_a_layernorm."),
    "linear_kv_up_proj": NameRemapping("model.layers.{}.self_attn.kv_b_proj."),
    "linear_proj": NameRemapping("model.layers.{}.self_attn.o_proj."),
    "core_attention": SelfAttentionScaling("model.layers.{}.self_attn."),
    "indexer.linear_wq_b": NameRemapping("model.layers.{}.self_attn.indexer.wq_b."),
    "indexer.linear_wk": NameRemapping("model.layers.{}.self_attn.indexer.wk."),
    "indexer.k_norm": NameRemapping("model.layers.{}.self_attn.indexer.k_norm."),
    "indexer.linear_weights_proj": NameRemapping("model.layers.{}.self_attn.indexer.weights_proj."),
    "indexer.index_kpool_compress_ape": NameRemapping(
        "model.layers.{}.self_attn.indexer.index_kpool_compress_ape"
    ),
    "indexer.index_kpool_compress_gate": NameRemapping(
        "model.layers.{}.self_attn.indexer.index_kpool_compress_gate"
    ),
    # Dense MLP (the pre-MLP norm is fused into linear_fc1)
    "pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm."),
    "fused_pre_mlp_layernorm": NameRemapping("model.layers.{}.post_attention_layernorm.weight"),
    "linear_fc1": GatedMLPSlicing("model.layers.{}.mlp."),
    "linear_fc2": NameRemapping("model.layers.{}.mlp.down_proj."),
    # MoE
    "router": NameRemapping(
        "model.layers.{}.mlp.gate.", {"mapping": {"expert_bias": "e_score_correction_bias"}}
    ),
    "shared_experts.linear_fc1": GatedMLPSlicing("model.layers.{}.mlp.shared_experts."),
    "shared_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.shared_experts.down_proj."),
    "local_experts.linear_fc1": GatedMLPSlicing("model.layers.{}.mlp.experts.{}."),
    "local_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.experts.{}.down_proj."),
    "experts.linear_fc1": GroupedGatedMLPSlicing("model.layers.{}.mlp.experts.{{}}"),
    "experts.linear_fc2": GroupedMLPSlicing("model.layers.{}.mlp.experts.{{}}.down_proj"),
    # MTP (split e_proj / h_proj are concatenated back into eh_proj)
    "mtp.enorm": NameRemapping("model.layers.{}.enorm."),
    "mtp.hnorm": NameRemapping("model.layers.{}.hnorm."),
    "mtp.eh_proj": NameRemapping("model.layers.{}.eh_proj.weight"),
    "mtp.final_layernorm": NameRemapping("model.layers.{}.shared_head.norm."),
}

glm5_next_causal_lm_export = with_language_model_prefix(_glm5_next_causal_lm_export)

glm_moe_dsa_causal_lm_export: dict = {
    **deepseek_causal_lm_export,
    "mtp_in_decoder_layers": True,
    "core_attention": SelfAttentionScaling("model.layers.{}.self_attn."),
    "indexer.linear_wq_b": NameRemapping("model.layers.{}.self_attn.indexer.wq_b."),
    "indexer.linear_wk": NameRemapping("model.layers.{}.self_attn.indexer.wk."),
    "indexer.k_norm": NameRemapping("model.layers.{}.self_attn.indexer.k_norm."),
    "indexer.linear_weights_proj": NameRemapping("model.layers.{}.self_attn.indexer.weights_proj."),
    "experts.linear_fc1": GroupedGatedMLPSlicing("model.layers.{}.mlp.experts.{{}}"),
    "experts.linear_fc2": GroupedMLPSlicing("model.layers.{}.mlp.experts.{{}}.down_proj"),
    "mtp.enorm": NameRemapping("model.layers.{}.enorm."),
    "mtp.hnorm": NameRemapping("model.layers.{}.hnorm."),
    "mtp.eh_proj": NameRemapping("model.layers.{}.eh_proj."),
    "mtp.final_layernorm": NameRemapping("model.layers.{}.shared_head.norm."),
}
