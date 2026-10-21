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

"""Native DeepSeek V4 HF names for the paired attention/MoE HybridModel.

Bridge supplies import and MXFP4/FP8 decoding. These rules export its floating-point
training weights through the existing packed and folded ModelOpt exporters.
"""

from .mcore_custom import GatedMLPSlicing, NameRemapping

__all__ = ["deepseek_v4_causal_lm_export"]

deepseek_v4_causal_lm_export = {
    "deepseek_v4": True,
    "fold_attn_mlp_layer_pairs": True,
    "word_embeddings": NameRemapping("embed."),
    "final_norm": NameRemapping("norm."),
    "output_layer": NameRemapping("head."),
    "hc_head_fn": NameRemapping("hc_head_fn"),
    "hc_head_base": NameRemapping("hc_head_base"),
    "hc_head_scale": NameRemapping("hc_head_scale"),
    "hc_fn": NameRemapping("layers.{}.hc_{}_fn"),
    "hc_base": NameRemapping("layers.{}.hc_{}_base"),
    "hc_scale": NameRemapping("layers.{}.hc_{}_scale"),
    "input_layernorm": NameRemapping("layers.{}.attn_norm."),
    "pre_mlp_layernorm": NameRemapping("layers.{}.ffn_norm."),
    "linear_q_down_proj": NameRemapping("layers.{}.attn.wq_a."),
    "linear_q_layernorm": NameRemapping("layers.{}.attn.q_norm."),
    "linear_q_up_proj": NameRemapping("layers.{}.attn.wq_b."),
    "linear_kv_proj": NameRemapping("layers.{}.attn.wkv."),
    "linear_kv_layernorm": NameRemapping("layers.{}.attn.kv_norm."),
    "linear_o_group_proj": NameRemapping("layers.{}.attn.wo_a.weight"),
    "linear_proj": NameRemapping("layers.{}.attn.wo_b."),
    "attn_sink": NameRemapping("layers.{}.attn.attn_sink"),
    "compressor.linear_wkv": NameRemapping("layers.{}.attn.compressor.wkv."),
    "compressor.linear_wgate": NameRemapping("layers.{}.attn.compressor.wgate."),
    "compressor.ape": NameRemapping("layers.{}.attn.compressor.ape"),
    "compressor.norm": NameRemapping("layers.{}.attn.compressor.norm."),
    "indexer.linear_wq_b": NameRemapping("layers.{}.attn.indexer.wq_b."),
    "indexer.linear_weights_proj": NameRemapping("layers.{}.attn.indexer.scorer.weights_proj."),
    "indexer.compressor.linear_wkv": NameRemapping("layers.{}.attn.indexer.compressor.wkv."),
    "indexer.compressor.linear_wgate": NameRemapping("layers.{}.attn.indexer.compressor.wgate."),
    "indexer.compressor.ape": NameRemapping("layers.{}.attn.indexer.compressor.ape"),
    "indexer.compressor.norm": NameRemapping("layers.{}.attn.indexer.compressor.norm."),
    "router": NameRemapping("layers.{}.ffn.gate.", {"mapping": {"expert_bias": "bias"}}),
    "shared_experts.linear_fc1": GatedMLPSlicing(
        "layers.{}.ffn.shared_experts.", {"gate_proj_name": "w1", "up_proj_name": "w3"}
    ),
    "shared_experts.linear_fc2": NameRemapping("layers.{}.ffn.shared_experts.w2."),
    "local_experts.linear_fc1": GatedMLPSlicing(
        "layers.{}.ffn.experts.{}.", {"gate_proj_name": "w1", "up_proj_name": "w3"}
    ),
    "local_experts.linear_fc2": NameRemapping("layers.{}.ffn.experts.{}.w2."),
}
