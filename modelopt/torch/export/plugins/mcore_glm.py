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

GLM-5 / GLM-5.2 (``glm_moe_dsa``) is DeepSeek-V3-style MLA with a DSA indexer. It keeps the MTP
layer at HF index ``num_hidden_layers`` under the decoder's names; Megatron-Bridge does not build
it, so the exporter copies it from the source checkpoint.
"""

from .mcore_custom import (
    GroupedGatedMLPSlicing,
    GroupedMLPSlicing,
    NameRemapping,
    SelfAttentionScaling,
)
from .mcore_deepseek import deepseek_causal_lm_export

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
}
