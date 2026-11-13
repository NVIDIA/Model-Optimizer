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

from modelopt.torch.export.plugins.mcore_custom import (
    CustomModuleMapping,
    NameRemapping,
    QKVSlicing,
    SelfAttentionScaling,
    register_mcore_hf_mapping,
)

__all__: list[str] = []

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

register_mcore_hf_mapping(
    "NemotronForCausalLM",
    export_mapping=nemotron_causal_lm_export,
)
