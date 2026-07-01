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

"""Gemma family export specs."""

from ...model_config import AttentionConfig
from ..base import ModelSpec
from ..hooks import ModelHooks
from ..registry import register


class GemmaHooks(ModelHooks):
    """Gemma2/3 carry extra layernorms; Gemma3 also has per-head q/k norms."""

    def place_submodule(self, name, module, built, layer_config, ctx):
        """Place Gemma2/3 extra layernorms and Gemma3 attention q/k norms."""
        if layer_config.decoder_type in ("gemma2", "gemma3"):
            if "post_attention_layernorm" in name:
                layer_config.post_layernorm = built
                return True
            if "pre_feedforward_layernorm" in name:
                layer_config.pre_feedforward_layernorm = built
                return True
            if "post_feedforward_layernorm" in name:
                layer_config.post_feedforward_layernorm = built
                return True
        if layer_config.decoder_type == "gemma3" and isinstance(built, AttentionConfig):
            layer_config.attention = built
            built.q_layernorm = ctx.build_layernorm_config(module.q_norm)
            built.k_layernorm = ctx.build_layernorm_config(module.k_norm)
            return True
        return False


register(
    ModelSpec(
        name="gemma4_moe",
        # Gemma4 MoE experts are unfused into per-expert nn.Linear layers.
        moe_block_names=("Gemma4TextDecoderLayer",),
        expert_linear_names=("gate_proj", "down_proj", "up_proj"),
        has_iterable_experts=True,
    )
)

# Dense Gemma 1/2/3: shared embedding/output table, plus Gemma2/3 layernorm layout.
register(
    ModelSpec(
        name="gemma",
        decoder_types=("gemma", "gemma2", "gemma3"),
        force_share_embedding_table=True,
        hooks=GemmaHooks(),
    )
)
