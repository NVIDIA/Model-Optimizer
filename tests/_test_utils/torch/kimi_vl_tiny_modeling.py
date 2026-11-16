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

"""A tiny Kimi-VL-shaped model, as remote code: tests copy it into a checkpoint as ``modeling_kimi_vl.py``.

It has what the Kimi-VL adapter patches, as the checkpoint's own code has it: a VLM class that builds
its language model, at construction, from the module-level name ``DeepseekV3ForCausalLM``, beside a
vision tower and a projector of its own, and whose ``tie_weights`` takes no arguments. The name is
the checkpoint's own DeepSeek-V3, whose MoE holds its routed experts as per-expert Linears that expert
parallelism cannot shard: only a swap that misses builds it, and a test then meets its layout.
"""

from torch import nn
from transformers import DeepseekV3Config, PretrainedConfig, PreTrainedModel
from transformers import DeepseekV3ForCausalLM as _NativeDeepseekV3ForCausalLM


class DeepseekV3ForCausalLM(_NativeDeepseekV3ForCausalLM):
    """The checkpoint's own: routed experts as per-expert Linears, which expert parallelism cannot shard."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            if hasattr(layer.mlp, "experts"):
                layer.mlp.experts = nn.ModuleList(
                    nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
                    for _ in range(config.n_routed_experts)
                )


class KimiVLConfig(PretrainedConfig):
    model_type = "kimi_vl"
    sub_configs = {"text_config": DeepseekV3Config}

    def __init__(self, text_config=None, vision_hidden_size=16, **kwargs):
        if text_config is None or isinstance(text_config, dict):
            text_config = DeepseekV3Config(**(text_config or {}))
        self.text_config = text_config
        self.vision_hidden_size = vision_hidden_size
        super().__init__(**kwargs)


class KimiVLForConditionalGeneration(PreTrainedModel):
    config_class = KimiVLConfig
    _no_split_modules = ["DeepseekV3DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        width = config.vision_hidden_size
        self.vision_tower = nn.Linear(width, width)
        self.multi_modal_projector = nn.Linear(width, config.text_config.hidden_size)
        self.language_model = DeepseekV3ForCausalLM(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(self, input_ids=None, **kwargs):
        return self.language_model(input_ids=input_ids, **kwargs)

    def tie_weights(self):  # as the checkpoint's own: no keyword arguments
        return self.language_model.tie_weights()
