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
# mypy: ignore-errors
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type

from transformers import PretrainedConfig

from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import (
    ForwardHook,
    IndependentKvHeadContributionHook,
)
from modelopt.torch.puzzletron.pruning.pruning_mixin import LayerDescriptor, PruningMixIn
from modelopt.torch.puzzletron.pruning.pruning_utils import (
    GQAInitMode,
    _init_attention_biases,
    _init_attention_weights,
)


@dataclass
class KVHeadsLayerDescriptor(LayerDescriptor):
    o_proj_name: str
    attn_prefix_name: str
    qkvo_weight_names: List[str] = field(default_factory=list)

    def module_name_regex(self) -> str:
        return self.o_proj_name

    def attn_prefix(self, layer_idx: int) -> str:
        return self.attn_prefix_name.format(layer_idx=layer_idx)


class KVHeadsPruningMixIn(PruningMixIn):
    def __init__(self, layer_descriptor: KVHeadsLayerDescriptor):
        assert isinstance(layer_descriptor, KVHeadsLayerDescriptor)
        super().__init__(layer_descriptor)

    def supported_hooks(self) -> List[Type[ForwardHook]]:
        return [IndependentKvHeadContributionHook]

    def prune_single_layer(
        self,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        keys: dict,
        keys_to_remove: dict,
        **kwargs,
    ):
        layer_out_state_dict = {}

        attn_prefix = self.layer_descriptor.attn_prefix(layer_idx)
        q_name, k_name, v_name, o_name = [
            f"{attn_prefix}.{proj_name}" for proj_name in self.layer_descriptor.qkvo_weight_names
        ]

        head_size = new_config.head_dim
        for part in ["weight", "bias"]:
            attn_keys = [f"{name}.{part}" for name in [q_name, k_name, v_name, o_name]]
            q_key, k_key, v_key, o_key = attn_keys

            # Drop attn keys that don't exist and required to be in the new state_dict
            attn_keys = [key for key in attn_keys if key in new_state_dict.keys()]
            if len(attn_keys) > 0 and all(key in keys for key in attn_keys):
                for key in attn_keys:
                    keys_to_remove[key] = keys[key]
                is_student_and_teacher_have_same_attention_implementation = all(
                    key in new_state_dict.keys() for key in attn_keys
                )
                if is_student_and_teacher_have_same_attention_implementation:
                    if part == "weight":
                        wq, wk, wv, wo = _init_attention_weights(
                            gqa_init_mode=gqa_init_mode,
                            layer_idx=layer_idx,
                            new_state_dict=new_state_dict,
                            new_config=new_config,
                            original_state_dict=parent_state_dict,
                            original_config=original_config,
                            q_key=q_key,
                            k_key=k_key,
                            v_key=v_key,
                            o_key=o_key,
                            is_original_mha=is_original_mha,
                            head_size=head_size,
                            mlp_init_config=mlp_init_config,
                        )
                        layer_out_state_dict[q_key], layer_out_state_dict[k_key] = wq, wk
                        layer_out_state_dict[v_key], layer_out_state_dict[o_key] = wv, wo
                    else:
                        bias_sd = _init_attention_biases(
                            gqa_init_mode=gqa_init_mode,
                            layer_idx=layer_idx,
                            new_state_dict=new_state_dict,
                            new_config=new_config,
                            original_state_dict=parent_state_dict,
                            original_config=original_config,
                            q_key=q_key,
                            k_key=k_key,
                            v_key=v_key,
                            o_key=o_key,
                            is_original_mha=is_original_mha,
                            head_size=head_size,
                            mlp_init_config=mlp_init_config,
                        )
                        for bias_key, sd_key in zip("qkvo", [q_key, k_key, v_key, o_key]):
                            if bias_key in bias_sd.keys():
                                layer_out_state_dict[sd_key] = bias_sd[bias_key]

        return layer_out_state_dict
