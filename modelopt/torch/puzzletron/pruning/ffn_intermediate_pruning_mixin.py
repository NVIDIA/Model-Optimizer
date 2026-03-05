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
from typing import Any, Dict, List, Optional, Type

import torch
from transformers import PretrainedConfig

from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import (
    ForwardHook,
    IndependentChannelContributionHook,
    IterativeChannelContributionHook,
)
from modelopt.torch.puzzletron.pruning.pruning_mixin import LayerDescriptor, PruningMixIn
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    MlpInitMode,
    _init_mlp_module,
)


@dataclass
class FFNIntermediateLayerDescriptor(LayerDescriptor):
    down_proj_name: str
    ffn_prefix_name: str
    linear_weight_names: List[str] = field(default_factory=list)

    def module_name_regex(self) -> str:
        return self.down_proj_name

    def ffn_prefix(self, layer_idx: int) -> str:
        return self.ffn_prefix_name.format(layer_idx=layer_idx)


class FFNIntermediatePruningMixIn(PruningMixIn):
    def __init__(self, layer_descriptor: FFNIntermediateLayerDescriptor):
        assert isinstance(layer_descriptor, FFNIntermediateLayerDescriptor)
        super().__init__(layer_descriptor)

    def supported_hooks(self) -> List[Type[ForwardHook]]:
        return [IndependentChannelContributionHook, IterativeChannelContributionHook]

    def prune_single_layer(
        self,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        mlp_init_mode: MlpInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        keys: dict,
        keys_to_remove: dict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        layer_out_state_dict = {}
        # Hardcoded strings
        mlp_prefix = self.layer_descriptor.ffn_prefix(layer_idx)
        mlp_key_names = [
            f"{mlp_prefix}.{name}.weight" for name in self.layer_descriptor.linear_weight_names
        ]
        mlp_keys = [keys.get(module_name) for module_name in mlp_key_names]
        mlp_keys = [k for k in mlp_keys if k is not None]

        for key in mlp_keys:
            keys_to_remove[f"{mlp_prefix}.{key.split('.')[-2]}.weight"] = key

        pruned_filters = None
        projection_matrix = None

        for mlp_key in mlp_keys:
            expanded_dim = 1 if self.layer_descriptor.down_proj_name in mlp_key else 0
            if mlp_key in new_state_dict.keys():
                mlp_module_weight, pruned_filters, projection_matrix = _init_mlp_module(
                    mlp_init_mode,
                    mlp_prefix,
                    expanded_dim,
                    layer_idx,
                    new_state_dict[mlp_key],
                    new_config,
                    parent_state_dict[mlp_key],
                    original_config,
                    mlp_init_config,
                    pruned_filters,
                    projection_matrix,
                )
                layer_out_state_dict[mlp_key] = mlp_module_weight

        return layer_out_state_dict
