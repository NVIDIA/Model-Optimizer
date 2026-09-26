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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from transformers import PretrainedConfig

from modelopt.torch.prune.importance_hooks.base_hooks import ForwardHook
from modelopt.torch.prune.importance_hooks.expert_removal_hooks import (
    NemotronHRemoveExpertsIndependentHook,
    Qwen3VLRemoveExpertsIndependentHook,
    RankedChoiceVotingHook,
    RankedChoiceVotingHookNemotronH,
)

from .pruning_mixin import LayerDescriptor, PruningMixIn
from .pruning_utils import MlpInitMode, _init_moe_module

__all__ = [
    "ExpertRemovalLayerDescriptor",
    "ExpertRemovalPruningMixIn",
]


@dataclass
class ExpertRemovalLayerDescriptor(LayerDescriptor):
    """Descriptor for expert-removal pruning layers."""

    # TODO: Add shared expert weights in case it's prunable.
    # TODO: Consider removing the segmentation between weight and bias.

    #: Module name for hook registration; supports ``regex:`` prefix.
    target_name: str
    #: MoE prefix layer name with ``{layer_idx}`` placeholder,
    #: e.g. ``model.layers.{layer_idx}.moe``.
    moe_prefix_name: str
    #: Expert prefix relative to *moe_prefix* with ``{expert_idx}`` placeholder,
    #: e.g. ``experts.{expert_idx}``.
    expert_prefix_name: str = ""
    #: Router weight names relative to *moe_prefix*.
    router_weights: List[str] = field(default_factory=list)
    #: Router bias names relative to *moe_prefix*.
    router_biases: List[str] = field(default_factory=list)
    #: Per-expert weight names relative to *expert_prefix* (per-expert format).
    expert_weights: List[str] = field(default_factory=list)
    #: Per-expert bias names relative to *expert_prefix* (per-expert format).
    expert_biases: List[str] = field(default_factory=list)
    #: If ``True``, experts are stored as single fused tensors (shape ``[num_experts, ...]``).
    is_fused_experts: bool = False
    #: Fused expert weight names relative to *moe_prefix*,
    #: e.g. ``["experts.gate_up_proj", "experts.down_proj"]``.
    fused_expert_weights: List[str] = field(default_factory=list)

    def module_name_regex(self) -> str:
        return self.target_name

    def moe_prefix(self, layer_idx: int) -> str:
        return self.moe_prefix_name.format(layer_idx=layer_idx)

    def expert_prefix(self, layer_idx: int, expert_idx: int) -> str:
        _expert_prefix = self.moe_prefix_name + "." + self.expert_prefix_name
        return _expert_prefix.format(layer_idx=layer_idx, expert_idx=expert_idx)


class ExpertRemovalPruningMixIn(PruningMixIn):
    def __init__(self, layer_descriptor: ExpertRemovalLayerDescriptor):
        assert isinstance(layer_descriptor, ExpertRemovalLayerDescriptor)
        super().__init__(layer_descriptor)

    def supported_hooks(self) -> List[Type[ForwardHook]]:
        return [
            RankedChoiceVotingHook,
            RankedChoiceVotingHookNemotronH,
            NemotronHRemoveExpertsIndependentHook,
            Qwen3VLRemoveExpertsIndependentHook,
        ]

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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        layer_out_state_dict = {}

        child_block_config = new_config.block_configs[layer_idx]
        parent_block_config = original_config.block_configs[layer_idx]

        if not parent_block_config.ffn.is_moe:
            return layer_out_state_dict

        new_num_experts = child_block_config.ffn.moe.num_local_experts
        orig_num_experts = parent_block_config.ffn.moe.num_local_experts

        child_router_keys, new_experts_keys = self._generate_moe_keys(layer_idx, new_num_experts)
        parent_router_keys, orig_experts_keys = self._generate_moe_keys(layer_idx, orig_num_experts)

        # Pop parent's router keys from copy list; child-only router keys will be initialized below
        for rk in sum(parent_router_keys.values(), []):
            if rk in keys:
                keys.pop(rk)
        for key in sum(orig_experts_keys.values(), []):
            if key in keys:
                keys.pop(key)

        if self.layer_descriptor.is_fused_experts:
            # Fused format: unbundle single tensor [num_experts, ...] into list of per-expert tensors
            orig_experts_weights = {}
            for name, fused_keys in orig_experts_keys.items():
                fused_tensor = parent_state_dict[fused_keys[0]]  # Single fused tensor
                orig_experts_weights[name] = [fused_tensor[i] for i in range(orig_num_experts)]

            new_experts_weights = {}
            for name, fused_keys in new_experts_keys.items():
                fused_tensor = new_state_dict[fused_keys[0]]  # Single fused tensor
                new_experts_weights[name] = [fused_tensor[i] for i in range(new_num_experts)]
        else:
            # Per-expert format: load each expert tensor separately
            orig_experts_weights = {
                name: [parent_state_dict[key] for key in orig_experts_module_keys]
                for name, orig_experts_module_keys in orig_experts_keys.items()
            }
            new_experts_weights = {
                name: [new_state_dict[key] for key in new_experts_module_keys]
                for name, new_experts_module_keys in new_experts_keys.items()
            }

        orig_router_weights = {
            name: [parent_state_dict[key] for key in _module_router_keys]
            for name, _module_router_keys in parent_router_keys.items()
        }
        new_router_weights = {
            name: [new_state_dict[key] for key in _module_router_keys]
            for name, _module_router_keys in child_router_keys.items()
        }

        out_router_weights, out_experts_weights = _init_moe_module(
            layer_idx=layer_idx,
            mlp_init_mode=mlp_init_mode,
            mlp_init_config=mlp_init_config,
            orig_router_weights=orig_router_weights,
            orig_experts_weights=orig_experts_weights,
            new_router_weights=new_router_weights,
            new_experts_weights=new_experts_weights,
            orig_num_experts=orig_num_experts,
            new_num_experts=new_num_experts,
        )
        assert new_experts_keys.keys() == out_experts_weights.keys(), (
            "new_experts_keys and out_experts_weights must have the same keys"
        )
        assert child_router_keys.keys() == out_router_weights.keys(), (
            "child_router_keys and out_router_weights must have the same keys"
        )

        for name in child_router_keys.keys():
            layer_out_state_dict.update(zip(child_router_keys[name], out_router_weights[name]))

        if self.layer_descriptor.is_fused_experts:
            # Fused format: rebundle list of per-expert tensors into single fused tensor
            for name in new_experts_keys.keys():
                fused_key = new_experts_keys[name][0]  # Single key for fused tensor
                fused_tensor = torch.stack(out_experts_weights[name], dim=0)  # [num_experts, ...]
                layer_out_state_dict[fused_key] = fused_tensor
        else:
            # Per-expert format: each expert has its own key
            for name in new_experts_keys.keys():
                layer_out_state_dict.update(zip(new_experts_keys[name], out_experts_weights[name]))

        return layer_out_state_dict

    def _generate_moe_keys(
        self, layer_idx: int, num_experts: int
    ) -> Tuple[Dict[str, List[str]], dict[str, list[str]]]:
        """
        Generate MoE weight keys for router and experts.
        TODO simplify or better define the data structure of the moe keys returned.

        :return: tuple of router_keys and expert_keys, all <weight_names> are absolute names relative to the model root:
            * router_keys structure:
                {"weight: [<weight_names>], bias: [<weight_names>]"}
            * expert_keys structure (per-expert format):
                {"<expert-key-name>: [<all_experts_weight_names>]}
                i.e:
                {
                    "down_proj.weight": ["model...experts.0.down_proj.weight", ..., "model...experts.N.down_proj.weight"],
                    ...
                }
            * expert_keys structure (fused format):
                {"<fused-key-name>: [<single_fused_weight_name>]}
                i.e:
                {
                    "experts.gate_up_proj": ["model...experts.gate_up_proj"],
                    "experts.down_proj": ["model...experts.down_proj"],
                }
        """
        self.layer_descriptor: ExpertRemovalLayerDescriptor
        moe_prefix = self.layer_descriptor.moe_prefix(layer_idx)

        router_keys = {
            "weight": [
                f"{moe_prefix}.{_weight}" for _weight in self.layer_descriptor.router_weights
            ],
            "bias": [f"{moe_prefix}.{_bias}" for _bias in self.layer_descriptor.router_biases],
        }

        if self.layer_descriptor.is_fused_experts:
            # Fused format: single tensor per weight type with shape [num_experts, ...]
            experts_module_names = {}
            for fused_weight in self.layer_descriptor.fused_expert_weights:
                experts_module_names[fused_weight] = [f"{moe_prefix}.{fused_weight}"]
        else:
            # Per-expert format: separate tensor for each expert
            expert_key_names = (
                self.layer_descriptor.expert_weights + self.layer_descriptor.expert_biases
            )
            experts_module_names = {}
            for key_name in expert_key_names:
                experts_module_names[key_name] = [
                    f"{self.layer_descriptor.expert_prefix(layer_idx, expert_idx)}.{key_name}"
                    for expert_idx in range(num_experts)
                ]

        return router_keys, experts_module_names
