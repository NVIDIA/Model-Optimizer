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

import importlib
import inspect
import pkgutil
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Type

import torch.nn as nn

from ....block_config import BlockConfig
from ....pruning.expert_removal_pruning_mixin import (
    ExpertRemovalLayerDescriptor,
    ExpertRemovalPruningMixIn,
)
from ....pruning.pruning_mixin import PruningMixIn
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer.no_op import MatchingZeros, Same

__all__ = ["NemotronHExpertRemovalLayerDescriptor", "NemotronHModelDescriptor"]


def get_dynamic_modules(module_cls_str: str) -> List[Type[nn.Module]]:
    import transformers_modules

    matches = []
    for finder, modname, ispkg in pkgutil.walk_packages(
        transformers_modules.__path__, transformers_modules.__name__ + "."
    ):
        module = importlib.import_module(modname)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__name__ == module_cls_str:
                matches.append(obj)

    return matches


@dataclass
class NemotronHExpertRemovalLayerDescriptor(ExpertRemovalLayerDescriptor):
    target_name: str = "mixer.gate"
    moe_prefix_name: str = "backbone.layers.{layer_idx}.mixer"
    expert_prefix_name: str = "experts.{expert_idx}"
    router_weights: List[str] = field(default_factory=lambda: ["gate.weight"])
    router_biases: List[str] = field(default_factory=lambda: ["gate.e_score_correction_bias"])
    expert_weights: List[str] = field(
        default_factory=lambda: ["up_proj.weight", "down_proj.weight"]
    )

    def get_modules_names_to_hook(self, model) -> List[Tuple[int, str]]:
        if self.target_name != "mixer":
            return super().get_modules_names_to_hook(model)

        # when target is `mixer` we'll target moe layers of class type: `NemotronHMOE`, as NemotronH models use auto-map we'll check for class name instead of class type.
        target_class_name = "NemotronHMOE"

        module_names_to_hook = []
        for module_name, module in model.named_modules():
            # restrict to attributes called "mixer" and with the desired class name
            if (
                module_name.endswith(self.target_name)
                and module.__class__.__name__ == target_class_name
            ):
                module_names_to_hook.append(
                    (self.block_idx_from_module_name(module_name), module_name)
                )
        return module_names_to_hook


@ModelDescriptorFactory.register_decorator("nemotron_h")
class NemotronHModelDescriptor(ModelDescriptor):
    _DECODER_LAYER_CLS: Type[nn.Module] = None

    @staticmethod
    def decoder_layer_cls():
        decoder_cls_list = get_dynamic_modules("NemotronHBlock")
        if not decoder_cls_list:
            raise AssertionError(
                "NemotronH contains dynamic modules that should be cached beforehand, make sure to load your config using `load_model_config` or manually call `force_cache_dynamic_modules(config, checkpoint_dir)`"
            )
        return decoder_cls_list

    @staticmethod
    def requires_trust_remote_code() -> bool:
        return True

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        override_kwargs = {}
        if block_config.ffn.intermediate_size is not None:
            override_kwargs["intermediate_size"] = block_config.ffn.intermediate_size

        if block_config.attention.num_key_value_heads is not None:
            override_kwargs["num_key_value_heads"] = block_config.attention.num_key_value_heads

        if block_config.ffn.moe is not None:
            override_kwargs["moe_intermediate_size"] = block_config.ffn.moe.expert_intermediate_dim
            override_kwargs["n_routed_experts"] = block_config.ffn.moe.num_local_experts

        return override_kwargs

    @staticmethod
    def _block_no_op_post_init(decoder_layer):
        """
        Due to the subblock structure of NemotronH always one of the subblock is set to no-op, for a real no-op both attention & ffn no-op should be set to True.
        """
        block_config = decoder_layer.config.block_configs[decoder_layer.layer_idx]
        if block_config.ffn.no_op and block_config.attention.no_op:
            decoder_layer.norm = Same()
            decoder_layer.mixer = MatchingZeros()

    @staticmethod
    def attn_no_op_post_init(decoder_layer):
        NemotronHModelDescriptor._block_no_op_post_init(decoder_layer)

    @staticmethod
    def mlp_no_op_post_init(decoder_layer):
        NemotronHModelDescriptor._block_no_op_post_init(decoder_layer)

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        dummy_block = super().create_dummy_block(original_layer, block_index)
        # Required by `NemotronHModel.forward`.
        dummy_block.block_type = original_layer.block_type
        # Preserve layer_idx if it exists (used by _block_no_op_post_init)
        if hasattr(original_layer, "layer_idx"):
            dummy_block.layer_idx = original_layer.layer_idx
        # Preserve config if it exists (used by _block_no_op_post_init to access block_configs)
        if hasattr(original_layer, "config"):
            dummy_block.config = original_layer.config
        return dummy_block

    @staticmethod
    def init_rotary_embedding(model, runtime):
        """
        NemotronH has no positional embeddings
        """

    @staticmethod
    def input_embedding_name():
        return "backbone.embeddings"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "backbone.norm_f"

    @staticmethod
    def layer_block_name(index: int):
        return f"backbone.layers.{index}"

    @classmethod
    def get_weight_groups(
        cls, layer_names: Iterable[str], num_hidden_layers: int
    ) -> Dict[str, List[str]]:
        """
        Problem with NemotronH is that `norm.weight` can be in both block_{i}_ffn and block_{i}_attention. duplicate groups with `norm.weight` should be removed.
        """
        weight_groups = defaultdict(list)
        for name in layer_names:
            is_matched = False
            for group, pattern in cls.layer_name_predicates(num_hidden_layers).items():
                if pattern.match(name):
                    weight_groups[group].append(name)
                    is_matched = True
            if not is_matched:
                raise ValueError(f"Couldn't find a match for {name}")

        valid_weight_groups = {}
        for group, names in weight_groups.items():
            if len(names) == 1:
                only_name = names[0]
                if only_name.endswith("norm.weight") and "layers" in only_name:
                    # Skip and don't append this group to valid_weight_groups
                    continue
            valid_weight_groups[group] = names

        return valid_weight_groups

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        layer_name_patterns = {
            "embeddings": re.compile(
                r"^(model\.embed_tokens\.weight|backbone\.embeddings?\.weight)$"
            ),
            "lm_head": re.compile(r"^(lm_head\.weight|backbone\.norm_f\.weight)$"),
        }

        def build_ffn_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_ffn": re.compile(
                    rf"^backbone\.layers\.{layer_idx}\."
                    r"(norm\.weight|"  # ← INCLUDED IN FFN
                    r"mixer\.(gate\.e_score_correction_bias"
                    r"|gate\.weight"
                    r"|experts\.\d+\.up_proj\.weight"
                    r"|experts\.\d+\.down_proj\.weight"
                    r"|shared_experts\.up_proj\.weight"
                    r"|shared_experts\.down_proj\.weight))$"
                )
                for layer_idx in range(num_layers)
            }

        def build_attention_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_attention": re.compile(
                    rf"^backbone\.layers\.{layer_idx}\."
                    r"(norm\.weight|"  # ← INCLUDED IN ATTENTION
                    r"mixer\.(norm\.weight"
                    r"|A_log"
                    r"|D"
                    r"|conv1d\.weight"
                    r"|conv1d\.bias"
                    r"|dt_bias"
                    r"|in_proj\.weight"
                    r"|out_proj\.weight"
                    r"|q_proj\.weight"
                    r"|k_proj\.weight"
                    r"|v_proj\.weight"
                    r"|o_proj\.weight))$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(
            **build_ffn_predicates(),
            **build_attention_predicates(),
        )

        return layer_name_patterns

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "experts_removal": ExpertRemovalPruningMixIn(NemotronHExpertRemovalLayerDescriptor()),
        }
