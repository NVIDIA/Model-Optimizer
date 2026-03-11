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

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import ForwardHook


class LayerDescriptor:
    def module_name_regex(self) -> str:
        return ""

    def block_idx_from_module_name(self, module_name: str) -> Optional[int]:
        block_idx_match = re.search(r"\.(\d+)\.", module_name)
        if block_idx_match:
            return int(block_idx_match.group(1))
        return None

    def get_modules_names_to_hook(self, model) -> List[Tuple[int, str]]:
        target_layer = self.module_name_regex()
        if target_layer.startswith("regex:"):
            target_layer_regex = target_layer[len("regex:") :]
            pattern = re.compile(target_layer_regex)
            match_predicate = lambda module_name: pattern.search(module_name)
        else:
            match_predicate = lambda module_name: module_name.endswith(target_layer)

        module_names_to_hook = []
        for module_name, module in model.named_modules():
            if match_predicate(module_name):
                module_names_to_hook.append(
                    (self.block_idx_from_module_name(module_name), module_name)
                )
        return module_names_to_hook


class PruningMixIn(ABC):
    def __init__(self, layer_descriptor: LayerDescriptor):
        self.layer_descriptor = layer_descriptor

    def get_module_names_to_hook(self, model) -> List[Tuple[int, str]]:
        return self.layer_descriptor.get_modules_names_to_hook(model)

    @abstractmethod
    def supported_hooks(self) -> List[Type[ForwardHook]]:
        raise NotImplementedError

    # @abstractmethod
    # def prune_single_layer(
    #         self,
    #         layer_idx: int,
    #         parent_state_dict: dict,
    #         new_state_dict: dict,
    #         original_config: PretrainedConfig,
    #         new_config: PretrainedConfig,
    #         **kwargs
    # ):
    #     raise NotImplementedError
