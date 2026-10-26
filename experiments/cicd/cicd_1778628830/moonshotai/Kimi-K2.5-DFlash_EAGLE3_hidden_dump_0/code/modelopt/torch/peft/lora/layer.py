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

"""LoRA (Low-Rank Adaptation) module implementation."""

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls

from ..config import PEFTAttributeConfig

__all__ = ["LoRAModule", "LoRAModuleRegistry"]


class LoRAModule(DynamicModule):
    """Base class for LoRA (Low-Rank Adaptation) modules.

    This module wraps existing layers and adds trainable low-rank decomposition
    matrices (LoRA adapters) that are added to the original layer's output.

    Attributes:
        _lora_adapters: Dictionary mapping adapter names to their LoRA A and B matrices
    """

    def _setup(self) -> None:
        """Initialize LoRA-specific attributes."""
        self._lora_adapters: dict[str, dict[str, Any]] = {}

    @property
    def adapter_names(self) -> set:
        """Return the set of all registered adapter names."""
        return set(self._lora_adapters.keys())

    def _register_adapter(
        self,
        adapter_name: str,
        lora_a: nn.Module,
        lora_b: nn.Module,
        rank: int,
        scale: float = 1.0,
        enable: bool = True,
    ) -> None:
        """Register a new LoRA adapter with explicit rank tracking.

        Args:
            adapter_name: Name of the adapter
            lora_a: LoRA A module (down-projection)
            lora_b: LoRA B module (up-projection)
            rank: Rank of the LoRA decomposition
            scale: Scale factor for the LoRA output
        """
        self.add_module(f"lora_a_{adapter_name}", lora_a)
        self.add_module(f"lora_b_{adapter_name}", lora_b)

        # Store in adapter dictionary with explicit rank
        if adapter_name in self._lora_adapters:
            raise ValueError(f"Adapter '{adapter_name}' already exists.")
        self._lora_adapters[adapter_name] = {
            "lora_a": lora_a,
            "lora_b": lora_b,
            "rank": rank,
            "scale": scale,
            "enable": enable,
        }

    @abstractmethod
    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter.

        This method must be implemented by subclasses to create the appropriate
        LoRA A and B matrices for the specific layer type.

        Args:
            adapter_name: Name for the new adapter
            attr_config: PEFTAttributeConfig containing rank, scale, and initialization settings
        """
        raise NotImplementedError("Subclasses must implement update_layer_lora")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor
            *args: Additional positional arguments for the base layer
            **kwargs: Additional keyword arguments for the base layer

        Returns:
            Output from the base layer plus active LoRA adaptations
        """
        output = super().forward(x, *args, **kwargs)

        if isinstance(output, tuple):
            result = output[0]
            other_outputs = output[1:]
        else:
            result = output
            other_outputs = ()

        for adapter_name in self._lora_adapters:
            adapter = self._lora_adapters[adapter_name]
            if adapter["enable"]:
                lora_a = adapter["lora_a"]
                lora_b = adapter["lora_b"]
                lora_a_output = lora_a(x)
                if isinstance(lora_a_output, tuple):
                    lora_a_output = lora_a_output[0]
                lora_b_output = lora_b(lora_a_output)
                if isinstance(lora_b_output, tuple):
                    lora_b_output = lora_b_output[0]
                scale = adapter["scale"]
                result = result + scale * lora_b_output

        if other_outputs:
            return (result, *other_outputs)
        else:
            return result


LoRAModuleRegistry = _DMRegistryCls("LoRA", LoRAModule)
