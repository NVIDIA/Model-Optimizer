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

"""Megatron-Core specific PEFT/LoRA plugins."""

import torch
import torch.nn as nn
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.megatron import (
    _MegatronColumnParallelLinear as QuantColumnParallelLinear,
)
from modelopt.torch.quantization.plugins.megatron import (
    _MegatronRowParallelLinear as QuantRowParallelLinear,
)

from ...config import PEFTAttributeConfig
from ...custom import CUSTOM_MODEL_PLUGINS
from ..layer import LoRAModule, LoRAModuleRegistry

DEFAULT_LORA_RANK = 64
DEFAULT_SCALE = 1.0

__all__ = []


def megatron_replace_lora_module_hook(model: torch.nn.Module):
    """Configure Megatron-Core model PEFT/LoRA support.

    This callback is called before the LoRAModule replacement to configure
    distributed checkpointing support. For each MegatronModule:
    1. We enable heterogeneous distributed checkpointing

    Note: LoRAModule already has built-in get_extra_state and set_extra_state methods,
    so we don't need to register callbacks for them.
    """
    for name, module in model.named_modules():
        if isinstance(module, MegatronModule):
            # Enable heterogeneous distributed checkpointing
            if hasattr(module, "config") and hasattr(
                module.config, "hetereogenous_dist_checkpoint"
            ):
                module.config.hetereogenous_dist_checkpoint = True


# Register the hook
CUSTOM_MODEL_PLUGINS.add(megatron_replace_lora_module_hook)


class _MegatronParallelLoRABase(LoRAModule):
    """Base class for Megatron tensor parallel LoRA implementations.

    This class provides common functionality for both ColumnParallel and RowParallel
    LoRA implementations, reducing code duplication.
    """

    def _register_adapter_with_device(
        self,
        adapter_name: str,
        lora_a: nn.Module,
        lora_b: nn.Module,
        rank: int,
        scale: float,
        enable: bool,
    ) -> None:
        """Register LoRA adapter modules and ensure correct device placement.

        Args:
            adapter_name: Name of the adapter
            lora_a: LoRA A module (down-projection)
            lora_b: LoRA B module (up-projection)
            rank: Rank of the LoRA decomposition
        """
        # Move LoRA modules to the same device and dtype as the parent module
        # Try to get device and dtype from parent module's parameters or buffers
        device = None
        dtype = None
        for p in self.parameters():
            device = p.device
            dtype = p.dtype
            break
        if device is None:
            for b in self.buffers():
                device = b.device
                dtype = b.dtype
                break

        # If we found a device and dtype, move LoRA modules to match
        if device is not None:
            lora_a = lora_a.to(device)
            lora_b = lora_b.to(device)
        if dtype is not None:
            lora_a = lora_a.to(dtype)
            lora_b = lora_b.to(dtype)

        super()._register_adapter(adapter_name, lora_a, lora_b, rank, scale, enable)


@LoRAModuleRegistry.register({ColumnParallelLinear: "megatron_ColumnParallelLinear"})
class _LoRAMegatronColumnParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron ColumnParallelLinear layers.

    This implementation creates column-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter for ColumnParallelLinear.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a = nn.Linear(
            in_features=self.input_size,
            out_features=attr_config.rank,
            bias=False,
        )
        with torch.no_grad():
            attr_config.lora_a_init(lora_a.weight)

        lora_b = ColumnParallelLinear(
            attr_config.rank,
            self.output_size,
            config=self.config,
            bias=False,
            gather_output=False,
            init_method=attr_config.lora_b_init,
        )

        self._register_adapter_with_device(
            adapter_name, lora_a, lora_b, attr_config.rank, attr_config.scale, attr_config.enable
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0 for ColumnParallelLinear, bias not sharded.

        For ColumnParallelLinear:
        (lora_a is a regular nn.Linear and is not sharded)
        - lora_b weight: sharded at dim 0
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        if hasattr(self, "_lora_adapters"):
            for adapter_name in self._lora_adapters:
                lora_b_name = f"lora_b_{adapter_name}"
                lora_b = self._lora_adapters[adapter_name]["lora_b"]

                assert isinstance(lora_b, ColumnParallelLinear)
                lora_b_sharded = lora_b.sharded_state_dict(
                    prefix=f"{prefix}{lora_b_name}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(lora_b_sharded)

        return sharded_state_dict


@LoRAModuleRegistry.register({RowParallelLinear: "megatron_RowParallelLinear"})
class _LoRAMegatronRowParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron RowParallelLinear layers.

    This implementation creates row-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter for RowParallelLinear.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a = RowParallelLinear(
            self.input_size,
            attr_config.rank,
            config=self.config,
            input_is_parallel=True,
            skip_bias_add=True,
            bias=False,
            init_method=attr_config.lora_a_init,
        )

        lora_b = nn.Linear(
            in_features=attr_config.rank,
            out_features=self.output_size,
            bias=False,
        )
        with torch.no_grad():
            attr_config.lora_b_init(lora_b.weight)

        self._register_adapter_with_device(
            adapter_name, lora_a, lora_b, attr_config.rank, attr_config.scale, attr_config.enable
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1 for RowParallelLinear, bias not sharded.

        For RowParallelLinear:
        - lora_a weight: sharded at dim 1 (RowParallelLinear)
        (lora_b is a regular nn.Linear and is not sharded)
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        if hasattr(self, "_lora_adapters"):
            for adapter_name in self._lora_adapters:
                lora_a_name = f"lora_a_{adapter_name}"
                lora_a = self._lora_adapters[adapter_name]["lora_a"]

                assert isinstance(lora_a, RowParallelLinear)
                lora_a_sharded = lora_a.sharded_state_dict(
                    prefix=f"{prefix}{lora_a_name}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(lora_a_sharded)

        return sharded_state_dict


# Register quantized versions if available
LoRAModuleRegistry.register({QuantColumnParallelLinear: "quant_megatron_ColumnParallelLinear"})(
    _LoRAMegatronColumnParallelLinear
)
LoRAModuleRegistry.register({QuantRowParallelLinear: "quant_megatron_RowParallelLinear"})(
    _LoRAMegatronRowParallelLinear
)


class _QuantLoRAMegatronColumnParallelLinear(
    _LoRAMegatronColumnParallelLinear, QuantColumnParallelLinear
):
    """Quantized LoRA ColumnParallelLinear that combines LoRA and quantization.

    This class ensures that the base layer functionality is quantized while
    preserving LoRA adapter functionality.
    """

    def _setup(self):
        QuantColumnParallelLinear._setup(self)


class _QuantLoRAMegatronRowParallelLinear(_LoRAMegatronRowParallelLinear, QuantRowParallelLinear):
    """Quantized LoRA RowParallelLinear that combines LoRA and quantization.

    This class ensures that the base layer functionality is quantized while
    preserving LoRA adapter functionality.
    """

    def _setup(self):
        QuantRowParallelLinear._setup(self)


QuantModuleRegistry.register(
    {_LoRAMegatronColumnParallelLinear: "lora_megatron_ColumnParallelLinear"}
)(_QuantLoRAMegatronColumnParallelLinear)
QuantModuleRegistry.register({_LoRAMegatronRowParallelLinear: "lora_megatron_RowParallelLinear"})(
    _QuantLoRAMegatronRowParallelLinear
)
