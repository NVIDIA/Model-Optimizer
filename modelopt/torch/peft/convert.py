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

"""User-facing PEFT API for LoRA module conversion and adapter management."""

from typing import Any

import torch.nn as nn

from modelopt.torch.opt import apply_mode
from modelopt.torch.peft.config import PEFTConfig
from modelopt.torch.peft.conversion import add_adapter
from modelopt.torch.utils.regex import matches_pattern

try:
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    MEGATRON_LAYERS = (ColumnParallelLinear, RowParallelLinear)
except Exception:
    MEGATRON_LAYERS = ()

from .lora.layer import LoRAModule
from .mode import PEFTModeRegistry

__all__ = [
    "disable_adapters",
    "enable_adapters",
    "is_peft_model",
    "update_model",
]


def update_model(
    model: nn.Module,
    config: dict[str, Any] | PEFTConfig,
):
    """Update model with PEFT/LoRA adapters.

    This function handles both initial PEFT conversion and adding additional adapters:
    - First call: Converts modules to LoRAModules and adds the first adapter
    - Subsequent calls: Adds new adapters to existing LoRAModules

    Args:
        model: The model to update
        config: PEFT configuration dict or PEFTConfig instance

    Returns:
        The updated model with LoRA adapters
    """
    if not is_megatron_core_model(model):
        raise ValueError("PEFT mode currently supports Megatron-Core models only.")

    # Check if model is already in PEFT mode by looking for LoRA modules
    if not is_peft_model(model):
        apply_mode(model, mode=[("peft", config)], registry=PEFTModeRegistry)
    else:
        if not isinstance(config, PEFTConfig):
            config = PEFTConfig(**config)
        add_adapter(model, config)
    return model


def is_peft_model(model: nn.Module) -> bool:
    """Check if the model has been converted to PEFT/LoRA model.

    This function checks if any modules in the model are LoRAModule instances,
    which indicates the model has already been converted to PEFT mode.

    Args:
        model: The model to check

    Returns:
        True if the model contains LoRA modules, False otherwise
    """
    return any(isinstance(module, LoRAModule) for _, module in model.named_modules())


def _set_adapter_state(model, enable_state, layer_patterns=None, adapter_patterns=None):
    """Helper function to set adapter states.

    Args:
        model: Model with LoRA adapters
        enable_state: Boolean state to set for matching adapters
        layer_patterns: Optional list of layer name patterns (wildcards or callables)
        adapter_patterns: Optional list of adapter name patterns (wildcards)
    """
    if not is_peft_model(model):
        raise ValueError("Model must be a PEFT model to set adapter states.")

    for module_name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            if layer_patterns is not None:
                if not matches_pattern(module_name, layer_patterns, allow_callable=True):
                    continue

            for adapter_name, adapter_dict in module._lora_adapters.items():
                if adapter_patterns is not None:
                    if not matches_pattern(adapter_name, adapter_patterns, allow_callable=False):
                        continue

                adapter_dict["enable"] = enable_state


def disable_adapters(model, layers_to_disable=None, adapters_to_disable=None):
    """Disable LoRA adapters in the model.

    Args:
        model: Model with LoRA adapters
        layers_to_disable: Optional list of layer name patterns (wildcards or callables)
                          to disable adapters on. If None, disables on all layers.
        adapters_to_disable: Optional list of adapter name patterns (wildcards) to disable.
                           If None, disables all adapters.

    Examples:
        # Disable all adapters
        disable_adapters(model)

        # Disable adapters only on attention layers
        disable_adapters(model, layers_to_disable=["*attention*"])

        # Disable only "default" adapters
        disable_adapters(model, adapters_to_disable=["*default*"])

        # Disable "default" adapters on attention layers only
        disable_adapters(model, layers_to_disable=["*attention*"], adapters_to_disable=["*default*"])
    """
    _set_adapter_state(
        model,
        enable_state=False,
        layer_patterns=layers_to_disable,
        adapter_patterns=adapters_to_disable,
    )


def enable_adapters(model, layers_to_enable=None, adapters_to_enable=None):
    """Enable LoRA adapters in the model.

    Args:
        model: Model with LoRA adapters
        layers_to_enable: Optional list of layer name patterns (wildcards or callables)
                         to enable adapters on. If None, enables on all layers.
        adapters_to_enable: Optional list of adapter name patterns (wildcards) to enable.
                          If None, enables all adapters.

    Examples:
        # Enable all adapters
        enable_adapters(model)

        # Enable adapters only on MLP layers
        enable_adapters(model, layers_to_enable=["*mlp*"])

        # Enable only "finetuned" adapters
        enable_adapters(model, adapters_to_enable=["*finetuned*"])

        # Enable "finetuned" adapters on MLP layers only
        enable_adapters(model, layers_to_enable=["*mlp*"], adapters_to_enable=["*finetuned*"])
    """
    _set_adapter_state(
        model,
        enable_state=True,
        layer_patterns=layers_to_enable,
        adapter_patterns=adapters_to_enable,
    )


def is_megatron_core_model(model) -> bool:
    if MEGATRON_LAYERS:
        for m in model.modules():
            if isinstance(m, MEGATRON_LAYERS):
                return True
    return False
