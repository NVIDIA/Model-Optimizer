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

"""PEFT conversion and restore utilities for LoRA modules."""

import torch.nn as nn

from modelopt.torch.opt.conversion import ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.utils.regex import matches_pattern

from .config import PEFTAttributeConfig, PEFTConfig
from .lora.layer import LoRAModule, LoRAModuleRegistry

# TODO: Add test cases to cover these functions
__all__ = [
    "freeze_lora_weights",
    "replace_lora_module",
]


def convert_to_peft_model(model: ModelLikeModule, config: PEFTConfig) -> ConvertReturnType:
    """Convert the model to a peft one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # Freeze all base model weights before replacing modules if freeze_base_model is True
    if config.freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

    replace_lora_module(model, version=ModeloptStateManager(model).state_version, config=config)

    metadata = {}
    add_adapter(model, config)
    # Update gradient settings for LoRA parameters only
    _update_lora_grads(model, config)

    return model, metadata


def restore_peft_model(
    model: ModelLikeModule, config: PEFTConfig, metadata: MetadataDict
) -> nn.Module:
    model, _ = convert_to_peft_model(model, config)
    return model


def replace_lora_module(
    model: nn.Module, version=None, config: PEFTConfig = None, registry=LoRAModuleRegistry
):
    """Replace modules with LoRA modules."""
    # Register custom plugins (e.g., for Megatron distributed checkpointing)
    from .custom import register_custom_model_plugins_on_the_fly

    register_custom_model_plugins_on_the_fly(model)

    # Iterate through all named modules and replace matching ones
    for name, module in list(model.named_modules()):
        if type(module) in registry:
            if name == "":
                model = registry.convert(model)
            else:
                *parent_path, attr_name = name.split(".")
                if parent_path:
                    parent = model.get_submodule(".".join(parent_path))
                else:
                    parent = model

                lora_module = registry.convert(module)
                setattr(parent, attr_name, lora_module)


def export_peft_model(model: nn.Module, config):
    raise NotImplementedError("Exporting a peft model is not supported yet.")


def restore_export_peft_model(model: nn.Module, config, metadata: MetadataDict):
    raise NotImplementedError("Restoring a peft & exported model is not supported yet.")


def update_peft_metadata(model: nn.Module, config: PEFTConfig, metadata: MetadataDict) -> None:
    """Placeholder for the metadata-related function; not needed in this mode."""


def add_adapter(model, config: PEFTConfig):
    """Add a new LoRA adapter to the model.

    Args:
        model: Model with LoRA modules to add adapters to
        config: PEFTConfig instance containing adapter_cfg and adapter_name

    Returns:
        The model with the new adapter added
    """
    adapter_cfg = config.adapter_cfg
    adapter_name = config.adapter_name

    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            # Collect all matching adapter settings and merge them
            # Later patterns override earlier ones
            merged_setting_dict = None
            for wildcard_or_filter_func, adapter_setting in adapter_cfg.items():
                if matches_pattern(name, wildcard_or_filter_func):
                    setting_dict = (
                        adapter_setting.model_dump(exclude_unset=True)
                        if isinstance(adapter_setting, PEFTAttributeConfig)
                        else dict(adapter_setting)
                    )
                    if merged_setting_dict is None:
                        merged_setting_dict = setting_dict
                    else:
                        merged_setting_dict.update(setting_dict)

            # Only call update_layer_lora if we have settings and enable is not False
            # If enable=False, skip adding the adapter entirely
            if merged_setting_dict is not None and merged_setting_dict.get("enable"):
                module.update_layer_lora(
                    adapter_name,
                    PEFTAttributeConfig(**merged_setting_dict),
                )

    return model


def _iter_lora_modules(model, layer_patterns=None):
    for module_name, module in model.named_modules():
        if isinstance(module, LoRAModule) and matches_pattern(module_name, layer_patterns):
            yield module_name, module


def _set_base_requires_grad(model, *, requires_grad: bool, layer_patterns=None):
    # Collect all LoRA parameter IDs across the entire model
    lora_param_ids = set()
    for _, module in _iter_lora_modules(model, layer_patterns=None):
        for adapter in module._lora_adapters.values():
            for submodule in ("lora_a", "lora_b"):
                for _, param in adapter[submodule].named_parameters():
                    lora_param_ids.add(id(param))

    # Set requires_grad for all parameters in the model (excluding LoRA parameters)
    for name, param in model.named_parameters():
        # Skip LoRA parameters
        if id(param) in lora_param_ids:
            continue
        # If layer_patterns is specified, only affect matching layers
        if layer_patterns is not None:
            module_name = ".".join(name.split(".")[:-1])  # Get module name without param name
            if not matches_pattern(module_name, layer_patterns):
                continue
        param.requires_grad = requires_grad


def _iter_adapter_names(module, adapter_patterns=None):
    for adapter_name in module._lora_adapters:
        if matches_pattern(adapter_name, adapter_patterns, allow_callable=False):
            yield adapter_name


def _set_lora_requires_grad(
    model, *, requires_grad: bool, layer_patterns=None, adapter_patterns=None
):
    for _, module in _iter_lora_modules(model, layer_patterns):
        for adapter_name in _iter_adapter_names(module, adapter_patterns):
            adapter = module._lora_adapters[adapter_name]
            for submodule in (adapter["lora_a"], adapter["lora_b"]):
                for _, param in submodule.named_parameters():
                    param.requires_grad = requires_grad


def freeze_lora_weights(model, *, layer_patterns=None, adapter_patterns=None):
    """Freeze LoRA adapter weights to prevent gradient updates during training.

    This function sets requires_grad=False for LoRA adapter parameters (lora_a and lora_b).
    Useful when you want to train only the base model weights or evaluate the model
    without updating LoRA adapters.

    Args:
        model: Model containing LoRA modules whose adapter weights should be frozen
        layer_patterns: Optional patterns (str, bytes, or Iterable) to match specific
            layer names. If provided, only layers matching these patterns will be affected.
            Supports Unix-style wildcards (e.g., "*.linear", "transformer.*")
        adapter_patterns: Optional patterns (str or Iterable) to match specific adapter
            names. If provided, only adapters matching these patterns will be affected.
            Supports Unix-style wildcards
    """
    _set_lora_requires_grad(
        model,
        requires_grad=False,
        layer_patterns=layer_patterns,
        adapter_patterns=adapter_patterns,
    )


def unfreeze_lora_weights(model, *, layer_patterns=None, adapter_patterns=None):
    """Unfreeze LoRA adapter weights to allow gradient updates during training.

    This function sets requires_grad=True for LoRA adapter parameters (lora_a and lora_b).
    This is the typical setting for LoRA fine-tuning where adapter weights are trained.

    Args:
        model: Model containing LoRA modules whose adapter weights should be unfrozen
        layer_patterns: Optional patterns (str, bytes, or Iterable) to match specific
            layer names. If provided, only layers matching these patterns will be affected.
            Supports Unix-style wildcards (e.g., "*.linear", "transformer.*")
        adapter_patterns: Optional patterns (str or Iterable) to match specific adapter
            names. If provided, only adapters matching these patterns will be affected.
            Supports Unix-style wildcards
    """
    _set_lora_requires_grad(
        model,
        requires_grad=True,
        layer_patterns=layer_patterns,
        adapter_patterns=adapter_patterns,
    )


def _update_lora_grads(model, config: PEFTConfig):
    """Update gradient computation settings for LoRA parameters only (internal function).

    This internal function configures which LoRA adapter parameters should have gradients
    computed based on the freeze_lora_weights setting in the PEFTConfig. It's typically
    called during model initialization after LoRA adapters have been added.

    Note: This function only affects LoRA parameters. Base model parameter gradients
    should be set separately (e.g., in convert_to_peft_model before LoRA module replacement).

    Args:
        model: Model containing LoRA modules to configure
        config: PEFTConfig instance with freeze_lora_weights setting
            - If config.freeze_lora_weights is True, LoRA weights will have requires_grad=False
            - If config.freeze_lora_weights is False, LoRA weights will have requires_grad=True
    """
    _set_lora_requires_grad(model, requires_grad=not config.freeze_lora_weights)
