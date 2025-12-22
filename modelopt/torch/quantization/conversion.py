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

"""Quantization conversion/restore utilities."""

import fnmatch
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch.nn as nn

from modelopt.torch.opt.conversion import ApplyModeError, ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.utils import get_unwrapped_name

from .config import (
    QuantizeConfig,
    QuantizeQuantCfgType,
    QuantizerAttributeConfig,
    _QuantizeExportConfig,
)
from .nn import (
    QuantModule,
    QuantModuleRegistry,
    SequentialQuantizer,
    SVDQuantLinear,
    TensorQuantizer,
)
from .utils import is_quantized, is_quantized_linear

__all__ = [
    "register",
    "replace_quant_module",
    "set_quantizer_attribute",
    "set_quantizer_by_cfg",
    "set_quantizer_by_cfg_context",
    "unregister",
]


def convert_to_quantized_model(model: ModelLikeModule, config: QuantizeConfig) -> ConvertReturnType:
    """Convert the model to a quantized one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    replace_quant_module(model, version=ModeloptStateManager(model).state_version)
    set_quantizer_by_cfg(model, config.get("quant_cfg", {}))

    metadata = {}
    update_quantize_metadata(model, config, metadata)

    return model, metadata


def convert_to_quantized_model_svdquant(
    model: ModelLikeModule, config: QuantizeConfig
) -> ConvertReturnType:
    """Convert the model to a quantized one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    create_and_replace_svdquant_linear_on_the_fly(model)
    set_quantizer_by_cfg(model, config.get("quant_cfg", {}))

    metadata = {}
    update_quantize_metadata(model, config, metadata)

    return model, metadata


def restore_quantized_model(
    model: ModelLikeModule, config: QuantizeConfig, metadata: MetadataDict
) -> nn.Module:
    """Insert quantizers to the model and restore the quantizer states from the given state dict."""
    # initialize the true module if necessary
    convert_to_quantized_model(model, config)

    return restore_quantizer_state(model, config, metadata)


def restore_quantizer_state(model: nn.Module, config: QuantizeConfig, metadata: MetadataDict):
    """Restore the quantizer states from the given state dict.

    For NeMo-MCore sharded checkpoint (torch-dist), quantizer_state is removed from the
    metadata and stored with the main checkpoint as extra_state (similar to TransformerEngine).
    This is because quantizer_state's keys also need to be sharded/remapped during resuming.
    The restore of the quantizer_state is moved to QuantModule.set_extra_state when
    load_state_dict is called.

    Here we detect whether quantizer_state exists in the metadata. The model already has
    QuantModule replaced but without quantizer_state nor any buffer attached. For more
    details regarding how NeMo-MCore sharded checkpoint is restored,
    see modelopt.torch.opt.plugins.mcore_dist_checkpointing.restore_sharded_modelopt_state.
    """
    if "quantizer_state" not in metadata:
        # MCore-NeMo sharded checkpoint (`torch-dist`) has its quantizer_state stored as the
        # extra_state of `QuantModule`. The quantizer_state is resumed with
        # QuantModule.set_extra_state().
        return model

    quantizer_state_dict = metadata["quantizer_state"]
    unmatched_keys = quantizer_state_dict.keys() - quantizer_state(model).keys()
    extra_keys = quantizer_state(model).keys() - quantizer_state_dict.keys()

    if unmatched_keys:
        raise ApplyModeError(f"Unmatched keys in quantizer state_dict: {unmatched_keys}")
    if extra_keys:
        raise ApplyModeError(f"Extra keys in quantizer state_dict: {extra_keys}")

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            name = get_unwrapped_name(name, model)
            module.set_from_modelopt_state(quantizer_state_dict[name])

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            name = get_unwrapped_name(name, model)
            module.modelopt_post_restore(name)

    return model


SVDQuantModuleRegistry = _DMRegistryCls("SVDQuant")


def create_and_replace_svdquant_linear_on_the_fly(model):
    for name, module in model.named_modules():
        if is_quantized_linear(module) and type(module) not in SVDQuantModuleRegistry:
            SVDQuantModuleRegistry.register({type(module): module.__class__.__name__})(
                SVDQuantLinear
            )
    print("Replacing instances of QuantLinear with SVDQuantLinear.")
    _replace_quant_module(
        model, version=ModeloptStateManager(model).state_version, registry=SVDQuantModuleRegistry
    )


def restore_svdquant_model(model: nn.Module, config: QuantizeConfig, metadata: MetadataDict):
    """Restore the svdquant states from the given state dict."""
    create_and_replace_svdquant_linear_on_the_fly(model)
    restore_quantizer_state(model, config, metadata)
    return model


def update_quantize_metadata(
    model: nn.Module, config: QuantizeConfig, metadata: MetadataDict
) -> None:
    """Update the quantizer state in the metadata dict."""
    metadata["quantizer_state"] = quantizer_state(model)


def quantizer_state(model: nn.Module) -> dict[str, Any]:
    """Returns the quantizer state dict describing the quantizer states in the model."""
    return {
        get_unwrapped_name(n, model): m.get_modelopt_state()
        for n, m in model.named_modules()
        if isinstance(m, (TensorQuantizer, SequentialQuantizer))
    }


def replace_quant_module(model: nn.Module, version=None, registry=QuantModuleRegistry):
    """Recursively replace the module with quantized module."""
    from .plugins.custom import (
        register_custom_model_plugins_on_the_fly,
        register_custom_post_conversion_plugins,
    )

    assert not is_quantized(model), "Model must not be quantized!"
    register_custom_model_plugins_on_the_fly(model)

    if type(model) in registry:
        model = registry.convert(model)

    _replace_quant_module(model, version=version, registry=registry)
    register_custom_post_conversion_plugins(model)
    replaced_modules = sum(isinstance(m, TensorQuantizer) for _, m in model.named_modules())
    print(f"Inserted {replaced_modules} quantizers")


def _replace_quant_module(model: nn.Module, version=None, registry=QuantModuleRegistry):
    """Helper function of replace_quant_module."""
    for name, child in model.named_children():
        if type(child) in registry:
            # REPLACE on the parent (model), not on child
            quantized = registry.convert(child)
            setattr(model, name, quantized)

        # now recurse into whichever module is now at `model.name`
        _replace_quant_module(getattr(model, name), version=version, registry=registry)


def set_quantizer_by_cfg(quant_model: nn.Module, quant_cfg: QuantizeQuantCfgType | dict):
    """Update the quantizer attributes based on the specified `quant_cfg`.

    `quant_cfg` is a dictionary mapping wildcards or filter functions
    to its quantizer attributes which are defined in
    :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>`.
    The wildcards or filter functions  are matched against the quantizer module names.
    The specified quantizer attributes of the matched quantizer modules are set accordingly.
    The key ``"default"`` is a special key that sets the quantizer attributes of all the quantizers for which
    no other wildcard or filter functions match the quantizer module name.

    In addition, the dictionary entries could also be pytorch module class names mapping the class specific
    quantization configuration. The pytorch modules should have a quantized equivalent.

    See :meth:`set_quantizer_attribute <modelopt.torch.quantization.conversion.set_quantizer_attribute>`
    for more details.
    """
    quant_cfg = quant_cfg.copy()
    if "default" in quant_cfg:
        set_quantizer_attribute(quant_model, "*", quant_cfg["default"])
        quant_cfg.pop("default")

    for pattern, cfg in quant_cfg.items():
        if str(pattern) in QuantModuleRegistry:
            parent_class = QuantModuleRegistry[str(pattern)]
            assert isinstance(cfg, dict), (
                f"Expected a dictionary for quantizer configuration for child tensor quantizers of {parent_class}."
            )
            for sub_pattern, sub_cfg in cfg.items():
                set_quantizer_attribute(quant_model, sub_pattern, sub_cfg, parent_class)
            continue
        set_quantizer_attribute(quant_model, pattern, cfg)


def set_quantizer_attribute(
    quant_model: nn.Module,
    wildcard_or_filter_func: str | Callable,
    attribute: QuantizerAttributeConfig
    | list[QuantizerAttributeConfig]
    | dict[
        str | Callable,
        QuantizerAttributeConfig | list[QuantizerAttributeConfig],
    ]
    | dict
    | list[dict],
    parent_class: type | None = None,
):
    """Finegrained adjustment of quantizer attribute by wildcard or filter function.

    Args:
        quant_model: A pytorch model
        wildcard_or_filter_func: a wildcard string or a filter function. The wildcard string is matched
            against the quantizer module names. The quantizer modules are
            instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The filter function takes a quantized module name as input and returns ``True`` if the
            quantizer should be adjusted and ``False`` otherwise.
        attribute:  An instance of :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>` or an equivalent
            dictionary or a list of these two types.
            If ``attribute`` is a list, the matched
            :class:`TensorQuantizer <nn.modules.tensor_quantizer.TensorQuantizer>`
            modules will be replaced with :class:`SequentialQuantizer <nn.modules.tensor_quantizer.SequentialQuantizer>`
            modules having one quantizer for each attribute instance from the list.
            See
            :meth:`set_from_attribute_config() <nn.modules.tensor_quantizer.TensorQuantizer.set_from_attribute_config>`
            for more details on the supported attributes and their types.
        parent_class: (Optional) The parent class of the quantizer modules matching ``wildcard_or_filter_func`` which
            should be adjusted. If ``None``, all the matching quantizer modules will be adjusted.
    """
    for name, module in quant_model.named_modules():
        if isinstance(module, (TensorQuantizer, SequentialQuantizer)):
            if isinstance(wildcard_or_filter_func, str):
                if not fnmatch.fnmatch(name, wildcard_or_filter_func):
                    continue
            elif callable(wildcard_or_filter_func):
                if not wildcard_or_filter_func(name):
                    continue
            else:
                raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")

            if parent_class is not None and not isinstance(
                quant_model.get_submodule(".".join(name.split(".")[:-1])), parent_class
            ):
                continue

            if isinstance(attribute, list) and not isinstance(module, SequentialQuantizer):
                parent_module = quant_model.get_submodule(name.rpartition(".")[0])
                module = SequentialQuantizer(*(TensorQuantizer() for _ in range(len(attribute))))
                setattr(parent_module, name.split(".")[-1], module)
            elif isinstance(attribute, list) and len(attribute) != len(module):
                warnings.warn(
                    f"The number of attributes ({len(attribute)}) does not match the number of "
                    f"quantizers of {module} leading to partial assignment.",
                )
            module.set_from_attribute_config(attribute)


@contextmanager
def set_quantizer_by_cfg_context(quant_model: nn.Module, quant_cfg: QuantizeQuantCfgType | dict):
    """Context manager for setting quantizer attributes using `quant_cfg`.

    The set attributes will be reset to the original attributes after exiting the context manager.
    See :meth:`set_quantizer_by_cfg` for more details.

    Use this context manager with caution. Changing certain attributes of the quantizer such as
    `calibrator` can lead to unexpected behavior.
    """
    assert not any(cfg for cfg in quant_cfg.values() if isinstance(cfg, (list, tuple))), (
        "list of config not support."
    )

    original_attributes = {}
    for name, module in quant_model.named_modules():
        if isinstance(module, TensorQuantizer):
            original_attributes[name] = module.get_modelopt_state(properties_only=True)

    set_quantizer_by_cfg(quant_model, quant_cfg)
    yield
    for name, module in quant_model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.set_from_modelopt_state(original_attributes[name], properties_only=True)


def register(original_cls: nn.Module, quantized_cls: nn.Module):
    """Register a quantized class for the given un-quantized original class.

    Args:
        original_cls: The original un-quantized class.
        quantized_cls: The quantized class. This class should have a `_setup` method which initializes
            various quantizers called in the forward. The forward function of the quantized class should call the
            quantizers at the correct location.

    Here is an example of defining a quantized class and registering it:

    .. code-block:: python

        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.nn import TensorQuantizer


        class QuantLayerNorm(nn.LayerNorm):
            def __init__(self, normalized_shape):
                super().__init__(normalized_shape)
                self._setup()

            def _setup(self):
                # Method to setup the quantizers
                self.input_quantizer = TensorQuantizer()
                self.weight_quantizer = TensorQuantizer()

            def forward(self, input):
                input = self.input_quantizer(input)
                weight = self.weight_quantizer(self.weight)
                return F.layer_norm(input, self.normalized_shape, weight, self.bias, self.eps)


        # Register the custom quantized module
        mtq.register(original_cls=nn.LayerNorm, quantized_cls=QuantLayerNorm)

    """
    assert hasattr(quantized_cls, "_setup"), (
        "Quantized class must have a _setup method which initializes various quantizers."
    )

    QuantModuleRegistry.register({original_cls: original_cls.__name__})(quantized_cls)


def unregister(original_cls: nn.Module):
    """Unregister the quantized class for the given un-quantized original class.

    Args:
        original_cls: The original un-quantized class.

    """
    QuantModuleRegistry.unregister(original_cls)


def export_quantized_model(model: nn.Module, config: _QuantizeExportConfig) -> ConvertReturnType:
    """Export the quantized model to a quantized model."""
    raise NotImplementedError("Exporting a quantized model is not supported yet.")


def restore_export_quantized_model(
    model: nn.Module, config: _QuantizeExportConfig, metadata: MetadataDict
) -> nn.Module:
    """Restores the quantized model from the given state dict."""
    raise NotImplementedError("Restoring a quantized & exported model is not supported yet.")
