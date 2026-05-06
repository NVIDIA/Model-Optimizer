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
import re
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, cast

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
    normalize_quant_cfg_list,
)
from .nn import (
    NVFP4StaticQuantizer,
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
    "set_quantizer_attributes_full",
    "set_quantizer_attributes_partial",
    "set_quantizer_by_cfg",
    "set_quantizer_by_cfg_context",
    "unregister",
]


def convert_to_quantized_model(model: ModelLikeModule, config: QuantizeConfig) -> ConvertReturnType:
    """Convert the model to a quantized one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    replace_quant_module(model, version=ModeloptStateManager(model).state_version)
    set_quantizer_by_cfg(model, config.get("quant_cfg", []))

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
    set_quantizer_by_cfg(model, config.get("quant_cfg", []))

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

    For MCore sharded checkpoint (torch-dist), quantizer_state is removed from the
    metadata and stored with the main checkpoint as extra_state (similar to TransformerEngine).
    This is because quantizer_state's keys also need to be sharded/remapped during resuming.
    The restore of the quantizer_state is moved to QuantModule.set_extra_state when
    load_state_dict is called.

    Here we detect whether quantizer_state exists in the metadata. The model already has
    QuantModule replaced but without quantizer_state nor any buffer attached. For more
    details regarding how MCore sharded checkpoint is restored,
    see modelopt.torch.opt.plugins.mcore_dist_checkpointing.restore_sharded_modelopt_state.
    """
    if "quantizer_state" not in metadata:
        # MCore sharded checkpoint (`torch-dist`) has its quantizer_state stored as the
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
            state = quantizer_state_dict[name]
            # TODO: Add a registry for TensorQuantizers and avoid this manual conversion.
            if state.get("_is_nvfp4_static_quantizer") and not isinstance(
                module, NVFP4StaticQuantizer
            ):
                NVFP4StaticQuantizer.from_tensor_quantizer(module)
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


def set_quantizer_by_cfg(quant_model: nn.Module, quant_cfg: QuantizeQuantCfgType):
    """Apply a quantization config list to the quantizers in ``quant_model``.

    ``quant_cfg`` is an **ordered list** of :class:`QuantizerCfgEntry <.config.QuantizerCfgEntry>`
    dicts. Each entry has the following fields:

    - ``quantizer_name`` *(required)*: wildcard matched against quantizer module names via
      :func:`fnmatch`.
    - ``cfg`` *(optional)*: a dict of :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>`
      fields, or a list of such dicts for sequential quantization.
    - ``enable`` *(optional)*: ``True`` or ``False`` to toggle matched quantizers on or off.
      When omitted but ``cfg`` is present, defaults to ``True``.  Every entry must specify at
      least one of ``cfg`` or ``enable`` — an entry with only ``quantizer_name`` is invalid.
    - ``parent_class`` *(optional)*: restricts matching to quantizers whose immediate parent
      module is of this PyTorch class name.

    **Ordering and atomicity:** entries are applied in list order; later entries override earlier
    ones for any quantizer they match. Each entry with a ``cfg`` is a **complete replacement** —
    unspecified attributes revert to their defaults rather than inheriting from a prior entry.
    The typical pattern is to deny all first (``{"quantizer_name": "*", "enable": False}``), then
    selectively enable and configure target quantizers in subsequent entries.

    **``enable`` and ``cfg`` are independent:**

    - An entry with ``cfg`` (and optionally ``enable``) fully replaces the matched quantizer's
      attributes. If ``enable`` is omitted, the quantizer is implicitly enabled.
    - ``{"enable": False}`` without ``cfg`` **only** toggles the matched quantizers off, leaving
      all other attributes unchanged.
    - ``{"enable": True}`` without ``cfg`` **only** toggles the matched quantizers on, using
      whatever attributes they currently have (or their defaults if never configured).

    See :ref:`quant-cfg` for the full format reference and common patterns.
    """
    quant_cfg = normalize_quant_cfg_list(quant_cfg)

    for entry in quant_cfg:
        quantizer_name: str = entry["quantizer_name"]
        cfg = entry["cfg"]  # None, dict, or list — always explicit after normalization
        enable: bool = entry["enable"]  # always explicit after normalization
        parent_class_name = entry.get("parent_class")
        if parent_class_name:
            try:
                parent_class = QuantModuleRegistry[parent_class_name]
            except KeyError:
                raise ValueError(
                    f"parent_class {parent_class_name!r} not found in QuantModuleRegistry. "
                    "Make sure the class has a registered quantized equivalent."
                ) from None
        else:
            parent_class = None

        if cfg is None:
            # No cfg: only toggle the enable state, leave all other attributes unchanged.
            set_quantizer_attributes_partial(
                quant_model, quantizer_name, {"enable": enable}, parent_class
            )
        else:
            # Has cfg: apply full replacement with the explicit enable value.
            if isinstance(cfg, QuantizerAttributeConfig):
                attributes = cfg.model_copy(update={"enable": enable})
            elif isinstance(cfg, dict):
                attributes = QuantizerAttributeConfig(**cfg, enable=enable)
            else:
                attributes = [
                    c.model_copy(update={"enable": enable})
                    if isinstance(c, QuantizerAttributeConfig)
                    else QuantizerAttributeConfig(**c, enable=enable)
                    for c in cfg
                ]
            set_quantizer_attributes_full(quant_model, quantizer_name, attributes, parent_class)


_FUSED_EXPERTS_QUANTIZER_LIST_RE = re.compile(
    r"(weight_quantizers?|input_quantizers?)\.\d+(?=$|\.)"
)


def _normalize_fused_experts_quantizer_name(name: str) -> str:
    """Strip the per-expert index from per-expert quantizer ModuleList names.

    Fused-experts modules register per-expert weight/input quantizers in a
    ``nn.ModuleList``; its children surface as dotted names like
    ``...gate_up_proj_weight_quantizers.0`` (plural) or — if a variant uses
    singular naming — ``...gate_up_proj_weight_quantizer.0``. Neither matches
    the singular-suffix wildcards (``*weight_quantizer``) used in the stock
    configs, so the experts stay at their defaults.

    Return a normalized name where either ``weight_quantizer[s]?.N`` or
    ``input_quantizer[s]?.N`` collapses to the singular form without the index
    so the standard wildcards match.
    """

    def _repl(m: re.Match) -> str:
        base = m.group(1)
        return base.removesuffix("s")

    return _FUSED_EXPERTS_QUANTIZER_LIST_RE.sub(_repl, name)


def _match_quantizer(
    wildcard_or_filter_func: str | Callable,
    name: str,
    module: nn.Module,
    parent_class: type[nn.Module] | None,
    full_model: nn.Module,
):
    if not isinstance(module, (TensorQuantizer, SequentialQuantizer)):
        return False
    if isinstance(wildcard_or_filter_func, str):
        normalized = _normalize_fused_experts_quantizer_name(name)
        if not (
            fnmatch.fnmatch(name, wildcard_or_filter_func)
            or (normalized != name and fnmatch.fnmatch(normalized, wildcard_or_filter_func))
        ):
            return False
    elif callable(wildcard_or_filter_func):
        if not wildcard_or_filter_func(name):
            return False
    else:
        raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")

    # Get the parent module of this quantizer. When name has no dots (root-level quantizer),
    # ".".join([]) == "" and get_submodule("") returns the model itself (PyTorch convention).
    return parent_class is None or isinstance(
        full_model.get_submodule(".".join(name.split(".")[:-1])), parent_class
    )


def set_quantizer_attributes_full(
    quant_model: nn.Module,
    wildcard_or_filter_func: str | Callable,
    attributes: QuantizerAttributeConfig | list[QuantizerAttributeConfig],
    parent_class: type[nn.Module] | None = None,
):
    """Set quantizer attributes by wildcard or filter function, fully overwriting existing attributes.

    Unlike :func:`set_quantizer_attributes_partial`, this function requires a complete
    :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>` and **replaces** the
    matched quantizer's attributes entirely rather than merging with existing ones.

    Args:
        quant_model: A pytorch model.
        wildcard_or_filter_func: A wildcard string or a filter function. The wildcard string is
            matched against the quantizer module names. The quantizer modules are instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The filter function takes a quantizer module name as input and returns ``True`` if the
            quantizer should be adjusted and ``False`` otherwise.
        attributes: A :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>` (or a
            list of them) that **fully replaces** the matched quantizer's current attributes. All
            fields of the config are applied — unspecified fields revert to their defaults.
            If ``attributes`` is a list, the matched
            :class:`TensorQuantizer <nn.modules.tensor_quantizer.TensorQuantizer>`
            modules will be replaced with
            :class:`SequentialQuantizer <nn.modules.tensor_quantizer.SequentialQuantizer>`
            modules having one quantizer per attribute instance in the list.
            See
            :meth:`set_from_attribute_config() <nn.modules.tensor_quantizer.TensorQuantizer.set_from_attribute_config>`
            for details on supported attributes and their types.
        parent_class: (Optional) Restrict matching to quantizers whose immediate parent module is
            an instance of this class. If ``None``, all quantizers matching
            ``wildcard_or_filter_func`` are adjusted.
    """
    if not isinstance(attributes, (QuantizerAttributeConfig, list)):
        raise ValueError(
            f"Invalid type for attributes: {type(attributes)}, "
            "expected QuantizerAttributeConfig or list of QuantizerAttributeConfig."
        )
    if isinstance(attributes, list) and not all(
        isinstance(attr, QuantizerAttributeConfig) for attr in attributes
    ):
        raise ValueError(
            "All elements in attributes list must be of type QuantizerAttributeConfig."
        )
    for name, module in quant_model.named_modules():
        if _match_quantizer(wildcard_or_filter_func, name, module, parent_class, quant_model):
            if isinstance(attributes, list):
                if not isinstance(module, SequentialQuantizer):
                    parent_module = quant_model.get_submodule(name.rpartition(".")[0])
                    module = SequentialQuantizer(
                        *(TensorQuantizer() for _ in range(len(attributes)))
                    )
                    setattr(parent_module, name.split(".")[-1], module)
                elif len(attributes) != len(module):
                    warnings.warn(
                        f"The number of attributes ({len(attributes)}) does not match the number of "
                        f"quantizers of {module} leading to partial assignment.",
                    )
                module.set_from_attribute_config(attributes)
            else:
                if isinstance(module, SequentialQuantizer):
                    # Downgrade SequentialQuantizer back to TensorQuantizer when the
                    # new entry provides a single (non-list) config.
                    parent_module = quant_model.get_submodule(name.rpartition(".")[0])
                    module = TensorQuantizer()
                    setattr(parent_module, name.split(".")[-1], module)
                cast("TensorQuantizer", module).set_from_attribute_config(attributes)


def set_quantizer_attributes_partial(
    quant_model: nn.Module,
    wildcard_or_filter_func: str | Callable,
    partial_attributes: dict[str, Any] | list[dict[str, Any]],
    parent_class: type[nn.Module] | None = None,
):
    """Update a subset of quantizer attributes by wildcard or filter function, merging with existing attributes.

    Unlike :func:`set_quantizer_attributes_full`, this function accepts an arbitrary subset of
    quantizer attributes as a plain ``dict`` and **merges** them into the matched quantizer's
    current attributes, leaving unspecified attributes unchanged.

    Args:
        quant_model: A pytorch model.
        wildcard_or_filter_func: A wildcard string or a filter function. The wildcard string is
            matched against the quantizer module names. The quantizer modules are instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The filter function takes a quantizer module name as input and returns ``True`` if the
            quantizer should be adjusted and ``False`` otherwise.
        partial_attributes: A ``dict`` (or a list of ``dict``) containing only the attributes to
            update. Keys must be valid fields of
            :class:`QuantizerAttributeConfig <.config.QuantizerAttributeConfig>`. Only the
            specified keys are written; all other attributes on the quantizer remain unchanged.
            When a ``dict`` is passed and the matched module is a
            :class:`SequentialQuantizer <nn.modules.tensor_quantizer.SequentialQuantizer>`,
            the dict is broadcast to every sub-quantizer.
            When a ``list`` is passed, the matched module must already be a
            :class:`SequentialQuantizer <nn.modules.tensor_quantizer.SequentialQuantizer>` —
            unlike :func:`set_quantizer_attributes_full`, this function will **not** replace a
            :class:`TensorQuantizer <nn.modules.tensor_quantizer.TensorQuantizer>` with a
            ``SequentialQuantizer``.
            See
            :meth:`set_from_attribute_config() <nn.modules.tensor_quantizer.TensorQuantizer.set_from_attribute_config>`
            for details on supported attributes and their types.
        parent_class: (Optional) Restrict matching to quantizers whose immediate parent module is
            an instance of this class. If ``None``, all quantizers matching
            ``wildcard_or_filter_func`` are adjusted.
    """
    if not isinstance(partial_attributes, (dict, list)):
        raise ValueError(
            f"Invalid type for attributes: {type(partial_attributes)}, expected dictionary or list of dict."
        )
    if isinstance(partial_attributes, list) and not all(
        isinstance(attr, dict) for attr in partial_attributes
    ):
        raise ValueError("All elements in attributes list must be of type dict.")

    for name, module in quant_model.named_modules():
        if _match_quantizer(wildcard_or_filter_func, name, module, parent_class, quant_model):
            module = cast("TensorQuantizer | SequentialQuantizer", module)  # for type checker
            if isinstance(partial_attributes, list):
                if not isinstance(module, SequentialQuantizer):
                    raise ValueError(
                        f"Attributes is a list but {module} is not a SequentialQuantizer."
                    )
                module.set_from_attribute_config(partial_attributes)
            elif isinstance(module, SequentialQuantizer):
                # Broadcast the dict to all sub-quantizers.
                module.set_from_attribute_config([partial_attributes] * len(module))
            else:
                module.set_from_attribute_config(partial_attributes)


@contextmanager
def set_quantizer_by_cfg_context(quant_model: nn.Module, quant_cfg: QuantizeQuantCfgType):
    """Context manager that temporarily applies a quantization config and restores the original state on exit.

    Calls :func:`set_quantizer_by_cfg` on entry and reverts every
    :class:`TensorQuantizer <nn.modules.tensor_quantizer.TensorQuantizer>` in
    ``quant_model`` to its original attributes on exit.

    .. caution::
        Changing stateful attributes such as ``calibrator`` inside this context may produce
        unexpected behavior because those objects are not deep-copied during save/restore.

    Args:
        quant_model: A quantized PyTorch model whose quantizers will be temporarily reconfigured.
        quant_cfg: A quantization config (or list of
            :class:`QuantizerCfgEntry <.config.QuantizerCfgEntry>` dicts) passed directly to
            :func:`set_quantizer_by_cfg`.  Sequential ``cfg`` lists are not allowed.

    Yields:
        None — the context body runs with the new quantizer attributes active.
    """
    quant_cfg = normalize_quant_cfg_list(quant_cfg)

    for entry in quant_cfg:
        if isinstance(entry.get("cfg"), list):
            raise ValueError(
                "Sequential cfg lists are not allowed in set_quantizer_by_cfg_context. "
                "Use only single-dict cfg entries."
            )

    original_attributes: dict[str, dict] = {}
    original_types: dict[str, type] = {}
    for name, module in quant_model.named_modules():
        if isinstance(module, SequentialQuantizer):
            # SequentialQuantizer.get_modelopt_state does not support properties_only;
            # save per-sub-quantizer state so we can fully reconstruct on restore.
            original_attributes[name] = {
                "is_sequential_quantizer": True,
                "sub_states": [tq.get_modelopt_state(properties_only=True) for tq in module],
            }
            original_types[name] = SequentialQuantizer
        elif isinstance(module, TensorQuantizer):
            original_attributes[name] = module.get_modelopt_state(properties_only=True)
            original_types[name] = TensorQuantizer

    set_quantizer_by_cfg(quant_model, quant_cfg)
    yield

    # Restore original quantizer types and attributes. If set_quantizer_by_cfg downgraded a
    # SequentialQuantizer to a TensorQuantizer (or vice-versa), we need to re-create the
    # original module type before restoring attributes.
    for name, module in list(quant_model.named_modules()):
        if name not in original_attributes:
            continue
        orig_type = original_types[name]
        if orig_type is SequentialQuantizer and not isinstance(module, SequentialQuantizer):
            # Restore the SequentialQuantizer that was downgraded
            saved = original_attributes[name]
            parent_name, _, attr_name = name.rpartition(".")
            parent_module = quant_model.get_submodule(parent_name) if parent_name else quant_model
            module = SequentialQuantizer(*(TensorQuantizer() for _ in saved["sub_states"]))
            setattr(parent_module, attr_name, module)
            for tq, sub_state in zip(module, saved["sub_states"]):
                tq.set_from_modelopt_state(sub_state, properties_only=True)
        elif orig_type is TensorQuantizer and not isinstance(module, TensorQuantizer):
            parent_name, _, attr_name = name.rpartition(".")
            parent_module = quant_model.get_submodule(parent_name) if parent_name else quant_model
            module = TensorQuantizer()
            setattr(parent_module, attr_name, module)
            module.set_from_modelopt_state(original_attributes[name], properties_only=True)
        elif orig_type is TensorQuantizer:
            module.set_from_modelopt_state(original_attributes[name], properties_only=True)
        elif orig_type is SequentialQuantizer:
            saved = original_attributes[name]
            for tq, sub_state in zip(module, saved["sub_states"]):
                tq.set_from_modelopt_state(sub_state, properties_only=True)


def set_quantizer_attribute(
    quant_model: nn.Module,
    wildcard_or_filter_func: str | Callable,
    attribute: Any,
    parent_class: type[nn.Module] | None = None,
):
    """Deprecated: use :func:`set_quantizer_attributes_partial` instead."""
    warnings.warn(
        "set_quantizer_attribute is deprecated, use set_quantizer_attributes_partial "
        "or set_quantizer_attributes_full instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return set_quantizer_attributes_partial(
        quant_model, wildcard_or_filter_func, attribute, parent_class
    )


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
