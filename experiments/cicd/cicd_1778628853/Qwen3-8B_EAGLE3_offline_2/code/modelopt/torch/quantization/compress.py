# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for compressing the model weights after quantization."""

__all__ = ["compress"]

import fnmatch
import warnings

import torch
import torch.nn as nn

from modelopt.torch.opt import apply_mode
from modelopt.torch.opt.conversion import ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from .backends.gemm_registry import disable_real_quant_gemm, enable_real_quant_gemm
from .config import CompressCfgType, CompressConfig
from .conversion import _replace_quant_module, set_quantizer_attributes_partial
from .nn.modules.quant_linear import RealQuantLinear
from .qtensor import QTensorWrapper, pack_real_quantize_weight
from .utils import is_quantized_linear

try:
    from .plugins.megatron import (
        _MegatronColumnParallelLinear,
        _MegatronRowParallelLinear,
        _RealQuantMegatronColumnParallelLinear,
        _RealQuantMegatronRowParallelLinear,
    )

    mcore_available = True
except ImportError:
    mcore_available = False

RealQuantModuleRegistry = _DMRegistryCls("RealQuant")


def compress_convert(
    model,
    config: CompressConfig,
    skip_real_quantize_weight: bool = False,
) -> ConvertReturnType:
    """Compress entry point.

    This function converts the model to a real quantized model.

    Args:
        model: The model to compress.
        config: The compression configuration.
        skip_real_quantize_weight: Whether to skip the real quantize step. Currently, it is
            only set to True in the Megatron restore path to unify the restore behavior regardless
            of whether the model is initialized on meta device or not.

    Returns:
        The compressed model.
    """
    for _, module in model.named_modules():
        if is_quantized_linear(module) and type(module) not in RealQuantModuleRegistry:
            class_to_register = RealQuantLinear
            if mcore_available:
                if issubclass(type(module), _MegatronRowParallelLinear):
                    class_to_register = _RealQuantMegatronRowParallelLinear
                elif issubclass(type(module), _MegatronColumnParallelLinear):
                    class_to_register = _RealQuantMegatronColumnParallelLinear
            RealQuantModuleRegistry.register({type(module): module.__class__.__name__})(
                class_to_register
            )
    # Convert QuantLinear to RealQuantLinear
    _replace_quant_module(
        model, version=ModeloptStateManager(model).state_version, registry=RealQuantModuleRegistry
    )

    compress_cfg = config.compress
    if "default" in compress_cfg and isinstance(compress_cfg["default"], bool):
        set_quantizer_attributes_partial(
            model, "*weight_quantizer*", {"fake_quant": not compress_cfg["default"]}
        )

    for pattern, to_compress in compress_cfg.items():
        if pattern == "default":
            continue
        if isinstance(to_compress, bool):

            def filter_func(name):
                return fnmatch.fnmatch(name, pattern) and "weight_quantizer" in name

            set_quantizer_attributes_partial(model, filter_func, {"fake_quant": not to_compress})
        else:
            raise ValueError(
                f"Invalid compression configuration: {to_compress}, expected a boolean as value."
            )
    # If real quant quantizer is present, real quantize the weights.
    if not skip_real_quantize_weight:
        pack_real_quantize_weight(model)

    def _has_qtensorwrapper(module):
        if hasattr(module, "weight") and isinstance(module.weight, QTensorWrapper):
            return True
        return any(_has_qtensorwrapper(submodule) for _, submodule in module.named_children())

    if _has_qtensorwrapper(model):
        warnings.warn(
            "Real quantization has been applied to the model. This feature is still "
            "experimental, and some functionalities may not be supported. For example, "
            "converting the model back to its original state or saving and restoring "
            "the quantized model may not be available."
        )

    # Turn on real quant gemm after compression
    if config.quant_gemm:
        enable_real_quant_gemm(model)
    else:
        disable_real_quant_gemm(model)

    metadata = {}
    update_compress_metadata(model, config, metadata)

    return model, metadata


def compress_restore(
    model: ModelLikeModule, config: CompressConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore the model from the compressed state.

    Note:
        When restoring Megatron distributed checkpoint, real_quantizer_state and q_tensor_state
        have been removed from metadata and stored as a part of QuantModule.extra_state.
        Restoring happens in set_extra_state when load_state_dict is called. We also skip real
        quantize weight (skip_real_quantize_weight). All these steps are
        delayed. For details, see plugins.megatron.quant_module_set_extra_state.
    """
    # Compress with dummy weights
    model, _ = compress_convert(
        model,
        config,
        skip_real_quantize_weight=("q_tensor_state" not in metadata),
    )
    # restore scale state in weight quantizer
    if "real_quantizer_state" in metadata:
        for name, module in model.named_modules():
            if isinstance(module, RealQuantLinear) and name in metadata["real_quantizer_state"]:
                if not metadata["real_quantizer_state"][name].items():
                    raise ValueError(f"Cannot find real quantizer state for {name}")
                module.weight_quantizer.set_from_modelopt_state(
                    metadata["real_quantizer_state"][name]
                )

    # restore real quant tensor states
    if "q_tensor_state" in metadata:
        for name, module in model.named_modules():
            if isinstance(module, RealQuantLinear) and name in metadata["q_tensor_state"]:
                module._parameters["weight"] = QTensorWrapper(
                    qtensor=torch.empty(
                        metadata["q_tensor_state"][name]["quantized_data.shape"],
                        dtype=metadata["q_tensor_state"][name]["quantized_data.dtype"],
                        device=module.weight.device,
                    ),
                    metadata=metadata["q_tensor_state"][name]["metadata"],
                )
    return model


def update_compress_metadata(model: nn.Module, config: CompressConfig, metadata: MetadataDict):
    # save scales state in weight quantizer
    real_quantizer_state = {}
    for name, module in model.named_modules():
        if isinstance(module, RealQuantLinear):
            real_quantizer_state[name] = module.weight_quantizer.get_modelopt_state()

    # real quant tensor states
    q_tensor_state = {}
    for name, module in model.named_modules():
        if isinstance(module, RealQuantLinear) and isinstance(module.weight, QTensorWrapper):
            q_tensor_state[name] = module.weight.get_state()

    metadata["real_quantizer_state"] = real_quantizer_state
    metadata["q_tensor_state"] = q_tensor_state


def compress(model, config: CompressCfgType = None):
    """Compress model weights of quantized model.

    This function compresses weights in layers that have an enabled `weight_quantizer` with
    a supported quantization format. The compression is controlled by a pattern-based configuration.

    Args:
        model: The quantized model to compress.
        config: Dictionary mapping layer patterns to boolean compression flags.
            If ``None``, defaults to ``{"default": True}`` which compresses all supported layers.

            Example configuration::

                {
                    "*.mlp.fc1*": False,  # Skip compression for fc1 layers
                    "default": True,  # Compress all other layers
                }

            Note: Each configuration except "default" is applied sequentially; therefore the later
            configurations will override the previous ones if the same layer is matched.


    Note: This function modifies the input model in-place.
    """
    if config is None:
        config = CompressConfig()
    apply_mode(model, [("real_quantize", config)])


def is_real_quantized(model):
    """Check if the model is real quantized."""
    return any(isinstance(_module, RealQuantLinear) for _module in model.modules())
