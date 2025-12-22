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

"""Base class for quantization modules."""

import contextlib
import warnings

import torch

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from ...utils import is_torch_export_mode
from .tensor_quantizer import SequentialQuantizer, TensorQuantizer

__all__ = [
    "QuantInputBase",
    "QuantLinearConvBase",
    "QuantModule",
    "QuantModuleRegistry",
]


class QuantModule(DynamicModule):
    """A base class for quantized modules."""

    def modelopt_post_restore(self, prefix: str = ""):
        """Post-restore to correctly configure the TensorQuantizer states.

        TensorQuantizer states are restored to their shape before saving. Now we need to further configure them.
            1. For non-sharded modules this simply involves moving the TensorQuantizer states to the right device.
                This applies for regular Pytorch models and HuggingFace models.
            2. For sharded modules the restored states of TensorQuantizer could be incorrect. This is because
                parallelism such as TP might have been changed between saving and resoring. So we need to re-calculate
                the state shapes. Hence such modules should override this and implement their own logic.
        """
        # Get a parameter or buffer that does not belong to a TensorQuantizer
        non_tq_param_or_buffer = None
        for name, param_or_buffer in self.state_dict().items():
            parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
            if not isinstance(parent, TensorQuantizer):
                non_tq_param_or_buffer = param_or_buffer
                break

        if non_tq_param_or_buffer is None:
            warnings.warn(
                f"Could not identify the device for TensorQuantizer states of {prefix}. "
                "Please move the model to the right device now. This can be done by calling "
                "`model.to(device)`."
            )
            return

        # Move the TensorQuantizer states to the right device (dtype should have been restored).
        for module in self.modules():
            if isinstance(module, TensorQuantizer):
                module.to(non_tq_param_or_buffer.device)

    def fold_weight(self):
        """Fold the weight for faster eval."""
        # Handle all attributes that end with _weight_quantizer
        for name in dir(self):
            attr = getattr(self, name)
            if (
                name.endswith("weight_quantizer")
                and isinstance(attr, TensorQuantizer)
                and attr.fake_quant
            ):
                # Get the corresponding weight name by removing _weight_quantizer suffix
                weight_name = name[:-10]

                assert hasattr(self, weight_name), (
                    f"{name} doesn't have a corresponding {weight_name} in {self.__class__.__name__}"
                )
                weight = getattr(self, weight_name)
                weight.data.copy_(attr(weight.float()).to(weight.dtype))
                attr.disable()
                _attrs = [
                    "_pre_quant_scale",
                    "_amax",
                ]
                for attr_name in _attrs:
                    if hasattr(attr, attr_name):
                        delattr(attr, attr_name)


QuantModuleRegistry = _DMRegistryCls("Quant", QuantModule)


class QuantInputBase(QuantModule):
    """Base class for modules where the input is quantized."""

    input_quantizer: TensorQuantizer
    output_quantizer: TensorQuantizer
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def forward(self, input, *args, **kwargs):
        """Quantize the input before calling the original forward method."""
        input = self.input_quantizer(input)
        output = super().forward(input, *args, **kwargs)
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    def _setup(self):
        """Patch the module's forward method to quantize the input."""
        self._register_temp_attribute(
            "input_quantizer", TensorQuantizer(self.default_quant_desc_input)
        )
        self._register_temp_attribute(
            "output_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self.output_quantizer.disable()


class QuantLinearConvBase(QuantInputBase):
    """Base class for quantized linear modules.

    Quantized linear modules are modules where both the input and the weight are quantized.
    """

    weight_quantizer: TensorQuantizer | SequentialQuantizer
    _enable_weight_quantization: bool
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @contextlib.contextmanager
    def quantize_weight(self):
        """Context in which `self.weight` is quantized."""
        self._enable_weight_quantization = True
        try:
            yield
        finally:
            self._enable_weight_quantization = False

    @staticmethod
    def _get_quantized_weight(module: "QuantLinearConvBase", weight: torch.Tensor) -> torch.Tensor:
        if module._enable_weight_quantization or is_torch_export_mode():
            return module.weight_quantizer(weight)
        return weight

    def forward(self, input, *args, **kwargs):
        """Quantize the input and the weight before calling the original forward method."""
        # self.quntize_weight() setting attributes is not allowed for torch.export.
        if is_torch_export_mode():
            return super().forward(input, *args, **kwargs)

        with self.quantize_weight():
            return super().forward(input, *args, **kwargs)

    def _setup(self):
        super()._setup()
        self._register_temp_attribute(
            "weight_quantizer", TensorQuantizer(self.default_quant_desc_weight)
        )
        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_dynamic_attribute("weight", self._get_quantized_weight)


class _LegacyQuantInputBaseMixin:
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantInputBase
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_input = quant_desc_input or self.default_quant_desc_input
        super().__init__(*args, **kwargs)
        QuantModuleRegistry.convert(self)


class _LegacyQuantLinearConvBaseMixin(_LegacyQuantInputBaseMixin):
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantLinearConvBase
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, quant_desc_weight=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight
        super().__init__(*args, quant_desc_input=quant_desc_input, **kwargs)
