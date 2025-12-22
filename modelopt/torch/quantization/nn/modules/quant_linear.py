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

"""Quantized Linear."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import backends, tensor_quant
from ...qtensor.base_qtensor import QTensorWrapper, dynamically_update_state_methods
from ...utils import is_torch_export_mode
from .quant_module import (
    QuantLinearConvBase,
    QuantModule,
    QuantModuleRegistry,
    _LegacyQuantLinearConvBaseMixin,
)
from .tensor_quantizer import TensorQuantizer

__all__ = ["Linear", "QuantLinear", "RealQuantLinear", "SVDQuantLinear"]


@QuantModuleRegistry.register({nn.Linear: "nn.Linear"})
class _QuantLinear(QuantLinearConvBase):
    """Quantized base class for nn.Linear type classes."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    @staticmethod
    def quantized_linear_fn(package, func_name, self, input, weight, *args, **kwargs):
        """Quantized version of a generic linear functional."""
        output = getattr(package, func_name)(
            self.input_quantizer(input),
            self.weight_quantizer(weight),
            *args,
            **kwargs,
        )
        return self.output_quantizer(output)


class QuantLinear(_LegacyQuantLinearConvBaseMixin, nn.Linear):
    """Quantized version of nn.Linear."""

    default_quant_desc_weight = _QuantLinear.default_quant_desc_weight


Linear = QuantLinear


class SVDQuantTensorQuantizer(TensorQuantizer):
    """TensorQuantizer with svdquant LoRA support."""

    @property
    def svdquant_lora_a(self):
        """Lora a weights for svdquant."""
        if not hasattr(self, "_svdquant_lora_a"):
            return None
        return self._svdquant_lora_a

    @svdquant_lora_a.setter
    def svdquant_lora_a(self, value):
        """Lora a weights for svdquant."""
        assert value is not None, "svdquant_lora_a cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_svdquant_lora_a"):
            self.register_buffer("_svdquant_lora_a", value.clone().detach())
        else:
            if self._svdquant_lora_a.shape != value.shape:
                raise RuntimeError("Changing shape when setting svdquant_lora_a is not allowed.")
            self._svdquant_lora_a.data.copy_(
                value.clone().detach().to(self._svdquant_lora_a.device)
            )

    @property
    def svdquant_lora_b(self):
        """Lora b weights for svdquant."""
        if not hasattr(self, "_svdquant_lora_b"):
            return None
        return self._svdquant_lora_b

    @svdquant_lora_b.setter
    def svdquant_lora_b(self, value):
        """Lora b weights for svdquant."""
        assert value is not None, "svdquant_lora_b cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_svdquant_lora_b"):
            self.register_buffer("_svdquant_lora_b", value.clone().detach())
        else:
            if self._svdquant_lora_b.shape != value.shape:
                raise RuntimeError("Changing shape when setting svdquant_lora_b is not allowed.")
            self._svdquant_lora_b.data.copy_(
                value.clone().detach().to(self._svdquant_lora_b.device)
            )


class SVDQuantLinear(QuantLinearConvBase):
    """Base class for quantized linear modules with SVDQuant."""

    def _setup(self):
        """Overrides and bypass the _setup function."""
        if isinstance(self.weight_quantizer, SVDQuantTensorQuantizer):
            return
        self.weight_quantizer.__class__ = SVDQuantTensorQuantizer

    def _not_sequential_quantizers(self):
        return isinstance(self.weight_quantizer, TensorQuantizer) and isinstance(
            self.input_quantizer, TensorQuantizer
        )

    def _apply_pre_quant_scale(self, input: torch.Tensor):
        """Applies pre_quant_scale if present."""
        if self.input_quantizer.pre_quant_scale is not None and self._not_sequential_quantizers():
            input = input * self.input_quantizer.pre_quant_scale
        return input

    def _compute_lora_residual(self, input: torch.Tensor):
        """Compute the LoRA residual if present, otherwise return None."""
        if (
            self._not_sequential_quantizers()
            and self.weight_quantizer.svdquant_lora_a is not None
            and self.weight_quantizer.svdquant_lora_b is not None
        ):
            lora_a = F.linear(input, weight=self.weight_quantizer.svdquant_lora_a)
            lora_b = F.linear(lora_a, weight=self.weight_quantizer.svdquant_lora_b)
            return lora_b
        return None

    def forward(self, input, *args, **kwargs):
        """SVDQuant layer forward function."""
        has_svdquant_lora = (
            self._not_sequential_quantizers()
            and self.weight_quantizer.svdquant_lora_a is not None
            and self.weight_quantizer.svdquant_lora_b is not None
        )
        if has_svdquant_lora:
            input = self._apply_pre_quant_scale(input)
            _svdquant_lora_outputs = self._compute_lora_residual(input)
            with self.input_quantizer.disable_pre_quant_scale():
                output = super().forward(input, *args, **kwargs) + _svdquant_lora_outputs
        else:
            output = super().forward(input, *args, **kwargs)
        return output

    def fold_weight(self):
        """Fold the weight for faster eval."""
        super().fold_weight()
        if (
            hasattr(self, "weight_quantizer")
            and hasattr(self, "weight")
            and self.weight_quantizer.fake_quant
        ):
            if (
                self._not_sequential_quantizers()
                and self.weight_quantizer.svdquant_lora_a is not None
                and self.weight_quantizer.svdquant_lora_b is not None
            ):
                self.weight.data.copy_(
                    self.weight
                    + self.weight_quantizer.svdquant_lora_b @ self.weight_quantizer.svdquant_lora_a
                )
            _attrs = [
                "_svdquant_lora_a",
                "_svdquant_lora_b",
            ]
            for attr in _attrs:
                if hasattr(self.weight_quantizer, attr):
                    delattr(self.weight_quantizer, attr)


class RealQuantLinear(QuantModule):
    """Quantized version of nn.Linear with real quantization."""

    list_of_scale_tensors = ["_scale", "double_scale", "_scale_zeros"]
    allow_real_quant_gemm = True

    @property
    def _should_run_real_quant_gemm(self):
        return (
            hasattr(self, "_use_real_quant_gemm")
            and self._use_real_quant_gemm
            and not (self.input_quantizer.is_enabled and self.input_quantizer._if_calib)
            and self.allow_real_quant_gemm
        )

    def has_real_quant_gemm_impl(self, input, *args, **kwargs) -> bool:
        """Get the real quant GEMM implementation base on input arguments."""
        if not hasattr(self, "_real_quant_gemm_impl"):
            self._real_quant_gemm_impl = backends.gemm_registry.find_match(
                self, input, *args, **kwargs
            )
            if self._real_quant_gemm_impl is None:
                warnings.warn(f"RealQuantLinear: No real-quant GEMM found: {self}.")

        return self._real_quant_gemm_impl is not None

    def forward(self, input, *args, **kwargs):
        """RealQuant layer forward function."""
        # For torch.export, we use the default fake quant
        if is_torch_export_mode():
            return super().forward(input, *args, **kwargs)

        # Check if real-quant GEMM is available
        if (
            self._should_run_real_quant_gemm
            and input.numel() > 1
            and self.has_real_quant_gemm_impl(input, *args, **kwargs)
        ):
            # Note: We cache the real-quant GEMM function to avoid matching overhead.
            # This assumes that the function will not change after the first call.
            assert self._real_quant_gemm_impl is not None
            with torch.cuda.nvtx.range("RealQuantLinear gemm"):
                output = self._real_quant_gemm_impl(
                    self, input, self.weight, self.bias, *args, **kwargs
                )
            return self.output_quantizer(output) if hasattr(self, "output_quantizer") else output

        # Otherwise, fallback to the default GEMM
        return super().forward(input, *args, **kwargs)

    def _setup(self):
        class RealQuantParameterDict(dict):
            def __init__(self, weight_quantizer: TensorQuantizer, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_quantizer = weight_quantizer

            def __setitem__(self, key, value):
                if (
                    key == "weight"
                    and self.weight_quantizer
                    and self.weight_quantizer.is_enabled
                    and not self.weight_quantizer._fake_quant
                    and value.element_size() > 1
                ):
                    # reset the amax for later calibration
                    if (
                        self.weight_quantizer.amax is not None
                        and self.weight_quantizer.amax.is_meta
                    ):
                        delattr(self.weight_quantizer, "_amax")
                        self.weight_quantizer.amax = self.weight_quantizer._get_amax(value)
                        self.weight_quantizer._calibrator.reset()
                    # compress the weight
                    real_quant_tensor = self.weight_quantizer(value)
                    real_quant_value = QTensorWrapper(real_quant_tensor)
                    del value  # delete the original weight to save memory
                    value = real_quant_value
                super().__setitem__(key, value)

        # Monkey patch the _parameters.__setitem__ to real quant the weight when loading
        # HF accelerate loads the weight by directly assigning the weight through the _parameters dict.
        self._parameters = RealQuantParameterDict(self.weight_quantizer, self._parameters)

        # Function to dynamically override load_state_dict
        dynamically_update_state_methods(self)

    def _apply(self, fn, recurse=True):
        """Override the _apply method to ensure that the weight is real-quantized."""
        # Check if fn is a tensor_cast_fun and print warning if so
        if hasattr(fn, "__name__") and "tensor_cast" in fn.__name__.lower():
            warnings.warn("RealQuantLinear does not support tensor_cast_fun.")
            return self
        elif "to_empty" in str(fn):
            # Handle meta device materialization using to_empty(). to_empty() calls _apply()
            # with a lambda function over torch.empty_like. The function's name is <lambda>;
            # hence we can only detect to_empty keyword in the __repr__. We take care
            # recursive _apply over all suubmodules (e.g. input and weight quantizers are
            # submodules). Parameters and buffer are all taken care.
            #
            # Since the parameter is reassigned, the QTensorWrapper will be gone entirely.
            # Hence we custom the behavior such that the QTensorWrapper is reapplied afterward.
            if recurse:
                for module in self.children():
                    module._apply(fn, recurse=recurse)

            for key, param in self._parameters.items():
                if param is None:
                    continue
                with torch.no_grad():
                    if "weight" in key and isinstance(param, QTensorWrapper):
                        self._parameters[key] = QTensorWrapper(fn(param), metadata=param.metadata)
                    else:
                        self._parameters[key] = torch.nn.Parameter(fn(param), requires_grad=False)

            for key, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[key] = fn(buf)

            return self
        else:
            # Process the function normally
            return super()._apply(fn, recurse=recurse)
