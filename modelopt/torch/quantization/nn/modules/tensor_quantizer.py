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

"""TensorQuantizer Module."""

import contextlib
import math
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None

if hasattr(torch.onnx, "_globals"):
    from torch.onnx._globals import GLOBALS
else:  # torch >= 2.9
    from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

import torch.nn.functional as F
from torch import nn

from modelopt.torch.utils import same_device_as, standardize_constructor_args
from modelopt.torch.utils.distributed import DistributedProcessGroup

from ... import calib
from ... import utils as quant_utils
from ...config import QuantizerAttributeConfig
from ...qtensor import (
    BaseQuantizedTensor,
    FP8QTensor,
    INT4QTensor,
    INT8QTensor,
    MXFP4QTensor,
    MXFP8QTensor,
    NF4QTensor,
    NVFP4QTensor,
    QTensorWrapper,
)
from ...tensor_quant import (
    dynamic_block_quant,
    fake_tensor_quant,
    scaled_e4m3,
    static_blockwise_fp4_fake_quant,
)
from ...utils import is_torch_export_mode
from ..functional import normalized_hadamard_transform

__all__ = [
    "NVFP4StaticQuantizer",
    "SequentialQuantizer",
    "TensorQuantizer",
    "TensorQuantizerCache",
    "is_registered_quant_backend",
    "register_quant_backend",
    "unregister_quant_backend",
]


QuantBackendEntrypoint = Callable[[torch.Tensor, "TensorQuantizer"], torch.Tensor]

_QUANT_FUNCTIONAL_BACKENDS: dict[str, QuantBackendEntrypoint] = {}


def register_quant_backend(name: str, entrypoint: QuantBackendEntrypoint) -> None:
    """Register a custom quantization backend.

    Args:
        name: The name of the backend.
        entrypoint: The entrypoint of the backend. The entrypoint should be a callable that takes in
            the inputs and the tensor quantizer as arguments and returns the quantized tensor.
            See :class:`modelopt.torch.quantization.config.QuantizerAttributeConfig`
            for details on choosing from the registered backends via the ``backend`` and
            ``backend_extra_args`` fields.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Backend name must be a non-empty string.")
    if not callable(entrypoint):
        raise TypeError("Entrypoint must be callable.")
    if name in _QUANT_FUNCTIONAL_BACKENDS:
        warnings.warn(f"Overwriting existing backend: {name}")
    _QUANT_FUNCTIONAL_BACKENDS[name] = entrypoint


def unregister_quant_backend(name: str) -> None:
    """Unregister a custom quantization backend.

    Args:
        name: The name of the backend to unregister.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Backend name must be a non-empty string.")
    _QUANT_FUNCTIONAL_BACKENDS.pop(name, None)


def is_registered_quant_backend(name: str) -> bool:
    """Check if a custom quantization backend is registered.

    Args:
        name: The name of the backend to check.
    """
    return name in _QUANT_FUNCTIONAL_BACKENDS


class TensorQuantizerCache(Protocol):
    """A protocol for a cache interface for TensorQuantizer."""


class TensorQuantizer(nn.Module):
    """Tensor quantizer module.

    This module manages quantization and calibration of input tensor. It can perform fake (simulated quantization)
    or real quantization for various precisions and formats such as FP8 per-tensor, INT8 per-channel,
    INT4 per-block etc.

    If quantization is enabled, it calls the appropriate quantization functional and
    returns the quantized tensor. The quantized tensor data type will be same as the input tensor data type for
    fake quantization. During calibration mode, the module collects the statistics using its calibrator.

    The quantization parameters are as described in
    :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`. They can be set
    at initialization using ``quant_attribute_cfg`` or later by calling :meth:`set_from_attribute_config`.

    Args:
        quant_attribute_cfg: An instance of
            :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>` or None.
            If None, default values are used.
        if_quant: A boolean. If True, quantization is enabled in the forward path.
        if_calib: A boolean. If True, calibration is enabled in the forward path.
        amax: None or an array like object such as list, tuple, numpy array, scalar
            which can be used to construct amax tensor.
    """

    _skip_properties_for_save_restore = {
        "_calibrator",
        "_bias_calibrator",
        "_original_shape",
        "_block_reshape_size",
        "_padding",
        # Extra flags added by huggingface
        "_is_hf_initialized",
        # Extra flags added by deepspeed
        "ds_external_parameters",
        "all_parameters",
        "_external_params",
        "_original_parameters",
        "post_bwd_fn",
        "ds_grads_remaining",
        "ds_id",
        "pre_bwd_fn",
        # quantizer cache for custom backends, like luts
        "_quantizer_cache",
    }

    def __init__(
        self,
        quant_attribute_cfg=None,
        if_quant=True,
        if_calib=False,
        amax=None,
    ):
        """Initialize quantizer and set up required variables."""
        super().__init__()
        quant_attribute_cfg = (
            quant_attribute_cfg if quant_attribute_cfg is not None else QuantizerAttributeConfig()
        )
        if amax is not None:
            self.amax = amax

        self.set_from_attribute_config(quant_attribute_cfg)

        self._if_quant = if_quant
        self._if_calib = if_calib
        self._enable_pre_quant_scale = True
        self._dequantize = False
        self._input_dtype = None

        # Lazy initialize the bias calibrator for KV cache quantization
        self._bias_calibrator = None

        # Optional quantizer cache for caching quantizer related encoding or tensors.
        self._quantizer_cache = None

    def set_from_attribute_config(self, attribute_cfg: QuantizerAttributeConfig | dict):
        """Set quantizer attributes from attribute_dict.

        The attributes are defined in
        :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`.
        """

        def _calibrator_setter(val):
            if val in ["max", "histogram"]:
                calib_cls = calib.MaxCalibrator if val == "max" else calib.HistogramCalibrator
                args, kwargs = (self._num_bits, self._axis, self._unsigned), {}
            else:
                calib_cls, args, kwargs = standardize_constructor_args(val)
            return calib_cls(*args, **kwargs)

        # Some attributes need custom handling.
        # By default, attributes from config are mapped to a name ``f"_{attribute}"``
        _custom_setters: dict[str, tuple[str, Callable]] = {
            "enable": ("_disabled", lambda val: val is False),
            "type": ("_dynamic", lambda val: val == "dynamic"),
            "calibrator": ("_calibrator", _calibrator_setter),
            "backend": ("backend", lambda val: val),
            "backend_extra_args": ("backend_extra_args", lambda val: val or {}),
        }

        for attribute, val in attribute_cfg.items():
            assert attribute in QuantizerAttributeConfig.model_fields, (
                f"{attribute} is not a valid `TensorQuantizer` attribute"
            )
            _tq_attribute_name, _setter = _custom_setters.get(
                attribute, (f"_{attribute}", lambda v: v)
            )
            setattr(self, _tq_attribute_name, _setter(val))

        if self.is_mx_format:
            self._pass_through_bwd = True

    def dequantize(self, inputs: BaseQuantizedTensor | QTensorWrapper):
        """De-quantize a real quantized tensor to a given dtype."""
        qtensor = inputs.get_qtensor() if isinstance(inputs, QTensorWrapper) else inputs
        assert isinstance(qtensor, BaseQuantizedTensor), "Expected input as real quantized tensors."
        kwarg = {
            "scale": self._scale,
            "block_sizes": self.block_sizes,
            "double_scale": getattr(self, "_double_scale", None),
            "scale_zeros": getattr(self, "_scale_zeros", None),
        }
        return qtensor.dequantize(**kwarg)

    @property
    def num_bits(self):
        """Return num_bits for quantization."""
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        self._num_bits = value
        self._calibrator._num_bits = value

    @property
    def maxbound(self):
        """Return maxbound for quantization."""
        if self._num_bits == (4, 3):
            return 448.0
        if self._num_bits == (2, 1) and self._block_sizes.get("scale_bits") == (4, 3):
            return 6.0
        return (1 << (self._num_bits - 1 + int(self._unsigned))) - 1

    @property
    def unsigned(self):
        """Return True if unsigned quantization is used."""
        return self._unsigned

    @unsigned.setter
    def unsigned(self, value):
        self._unsigned = value
        self._calibrator._unsigned = value

    @property
    def pre_quant_scale(self):
        """Return pre_quant_scale used for smoothquant."""
        if not hasattr(self, "_pre_quant_scale") or not self._enable_pre_quant_scale:
            return None
        return self._pre_quant_scale

    @pre_quant_scale.setter
    def pre_quant_scale(self, value):
        assert value is not None, "pre_quant_scale cannot be set to None."
        assert self._enable_pre_quant_scale, (
            "pre_quant_scale cannot be set when forward_with_pre_quant_scale is False."
        )
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not hasattr(self, "_pre_quant_scale"):
            self.register_buffer("_pre_quant_scale", value.clone().detach())
        else:
            if self._pre_quant_scale.shape != value.shape:
                raise RuntimeError("Changing shape when setting pre_quant_scale is not allowed.")
            self._pre_quant_scale.data.copy_(
                value.clone().detach().to(self._pre_quant_scale.device)
            )

    @property
    def amax(self):
        """Return amax for quantization."""
        if not hasattr(self, "_amax") or self.is_mx_format:
            return None
        assert not self._dynamic, "Dynamic quantization does not have fixed amax"
        return self._amax

    @amax.setter
    def amax(self, value):
        assert value is not None, "amax cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", value.clone().detach())
        else:
            if self._amax.shape != value.shape:
                raise RuntimeError("Changing shape when setting amax is not allowed.")
            self._amax.data.copy_(value.clone().detach().to(self._amax.device))

    def reset_amax(self):
        """Reset amax to None."""
        if hasattr(self, "_amax"):
            delattr(self, "_amax")
        self._calibrator.reset()
        self.reset_bias()

    def reset_bias(self):
        """Reset bias to None."""
        if hasattr(self, "_bias_value"):
            delattr(self, "_bias_value")
        if self._bias_calibrator is not None:
            self._bias_calibrator.reset()

    @property
    def step_size(self):
        """Return step size for integer quantization."""
        if not hasattr(self, "_amax"):
            warnings.warn("step_size is undefined under dynamic amax mode!")
            return None
        assert isinstance(self._num_bits, int), (
            "Step size is not defined for non-integer quantization."
        )
        return self._amax / (2.0 ** (self._num_bits - 1 + int(self._unsigned)) - 1.0)

    @property
    def axis(self):
        """Return axis for quantization."""
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = value
        self._calibrator._axis = value

    @property
    def block_sizes(self):
        """Return block_sizes for quantization."""
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, value):
        self._axis = None
        self._block_sizes = value

    @property
    def bias(self):
        """Return bias for quantization."""
        if not hasattr(self, "_bias"):
            return None
        return self._bias

    @property
    def bias_axis(self):
        """Return bias_axis for quantization."""
        if not hasattr(self, "_bias_axis"):
            return None
        return self._bias_axis

    @bias_axis.setter
    def bias_axis(self, value):
        assert value is not None, "bias_axis cannot be set to None."
        assert isinstance(value, (tuple, list)), "bias_axis must be a tuple or a list."
        self._bias_axis = value

    @property
    def bias_method(self):
        """Return bias_method for quantization."""
        if self._bias is None:
            return None
        return self._bias.get("method", "mean")

    @property
    def bias_type(self):
        """Return bias_type for quantization."""
        if self._bias is None:
            return None
        return self._bias.get("type", "static")

    @bias_type.setter
    def bias_type(self, value):
        assert value in [
            "static",
            "dynamic",
        ], "bias_type must be either 'static' or 'dynamic'."
        self._bias["type"] = value

    @property
    def bias_value(self):
        """Return bias for quantization."""
        if not hasattr(self, "_bias_value"):
            return None
        return self._bias_value

    @bias_value.setter
    def bias_value(self, value):
        assert value is not None, "bias cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_bias_value"):
            self.register_buffer("_bias_value", value.clone().detach())
        else:
            if self._bias_value.shape != value.shape:
                raise RuntimeError("Changing shape when setting bias is not allowed.")
            self._bias_value.data.copy_(value.clone().detach().to(self._bias_value.device))

    @property
    def bias_calibrator(self):
        """Return bias_calibrator for quantization."""
        # Get reduce_axis from bias config
        # Bias calibration supports per-channel and per-token quantization
        if self._bias_calibrator is None and self.bias is not None:
            self.bias_axis = tuple(k for k in self.bias if isinstance(k, int))
            if self._bias is not None:
                self._bias_calibrator = calib.BiasCalibrator(
                    method=self.bias_method,
                    axis=self.bias_axis,
                )

        return self._bias_calibrator

    @property
    def fake_quant(self):
        """Return True if fake quantization is used."""
        return self._fake_quant

    @property
    def narrow_range(self):
        """Return True if symmetric integer range for signed quantization is used."""
        return self._narrow_range

    @narrow_range.setter
    def narrow_range(self, value):
        self._narrow_range = value

    @property
    def is_enabled(self):
        """Return true if the modules is not disabled."""
        return not self._disabled

    def disable(self):
        """Bypass the module.

        Neither of calibration, clipping and quantization will be performed if the module is disabled.
        """
        self._disabled = True

    def enable(self):
        """Enable the module."""
        self._disabled = False

    @property
    def trt_high_precision_dtype(self):
        """Return True if FP16 AMAX is used when exporting the model."""
        return self._trt_high_precision_dtype

    @trt_high_precision_dtype.setter
    def trt_high_precision_dtype(self, value):
        self._trt_high_precision_dtype = value

    @property
    def is_mx_format(self):
        """Check if is MX formats."""
        return (
            self.block_sizes is not None
            and self.block_sizes.get("type", None) == "dynamic"
            and self.block_sizes.get("scale_bits", None) == (8, 0)
        )

    def is_mxfp(self, bits):
        """Check if is MXFP4/MXFP6/MXFP8."""
        if bits == 4:
            return (
                self.is_mx_format
                and self.num_bits == (2, 1)
                and self.block_sizes.get(-1, None) == 32
            )
        elif bits == 6:
            return (
                self.is_mx_format
                and self.num_bits == (3, 2)
                and self.block_sizes.get(-1, None) == 32
            )
        elif bits == 8:
            return (
                self.is_mx_format
                and self.num_bits == (4, 3)
                and self.block_sizes.get(-1, None) == 32
            )
        else:
            raise NotImplementedError()

    @property
    def is_static_block_quant(self):
        """Check if is static block quantization."""
        return (
            self.block_sizes is not None
            and self.block_sizes.get("type", None) != "dynamic"
            and self._fake_quant
        )

    def disable_calib(self):
        """Disable calibration."""
        self._if_calib = False

    def enable_calib(self):
        """Enable calibration."""
        if self._calibrator is None:
            raise ValueError("Calibrator was not created, cannot enable calibration.")

        # Dynamic quantization does not need calibration.
        if self._dynamic:
            return
        self._if_calib = True

    def disable_quant(self):
        """Disable quantization."""
        self._if_quant = False

    def enable_quant(self):
        """Enable quantization."""
        self._if_quant = True

    def load_calib_amax(self, *args, **kwargs):
        """Load amax from calibrator.

        Updates the amax buffer with value computed by the calibrator, creating it if necessary.
        ``*args`` and ``**kwargs`` are directly passed to ``compute_amax``, except ``"strict"`` in
        ``kwargs``. Refer to ``compute_amax`` for more details.
        """
        assert not self._dynamic, "Dynamic quantization does not need calibration."

        strict = kwargs.pop("strict", True)
        if getattr(self, "_calibrator", None) is None:
            raise RuntimeError("Calibrator not created.")
        calib_amax = self._calibrator.compute_amax(*args, **kwargs)
        if calib_amax is None:
            err_msg = (
                "Calibrator returned None. This usually happens when calibrator hasn't seen any"
                " tensor."
            )
            if not strict:
                warnings.warn(err_msg)
                warnings.warn("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(
                    err_msg
                    + " Passing 'strict=False' to `load_calib_amax()` will ignore the error."
                )
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", calib_amax.clone().detach())
        else:
            self._amax.data.copy_(calib_amax.clone().detach())

    def load_calib_bias(self, *args, **kwargs):
        """Load affine bias for quantization."""
        assert not self._dynamic, "Dynamic quantization does not need calibration."
        calib_bias = self.bias_calibrator.compute_bias(*args, **kwargs)
        if calib_bias is None:
            raise RuntimeError(
                "Calibrator returned None. This usually happens when calibrator hasn't seen any tensor."
            )

        if not hasattr(self, "_bias_value"):
            self.register_buffer("_bias_value", calib_bias.clone().detach())
        else:
            self._bias_value.data.copy_(calib_bias.clone().detach())

    def _get_amax(self, inputs):
        """Get amax from buffer or compute it dynamically."""
        if hasattr(self, "_amax"):
            amax = self._amax
        else:
            reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(inputs, self._axis)
            amax = quant_utils.reduce_amax(inputs, axis=reduce_axis, keepdims=True).detach()

        amax = amax.detach() if is_torch_export_mode() else amax.data
        return amax

    def validate_attr(
        self, attr_value=None, attr_name="amax", raise_error=False, warn_error=False, name=""
    ):
        """Validate attribute."""
        attr_value = attr_value if attr_value is not None else getattr(self, attr_name, None)
        if attr_value is None or (isinstance(attr_value, torch.Tensor) and attr_value.is_meta):
            return True
        is_valid = (
            torch.all(attr_value >= 0)
            and not torch.any(torch.isinf(attr_value))
            and not torch.any(torch.isnan(attr_value))
        )
        if is_valid:
            return True
        name = f"{name}." if name else ""
        msg = f"{name}{attr_name} contains invalid values: {attr_value}"
        if warn_error:
            warnings.warn(msg)
        if raise_error:
            raise ValueError(msg)
        return False

    def _get_bias(self, inputs):
        """Get bias from buffer or compute it dynamically."""
        if self.bias_calibrator is None:
            return None

        if self.bias_type == "static":
            bias = self._bias_value
        elif self.bias_type == "dynamic":
            bias = self.bias_calibrator.compute_dynamic_bias(inputs)
        else:
            raise ValueError(f"Unsupported bias type: {self.bias_type}")
        return bias

    def _is_real_quantize_support(self):
        """Check if real quantization is supported for this quant config."""
        return (
            (self._num_bits == 4 and self._block_sizes)
            or (self._num_bits == (2, 1) and self._block_sizes)
            or self._num_bits in ((4, 3), 8)  # Int8
        )

    def _real_quantize(self, inputs):
        assert self._is_real_quantize_support(), "Real quantization not supported for this format."

        buffer_to_register = {}
        # Check MX formats first (before FP8) since MXFP8 also has num_bits=(4,3)
        if (
            self._block_sizes
            and self._block_sizes.get("scale_bits") == (8, 0)
            and self._block_sizes.get("type") == "dynamic"
        ):
            # MX quantization (MXFP4/MXFP8)
            if self._num_bits == (2, 1):
                # MXFP4
                outputs, scales = MXFP4QTensor.quantize(inputs, self._block_sizes[-1])
                buffer_to_register["_scale"] = scales
            elif self._num_bits == (4, 3):
                # MXFP8
                assert self._block_sizes[-1] == MXFP8QTensor.BLOCK_SIZE, (
                    f"MXFP8 requires block size {MXFP8QTensor.BLOCK_SIZE}, "
                    f"got {self._block_sizes[-1]}"
                )
                outputs, scales = MXFP8QTensor.quantize(inputs)
                buffer_to_register["_scale"] = scales
            else:
                raise ValueError(
                    f"Unsupported MX format: num_bits={self._num_bits}. "
                    f"Expected (2, 1) for MXFP4 or (4, 3) for MXFP8."
                )
        elif self._num_bits == (4, 3):
            # FP8 quantization (non-MX)
            # For per-tensor/per-channel quantization, we might need amax which is synced across all ranks
            # For blockwise quantization, amax will be recomputed in the kernel
            use_amax = self.amax is not None and not (self._block_sizes and self.amax.numel() == 1)
            outputs, _scale = FP8QTensor.quantize(
                inputs,
                axis=self._axis,
                block_sizes=self._block_sizes,
                scales=self.amax / 448.0 if use_amax else None,
            )
            buffer_to_register["_scale"] = _scale
        elif self._num_bits == 8:
            outputs, _scale = INT8QTensor.quantize(
                inputs, axis=self._axis, block_sizes=self._block_sizes
            )
            buffer_to_register["_scale"] = _scale
        elif self._block_sizes.get("scale_bits", 0) == 8 and self._block_sizes.get(
            "scale_block_sizes", None
        ):
            # NF4 double quantization
            # Return real quantized tensor class and store scales inside the TensorQuantizer
            outputs, scales = NF4QTensor.quantize(
                inputs, self._block_sizes[-1], self._block_sizes["scale_block_sizes"][-1]
            )

            _scale, _double_scale, _scale_zeros = NF4QTensor.double_quantization(
                scales,
                self._block_sizes["scale_block_sizes"][-1],
                self._block_sizes["scale_bits"],
            )
            buffer_to_register["_scale"] = _scale
            buffer_to_register["_double_scale"] = _double_scale
            buffer_to_register["_scale_zeros"] = _scale_zeros
        elif self._block_sizes.get("scale_bits") == (4, 3):
            # NVFP4 default quantization
            # Return real quantized tensor and store scales inside TensorQuantizer
            outputs, _weights_scaling_factor, _weights_scaling_factor_2 = NVFP4QTensor.quantize(
                inputs,
                self._block_sizes[-1],
                weights_scaling_factor_2=self.amax.float() / (448.0 * 6.0)
                if self.amax is not None
                else None,
                try_tensorrt=True,
            )
            buffer_to_register["_scale"] = _weights_scaling_factor
            buffer_to_register["_double_scale"] = _weights_scaling_factor_2
        else:
            outputs, _scale = INT4QTensor.quantize(inputs, self._block_sizes[-1])
            buffer_to_register["_scale"] = _scale
        for k, v in buffer_to_register.items():
            self._set_buffer(k, v)

        # We assume _real_quantize is called when compress the model weights, and we set
        # self._dequantize to True so that future forward call will only do dequantize.
        self._dequantize = True
        return outputs

    def _fake_quantize(self, inputs):
        """Fake quantization."""
        if self.backend is not None:
            if self.backend not in _QUANT_FUNCTIONAL_BACKENDS:
                raise KeyError(f"Quant backend '{self.backend}' is not registered.")
            entrypoint = _QUANT_FUNCTIONAL_BACKENDS[self.backend]
            return entrypoint(inputs, self)

        amax = None
        if not self.is_mx_format:
            amax = self._get_amax(inputs)

        if self.block_sizes is not None and self.block_sizes.get("type", "static") == "dynamic":
            # Dynamic block quantization
            block_size = self.block_sizes.get(-1, None) or self.block_sizes.get(
                inputs.dim() - 1, None
            )
            if block_size is None:
                raise ValueError("block size for dynamic quantization not found.")

            outputs = dynamic_block_quant(
                inputs,
                block_size,
                amax,
                self._get_bias(inputs),
                self._num_bits,
                self.block_sizes.get("scale_bits", None),
                getattr(self, "_trt_high_precision_dtype", None),
                getattr(self, "_onnx_quantizer_type", None),
                self._pass_through_bwd,
            )
        elif isinstance(self._num_bits, tuple):
            # Float-point quantization, e.g., FP8
            E, M = self._num_bits  # noqa: N806

            outputs = scaled_e4m3(
                inputs,
                amax,
                self._get_bias(inputs),
                E,
                M,
                self._trt_high_precision_dtype,
                self._pass_through_bwd,
            )

        else:
            # Integer quantization, e.g., INT8
            outputs = fake_tensor_quant(
                inputs,
                amax,
                self._get_bias(inputs),
                self._num_bits,
                self._unsigned,
                self._narrow_range,
                self._trt_high_precision_dtype,
                self._pass_through_bwd,
                self.block_sizes.get(-1) if self.block_sizes else None,
                self.axis[0] if isinstance(self.axis, tuple) else self.axis,
            )
        return outputs

    def _check_onnx_readiness(self, inputs):
        """Check if quantizer is ready for ONNX export."""
        if not self.block_sizes or self.block_sizes.get("scale_bits", None) != (8, 0):
            assert hasattr(self, "_amax"), (
                "Quantizer has not been calibrated. ONNX export requires the quantizer to be"
                " calibrated. Calibrate and load amax before exporting to ONNX."
            )

        if self._if_calib:
            warnings.warn(
                "Quantizer is in calibration mode. "
                "Please complete calibration before exporting to ONNX for correct results."
            )

        amax = self._get_amax(inputs)

        # We only support scalar amax for E4M3 ONNX export
        if isinstance(self.num_bits, tuple):
            assert amax.numel() == 1, (
                "E4M3 supports ONNX export only for per-tensor quantization."
                " Per-tensor quantization requires scalar amax. "
                f"Received non-scalar amax of shape: {amax.shape}"
            )

    def _setup_for_blockquant(self, inputs: torch.Tensor):
        # Get reshape sizes and paddings for block-quantization
        def get_axis_quant_params(ax):
            ax = ax if ax in self.block_sizes else ax - inputs.dim()
            bsize = self.block_sizes.get(ax, None)
            padding, ax_slice = None, None
            if bsize is not None and inputs.shape[ax] % bsize != 0:
                padding = (bsize - (inputs.shape[ax] % bsize), 0)
                ax_slice = slice(inputs.shape[ax])
            return bsize, padding, ax_slice

        def set_quant_params(axis, block_reshape_size, padding, slices, amax_shape=None):
            self._axis = tuple(axis)
            if hasattr(self, "_calibrator"):
                self._calibrator._axis = self._axis
            self._original_shape = inputs.shape
            self._block_reshape_size = torch.Size(block_reshape_size)
            if padding is not None:
                self._padding = tuple(padding)
                self._original_shape = F.pad(inputs, self._padding, "constant", 0).shape
            if slices is not None:
                self._slices = slices
            if amax_shape:
                self._amax_shape_for_export = amax_shape

        # Reshape size have already been set
        if hasattr(self, "_block_reshape_size"):
            return

        reshape_size, quantize_axis, paddings, slices = [], [], [], []

        # special handling for block-quantization along the last axis:
        # flatten the input for faster execution
        if (self.block_sizes.get(inputs.dim() - 1, None) or self.block_sizes.get(-1, None)) and len(
            QuantizerAttributeConfig._get_block_quant_axes_and_sizes(self.block_sizes)
        ) == 1:
            bsize, padding, ax_slice = get_axis_quant_params(inputs.dim() - 1)
            slices = None if ax_slice is None else (*(slice(None),) * (inputs.dim() - 1), ax_slice)
            padding = padding if not padding else tuple(reversed(padding))
            amax_shape_for_export = (*(inputs.shape[:-1]), -1)
            set_quant_params((0,), (-1, bsize), padding, slices, amax_shape_for_export)
            return

        for ax in range(inputs.dim()):
            bsize, padding, ax_slice = get_axis_quant_params(ax)
            paddings.append(padding)
            slices.append(ax_slice)
            if bsize is not None:
                reshape_size.extend([math.ceil(inputs.shape[ax] / bsize), bsize])
                quantize_axis.extend([True, False])
            else:
                reshape_size.append(inputs.shape[ax])
                quantize_axis.append(True)

        quant_axis = [i for i in range(len(quantize_axis)) if quantize_axis[i]]

        slices = (
            None if all(s is None for s in slices) else [s if s else slice(None) for s in slices]
        )

        if all(p is None for p in paddings):
            paddings = None
        else:
            new_paddings = []
            for padding in paddings:
                if not (new_paddings or padding):
                    continue
                new_paddings.extend(padding if padding else (0, 0))
            paddings = tuple(reversed(new_paddings))

        set_quant_params(quant_axis, reshape_size, paddings, slices)

    def _process_for_blockquant(self, inputs: torch.Tensor):
        if hasattr(self, "_padding"):
            inputs = F.pad(inputs, self._padding, "constant", 0)

        if inputs.shape != self._original_shape:
            raise ValueError(
                f"Input shape has changed from {self._original_shape} to {inputs.shape}."
                " Block-quantization requires a fixed input shape."
            )
        inputs = inputs.reshape(self._block_reshape_size)
        return inputs

    def _reset_to_original_shape(self, outputs: torch.Tensor):
        outputs = outputs.reshape(self._original_shape)
        if hasattr(self, "_slices"):
            outputs = outputs[self._slices]
        return outputs

    def _block_sizes_to_axis(self, x: torch.Tensor):
        """Convert block_sizes to axis in per-channel/tensor quantization.

        For example, for input tensor with shape (B, T, H),
        {"block_sizes": {-1: None, -3: None}} equals to {axis: (-2)}, amax shape: (1, T, 1),
        {"block_sizes": {-1: None, -2: None, -3: None}} equals to {axis: None}, amax shape: (1, T, 1)
        """
        block_sizes = self._block_sizes
        if block_sizes is None:
            return

        def _check_per_channel_block_sizes(block_sizes):
            # Check per-channel/block quant
            return all(v is None for k, v in block_sizes.items() if isinstance(k, int))

        if _check_per_channel_block_sizes(block_sizes):
            # Convert block_sizes to axis
            assert self.axis is None, "Axis and block_sizes are both set."
            axis = tuple(k if k >= 0 else k + x.dim() for k in block_sizes if isinstance(k, int))
            self.axis = tuple(i for i in range(x.dim()) if i not in axis) or None

            # remove block_sizes
            self._block_sizes = None

    def export_amax(self) -> torch.Tensor | None:
        """Export correctly formatted/shaped amax."""
        if self.block_sizes is not None and self.block_sizes.get("type", None) == "dynamic":
            return self.amax

        if self.amax is None:
            return None

        if not hasattr(self, "_amax_shape_for_export"):
            amax = self.amax
        else:
            amax = self.amax.reshape(self._amax_shape_for_export)
        amax[amax == 0] = self.maxbound
        amax = torch.nan_to_num(amax, nan=self.maxbound)
        clamp_min, clamp_max = torch.finfo(amax.dtype).tiny, torch.finfo(amax.dtype).max
        amax = amax.clamp(min=clamp_min, max=clamp_max)

        self.validate_attr(attr_name="_amax", attr_value=amax)

        if self.block_sizes is None:
            # tensorrt_llm assumes the scaling_factor dim >= 1 for per-tensor.
            if self.axis is None:
                amax = amax.unsqueeze(0)

            # If single-axis quantization, squeeze amax
            elif isinstance(self.axis, int) or (
                isinstance(self.axis, (list, tuple)) and len(self.axis) == 1
            ):
                amax = amax.squeeze()

        return amax

    def forward(self, inputs):
        """Apply tensor_quant function to inputs.

        Args:
            inputs: A Tensor of type float32/float16/bfloat16.

        Returns:
            outputs: A Tensor of type output_dtype
        """
        if DTensor is not None and isinstance(inputs, DTensor):
            # TensorQuantizer only handles regular non-DTensor inputs
            device_mesh, placements = inputs.device_mesh, inputs.placements
            outputs = self.forward(inputs.to_local())
            return DTensor.from_local(outputs, device_mesh, placements)

        if isinstance(inputs, (BaseQuantizedTensor, QTensorWrapper)):
            assert self._dequantize, "No dequantization stats in the tensor quantizer."
            return self.dequantize(inputs)

        # Early return if nothing is collected during the forward (e.g. MoE)
        if inputs.numel() == 0:
            return inputs

        # Activation scaling for smoothquant
        if self.pre_quant_scale is not None:
            inputs = inputs * self.pre_quant_scale

        # Rotating the input
        if self._rotate:
            inputs = normalized_hadamard_transform(inputs)

        if self._disabled:
            # if quantizer is disabled, we still need to track the input dtype for saving the model
            # TODO: This is a temporary solution and needs to be removed once megatron supports
            # non-homogeneous layers
            self._input_dtype = inputs.dtype if hasattr(inputs, "dtype") else None
            return inputs

        if (
            not is_torch_export_mode()
            and not torch.compiler.is_compiling()
            and GLOBALS.in_onnx_export
        ):
            # GLOBALS could break TorchDynamo for some Pytorch versions (i.e., 2.3.0)
            self._check_onnx_readiness(inputs)

        if self.block_sizes is not None and self._fake_quant:
            # To support the new block_sizes representation for per-channel quantization,
            # convert the dim dict in block_sizes to axis.
            # The axis attribute is still preserved for backward compatibility.
            self._block_sizes_to_axis(inputs)

        if self.is_static_block_quant:
            # Tensor reshaping is required for static block quantization
            # Tensor shapes are handled separately by the quantization kernels for dynamic block quantization
            self._setup_for_blockquant(inputs)
            inputs = self._process_for_blockquant(inputs)

        outputs = inputs

        block_size = None
        if self._if_calib and not self._dynamic:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            # Shape is only known when it sees the first tensor
            if self.block_sizes is not None and self.block_sizes.get("type", None) == "dynamic":
                block_size = self.block_sizes.get(-1, None) or self.block_sizes.get(
                    inputs.dim() - 1, None
                )
                assert block_size is not None, "block size for dynamic quantization not found."

            self.collect(inputs)

        if self._if_quant:
            # Check if the input tensor is contiguous
            # Non-contiguous tensors will generate incorrect FP4 quantization results
            if hasattr(inputs, "is_contiguous") and not inputs.is_contiguous():
                inputs = inputs.contiguous()
            if self.fake_quant:
                with same_device_as(inputs):
                    outputs = self._fake_quantize(inputs)
            elif not self._dequantize:
                outputs = self._real_quantize(inputs)
            else:
                raise ValueError(
                    "self._dequantize is True and self.fake_quant is False. "
                    "This case should have been handled."
                )

        if self.is_static_block_quant:
            outputs = self._reset_to_original_shape(outputs)

        return outputs

    def _short_amax(self, fmt=".4f"):
        """Short description of amax.

        Returns:
            'dynamic': if _amax is not registered
            'amax': if _amax is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if self.is_mx_format:
            return "None"
        if not hasattr(self, "_amax"):
            return "dynamic"
        if self._amax is None:
            return "None"
        if self._amax.is_meta:
            return "meta"
        return self._short_tensor(self._amax, fmt)

    def _short_tensor(self, tensor: torch.Tensor, fmt=".4f"):
        """Short description of tensor."""
        if tensor.numel() == 1:
            return f"{tensor.item():{fmt}}"
        return f"[{tensor.min().item():{fmt}}, {tensor.max().item():{fmt}}]({tensor.numel()})"

    def extra_repr(self):
        """Set the extra information about this module."""
        if self._disabled:
            s = "disabled"
            s += (
                f" pre_quant_scale={self._short_tensor(self.pre_quant_scale)}"
                if self.pre_quant_scale is not None
                else ""
            )
            return "disabled"
        s = f"{'unsigned ' if self._unsigned else ''}{self._num_bits} bit"
        s += " narrow" if (self._narrow_range) else ""
        s += " fake" if (self._fake_quant) else ""
        if self.block_sizes is not None:
            s += f" block_sizes={self._block_sizes},"
        else:
            s += f" axis={self._axis}" if self._axis is not None else " per-tensor"
        s += f" amax={self._short_amax()}"
        s += (
            f" pre_quant_scale={self._short_tensor(self.pre_quant_scale)}"
            if self.pre_quant_scale is not None
            else ""
        )
        s += " rotated" if self._rotate else ""
        s += (
            f" calibrator={self._calibrator.__class__.__name__}"
            if (self._calibrator is not None)
            else ""
        )
        if self._bias:
            s += f" bias={self._bias}"

        s += " quant" if (self._if_quant) else ""
        s += " calib" if (self._if_calib) else ""
        s += (
            f" backend={self.backend}, extra_args={self.backend_extra_args}"
            if self.backend is not None
            else ""
        )
        return s

    def _get_properties_for_modelopt_state(self):
        return (
            self.__dict__.keys()
            - nn.Module().__dict__.keys()
            - self._skip_properties_for_save_restore
        )

    def _get_pytorch_state_metadata(self):
        """Get Pytorch state metadata.

        Current we only store the shape of the state_dict values.
        """
        metadata = {"params": {}, "buffers": {}}
        for k, v in self._parameters.items():
            metadata["params"][k] = {"shape": v.shape, "dtype": v.dtype}
        for k, v in self._buffers.items():
            if k in self._non_persistent_buffers_set:
                continue
            metadata["buffers"][k] = {"shape": v.shape, "dtype": v.dtype}
        return metadata

    def _del_pytorch_state(self):
        # Lets delete the parameters and buffers
        self._parameters.clear()
        self._buffers.clear()
        self._non_persistent_buffers_set.clear()

    def _reset_pytorch_state_from_metadata(self, metadata: dict[str, Any]):
        # Lets delete existing parameters and buffers and create fresh ones
        self._del_pytorch_state()
        for k, v in metadata.get("params", {}).items():
            dtype = v.get("dtype", None)
            self.register_parameter(k, nn.Parameter(torch.empty(v["shape"], dtype=dtype)))
        for k, v in metadata.get("buffers", {}).items():
            dtype = v.get("dtype", None)
            self.register_buffer(k, torch.empty(v["shape"], dtype=dtype))

    def get_modelopt_state(self, properties_only: bool = False) -> dict[str, Any]:
        """Get meta state to be saved in checkpoint.

        If `properties_only` is True, only the quantizer properties such as `num_bits`, `axis` etc are included.
        For restoring the quantizer fully including the parameters and buffers, use `properties_only=False`.
        """
        modelopt_state = {}
        for k in self._get_properties_for_modelopt_state():
            modelopt_state[k] = getattr(self, k)

        if properties_only:
            return modelopt_state

        modelopt_state["_pytorch_state_metadata"] = self._get_pytorch_state_metadata()

        return modelopt_state

    def set_from_modelopt_state(self, modelopt_state, properties_only: bool = False):
        """Set meta state from checkpoint.

        If `properties_only` is True, only the quantizer properties such as `num_bits`, `axis` etc are included.
        For restoring the quantizer fully including the parameters and buffers, use `properties_only=False`.
        """
        # Set all properties except the skip properties; this is done for backward compatibility
        for key in modelopt_state.keys() - self._skip_properties_for_save_restore:
            setattr(self, key, modelopt_state[key])

        # Set the calibrator properties
        # TODO: This might not be sufficient for the custom calibrators - however there is no use-case for it yet
        for key in ["_num_bits", "_axis", "_unsigned"]:
            setattr(self._calibrator, key, getattr(self, key))

        if not properties_only:
            self._reset_pytorch_state_from_metadata(
                modelopt_state.get("_pytorch_state_metadata", {})
            )

    def sync_amax_across_distributed_group(self, parallel_group: DistributedProcessGroup):
        """Synchronize the amax across all ranks in the given group."""
        if parallel_group.is_initialized() and getattr(self, "_amax", None) is not None:
            try:
                dist.all_reduce(self._amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
            except RuntimeError as e:
                # This error happens if the distributed backend is using GPU and
                # the tensor is not on GPU (or vice versa).
                warnings.warn(
                    f"Failed to synchronize amax: {e}, probably because the tensor is on a device which is not"
                    "supported by the current distributed backend. This warning can be ignored"
                    "if happening during modelopt restore."
                )

    @contextlib.contextmanager
    def disable_pre_quant_scale(self):
        """Context manager to turn off pre_quant_scale inside this quantizer."""
        was_enabled = self._enable_pre_quant_scale
        self._enable_pre_quant_scale = False
        try:
            yield
        finally:
            self._enable_pre_quant_scale = was_enabled

    def collect(self, inputs) -> None:
        """Collect calibration data."""
        if not self._if_calib or self._dynamic:
            return

        # Collect bias data if bias calibration is enabled
        if self.bias_calibrator is not None and self.bias_type == "static":
            self.bias_calibrator.collect(inputs)
            inputs = inputs - self.bias_calibrator.compute_bias()

        self._calibrator.collect(inputs)

    def _set_buffer(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.register_buffer(key, value)


class NVFP4StaticQuantizer(TensorQuantizer):
    """TensorQuantizer for NVFP4 static block quantization with two-level scaling.

    Uses _global_amax and inherited _amax for per-block amax values.
    """

    @classmethod
    def from_tensor_quantizer(
        cls, tq: TensorQuantizer, global_amax: torch.Tensor | None = None
    ) -> "NVFP4StaticQuantizer":
        """Convert a TensorQuantizer to NVFP4StaticQuantizer in-place.

        Args:
            tq: The TensorQuantizer to convert.
            global_amax: Optional global amax value to set on the quantizer.
        """
        if isinstance(tq, cls):
            if global_amax is not None:
                tq.global_amax = global_amax
            return tq
        tq.__class__ = cls
        tq._is_nvfp4_static_quantizer = True
        if global_amax is not None:
            tq.global_amax = global_amax
        return tq

    @property
    def global_amax(self):
        """Return global_amax for quantization."""
        if not hasattr(self, "_global_amax"):
            return None
        return self._global_amax

    @global_amax.setter
    def global_amax(self, value):
        if value is None:
            if hasattr(self, "_global_amax"):
                self._global_amax = None
            return
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not hasattr(self, "_global_amax") or self._global_amax is None:
            self.register_buffer("_global_amax", value.clone().detach())
        else:
            self._global_amax.data.copy_(value.clone().detach().to(self._global_amax.device))

    def _fake_quantize(self, inputs):
        """Fake quantization using two-level scaling with _amax and _global_amax."""
        if self.amax is not None:
            return static_blockwise_fp4_fake_quant(
                inputs,
                self.amax,
                self.global_amax,  # Can be None, will be computed internally
                True,  # quantize_block_scales
                inputs.dtype,
                self._pass_through_bwd,
            )
        return super()._fake_quantize(inputs)


class SequentialQuantizer(nn.Sequential):
    """A sequential container for  :class:`TensorQuantizer` modules.

    This modules is used to quantize a tensor in multiple formats sequentially. It takes as input
    :class:`TensorQuantizer` modules and containerize them similar to :class:`torch.nn.Sequential`.

    We delegate certain properties and methods to all contained quantizers.
    In the case of conflicts, the first quantizer's property or method takes priority.

    `SequentialQuantizer` is useful in cases like INT4 weights, FP8 activations where weight quantization is not the
    same as the gemm quantization. It allows for applying multiple quantization formats to the same tensor in sequence.

    Use `SequentialQuantizer` methods in lower level implementations for better code organization and readability.

    Args:
        quantizers (TensorQuantizer): :class:`TensorQuantizer` modules to be added to the container.

    """

    _delegated_properties = ["fake_quant", "is_enabled"]
    _delegated_methods = [
        "reset_amax",
        "disable",
        "enable",
        "load_calib_amax",
        "load_calib_bias",
    ]

    def __init__(self, *quantizers: TensorQuantizer):
        """Initialize SequentialQuantizer module."""
        super().__init__(*quantizers)
        assert all(isinstance(q, TensorQuantizer) for q in self), (
            "All quantizers must be a TensorQuantizer."
        )

    def __getattr__(self, name):
        """Delegate properties and methods to all contained quantizers."""
        if name in self._delegated_properties:
            # Return the property of the first quantizer
            return getattr(self[0], name)

        if name in self._delegated_methods:

            def method_wrapper(*args, **kwargs):
                outputs = getattr(self[0], name)(*args, **kwargs)
                for quantizer in self[1:]:
                    outputs = getattr(quantizer, name)(*args, **kwargs)
                return outputs

            return method_wrapper

        # Defer to super class for attributes not handled here
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in self._delegated_properties:
            for quantizer in self:
                setattr(quantizer, name, value)
        else:
            super().__setattr__(name, value)

    def get_modelopt_state(self) -> dict[str, Any]:
        """Get meta state to be saved in checkpoint."""
        return {"num_quantizers": len(self), "is_sequential_quantizer": True}

    def set_from_attribute_config(
        self,
        attributes: list[dict[str, Any] | QuantizerAttributeConfig]
        | dict[str, Any]
        | QuantizerAttributeConfig,
    ):
        """Set the attributes of contained quantizers from a list of attribute_dicts."""
        if not isinstance(attributes, (list, tuple)):
            assert isinstance(attributes, (dict, QuantizerAttributeConfig)), (
                "attributes must be a list or a dict."
            )
            attributes = [attributes] * len(self)

        for attribute, quantizer in zip(attributes, self):
            quantizer.set_from_attribute_config(attribute)

    @staticmethod
    @contextlib.contextmanager
    def convert_to_single_quantizer(model, indx: int = 0):
        """Replace instances of :class:`SequentialQuantizer` in the model with single `TensorQuantizer` quantizer.

        The quantizer indexed by ``indx`` from the sequential quantizer is used to replace it.
        This method is useful for individually calibrating the quantizers in a sequential quantizer.
        """
        original_sequential_quantizers: dict[nn.Module, list] = {}
        for name, module in list(model.named_modules()):
            if isinstance(module, SequentialQuantizer):
                assert len(module) > indx
                parent_module = model.get_submodule(name.rpartition(".")[0])
                if parent_module not in original_sequential_quantizers:
                    original_sequential_quantizers[parent_module] = []
                original_sequential_quantizers[parent_module].append(
                    (name.rpartition(".")[-1], module)
                )
                setattr(parent_module, name.rpartition(".")[-1], module[indx])

        yield

        for (
            parent_module,
            sequential_quantizers_list,
        ) in original_sequential_quantizers.items():
            for name, sequential_quantizer in sequential_quantizers_list:
                setattr(parent_module, name, sequential_quantizer)
