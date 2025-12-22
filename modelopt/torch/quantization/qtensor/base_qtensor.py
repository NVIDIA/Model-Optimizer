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

"""Base Class for Real Quantized Tensor."""

import enum

import torch
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor

from modelopt.torch.quantization.utils import fsdp2_aware_weight_update, patch_fsdp_mp_dtypes


class QTensorType(enum.Enum):
    """Enumeration for defining types of quantization."""

    INT4 = 1
    INT8 = 2
    FP8 = 3
    NF4 = 4


__all__ = ["BaseQuantizedTensor", "QTensorWrapper", "pack_real_quantize_weight"]


class BaseQuantizedTensor:
    """Base class for quantized tensors, providing methods for quantization and dequantization.

    This class should be subclassed to implement specific types of quantized tensors. It handles the
    storage of quantized data along with the necessary configurations and original attributes.

    Attributes:
        original_meta_tensor (torch.Tensor): Original meta to keep attributes of original tensors.
        quantized_data (torch.Tensor): Storage for the quantized tensor data. Quantized_data dtype is
            customized per QuantizedTensor implementation.
    """

    _quantized_data: torch.Tensor

    def __init__(
        self,
        original_shape: torch.Size,
        original_dtype: torch.dtype,
        quantized_data: torch.Tensor,
    ):
        """Initialize data attributes."""
        self.metadata = {
            "shape": original_shape,
            "dtype": original_dtype,
        }
        self._quantized_data = quantized_data

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int):
        """Pack a fake torch.Tensor into a real quantized tensor.

        Args:
            fake_quant_tensor (torch.Tensor): The fake quantized tensor.

        Returns:
            A real quantized tensor, scales.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def dequantize(self, dtype: torch.Tensor = None, **kwarg):
        """Converts the quantized tensor back to a standard torch.Tensor.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")


class QTensorWrapper(torch.nn.Parameter):
    """A wrapper class for quantized tensors to make them compatible with torch.nn.Parameter.

    Args:
        qtensor (BaseQuantizedTensor): The quantized tensor to be wrapped.
    """

    def __new__(cls, qtensor: BaseQuantizedTensor | torch.Tensor, metadata: dict | None = None):
        """Create a new QTensorWrapper instance."""
        quantized_tensor = (
            qtensor._quantized_data if isinstance(qtensor, BaseQuantizedTensor) else qtensor
        )
        instance = super().__new__(cls, quantized_tensor, requires_grad=False)
        if metadata is None:
            instance.metadata = qtensor.metadata
            instance.metadata["qtensor_class"] = qtensor.__class__
        else:
            assert all(key in metadata for key in ["qtensor_class", "shape", "dtype"]), (
                f"metadata: {metadata}"
            )
            instance.metadata = metadata
        return instance

    def dim(self):
        """Return the number of dimensions of the meta_tensor."""
        return len(self.metadata["shape"])

    def to(self, *args, **kwargs):
        """Override the `to` method to move real quantized tensors to the specified device."""
        changing_device, changing_dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if changing_device:
            self.data = self.data.to(device=changing_device)
        dtype = changing_dtype if changing_dtype else self.metadata["dtype"]
        return QTensorWrapper(
            self.metadata["qtensor_class"](self.metadata["shape"], dtype, self.data)
        )

    def get_qtensor(self):
        """Get the quantized tensor class from QTensorWrapper."""
        return self.metadata["qtensor_class"](
            self.metadata["shape"], self.metadata["dtype"], self.data
        )

    def get_state(self):
        """Get the state of the QTensorWrapper."""
        return {
            "metadata": self.metadata,
            "quantized_data.shape": self.data.shape,
            "quantized_data.dtype": self.data.dtype,
        }


class QFSDPParam(FSDPParam):
    """A Quantized FSDPParam class to make weight updates compatible with BaseQuantizedTensor and QTensorWrapper.

    With this class, we can keep track of the quantized tensor's metadata when compressing the weights
    and recreate the QTensorWrapper with the correct metadata, when unsharding the FSDPModule.

    Args:
        qtensor (BaseQuantizedTensor): The quantized tensor to be wrapped.
    """

    def __init__(self, *args, **kwargs):
        # Store qtensor information
        self.metadata = args[0].metadata
        super().__init__(*args, **kwargs)
        self.init_dtype_attrs(self.mp_policy)

    def _setattr_on_modules(self, param: torch.nn.Parameter) -> None:
        if not isinstance(param, DTensor):
            # Create a QTensorWrapper with the correct metadata during unsharding
            param = QTensorWrapper(param, metadata=self.metadata)
        super()._setattr_on_modules(param)


# Function to dynamically override load_state_dict
def dynamically_update_state_methods(module):
    # Original method
    original_load_from_state_dict = module._load_from_state_dict

    def custom_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Override _load_from_state_dict to handle custom parameters dynamically."""
        args_list = list(args)
        deleted_stat_dict = {}

        # Load parameters
        for name, param in self.named_parameters():
            if prefix + name in state_dict and isinstance(param, QTensorWrapper):
                if param.data.is_meta:
                    # Wrap the loaded tensor in a QTensorWrapper
                    param = QTensorWrapper(state_dict[prefix + name], metadata=param.metadata)
                    self.register_parameter(name, param)
                else:
                    param.copy_(state_dict[prefix + name])
                deleted_stat_dict[prefix + name] = state_dict[prefix + name]
                del state_dict[prefix + name]

        # Set strict=False because weight keys are removed
        kwargs = {}
        if len(args_list) > 3:
            args_list[1] = False
        else:
            kwargs = {"strict": False}
        original_load_from_state_dict(state_dict, prefix, *args_list, **kwargs)
        state_dict.update(**deleted_stat_dict)

    module._load_from_state_dict = custom_load_from_state_dict.__get__(module, type(module))


def pack_real_quantize_weight(module, force_quantize: bool = False):
    """Pack real quantized tensors to a compressed format and set proper load_state_dict function."""
    # Import SequentialQuantizer here to avoid circular import
    from ..nn import SequentialQuantizer

    def _compress_and_update_module_weight(module):
        """Compresses and updates module weights if quantizer is enabled. Returns True when compression is applied."""
        if hasattr(module, "weight") and (module.weight is None or module.weight.is_meta):
            # We dont compress meta tensors or None
            return False
        if (
            hasattr(module, "weight_quantizer")
            and module.weight_quantizer.is_enabled
            and not module.weight_quantizer._fake_quant
            and module.weight.element_size() > 1
        ):
            if force_quantize:
                module.weight_quantizer._dequantize = False

            real_quant_tensor = module.weight_quantizer(module.weight)
            module.weight = QTensorWrapper(real_quant_tensor)
            return True

        return False

    with (
        SequentialQuantizer.convert_to_single_quantizer(module),
        torch.no_grad(),
        patch_fsdp_mp_dtypes(),
    ):
        for name, m in module.named_modules():
            if name != "":
                with fsdp2_aware_weight_update(module, m):
                    _compress_and_update_module_weight(m)
