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

"""Utility functions for PyTorch tensors."""

from collections import abc
from contextlib import nullcontext

import numpy as np
import torch

__all__ = [
    "numpy_to_torch",
    "same_device_as",
    "to_empty_if_meta_device",
    "torch_detach",
    "torch_to",
    "torch_to_numpy",
]


def same_device_as(inputs: torch.Tensor):
    """Return a context manager that sets the CUDA device to be the same as the input tensor.

    Returns a null context if the tensor is on CPU or on the same device as the current CUDA device.
    """
    if not inputs.is_cuda or inputs.device.index == torch.cuda.current_device():
        return nullcontext()
    return torch.cuda.device(inputs.device.index)


def torch_to(data, *args, **kwargs):
    """Try to recursively move the data to the specified args/kwargs."""
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_to(val, *args, **kwargs) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_to(val, *args, **kwargs) for k, val in data.items()}
    return data


def torch_detach(data):
    """Try to recursively detach the data from the computation graph."""
    if isinstance(data, torch.Tensor):
        return torch.detach(data)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_detach(val) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_detach(val) for k, val in data.items()}
    return data


def torch_to_numpy(inputs: list[torch.Tensor]) -> list[np.ndarray]:
    """Convert torch tensors to numpy arrays."""
    return [t.detach().cpu().numpy() for t in inputs]


def numpy_to_torch(np_outputs: list[np.ndarray]) -> list[torch.Tensor]:
    """Convert numpy arrays to torch tensors."""
    return [torch.from_numpy(arr) for arr in np_outputs]


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().

    Officially, torch suggests to_empty() for meta device materialization. Under the hood,
    torch.empty_like() is applied to all parameters or buffers (see _apply). This may
    accidentally overwrite buffers with precomputed values during construction. Given the
    goal is to only materialize those tensors on meta device, this function checks the
    device first and only move the tensor to the destination if it is not on meta device.

    Args:
        module: The target module to apply this transformation.
        device: The desired device of the parameters
            and buffers in this module.
        recurse: Whether parameters and buffers of submodules should
            be recursively moved to the specified device.
    """

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            return torch.empty_like(tensor, device=device)
        else:
            return tensor.to(device)

    return module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)
