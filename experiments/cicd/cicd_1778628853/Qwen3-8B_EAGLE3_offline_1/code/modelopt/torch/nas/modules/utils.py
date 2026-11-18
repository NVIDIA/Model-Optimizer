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

"""Internal module for utility functions."""

import torch

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.hparam import Hparam


def get_sliced_tensor_by_slices(
    tensor: torch.Tensor | None,
    slices: list[Hparam.ActiveSlice],
) -> torch.Tensor | None:
    """Get the tensor based on the active slice."""
    if tensor is None:
        return tensor

    # check if we can return the original tensor
    if all(
        isinstance(s, slice) and (s.stop is None or s.indices(s.stop) == (0, tensor.shape[i], 1))
        for i, s in enumerate(slices)
    ):
        return tensor

    # slice tensor with minimal number of index operations
    tensor_sliced = tensor
    for i, _ in enumerate(slices):
        if sum(not isinstance(s, slice) for s in slices) < 2:
            tensor_sliced = tensor_sliced[tuple(slices)]
            break
        tensor_sliced = tensor_sliced[tuple(slices[: i + 1])]
        slices[i] = slice(None)  # replace with a vanilla slice ("[:]") for next slicing iteration

    # return sliced, contiguous tensor
    return tensor_sliced.contiguous()


def get_sliced_tensor(
    mod: DynamicModule,
    tensor: torch.Tensor | None,
    *hp_names: str | None,
) -> torch.Tensor | None:
    """Get the tensor based on the slices."""
    slices = [
        mod.get_hparam(hp_name).active_slice if hp_name else slice(None) for hp_name in hp_names
    ]
    return get_sliced_tensor_by_slices(tensor, slices)
