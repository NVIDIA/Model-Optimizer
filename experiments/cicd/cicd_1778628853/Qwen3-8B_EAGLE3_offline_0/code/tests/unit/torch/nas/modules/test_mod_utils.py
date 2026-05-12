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

import torch

from modelopt.torch.nas.modules.utils import get_sliced_tensor_by_slices


def test_get_sliced_tensor_by_slices_simple():
    tensor = torch.randn(2, 3, 4, 5, 6)

    sliced = get_sliced_tensor_by_slices(tensor, slices=[slice(2), slice(3), slice(4)])
    assert torch.all(sliced == tensor)


def test_get_sliced_tensor_by_slices_complex():
    tensor = torch.randn(2, 3, 4, 5, 6)

    sliced = get_sliced_tensor_by_slices(
        tensor, slices=[slice(None), [2, 1], slice(2, 4), [0, 1, 2], slice(0, 6, 2)]
    )
    assert torch.all(sliced == tensor[:, [2, 1], 2:4, :3, :6:2])
    assert sliced.is_contiguous()
