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

import pytest
import torch.nn as nn

pytest.importorskip("torchprofile")

from modelopt.torch.nas.utils import inference_flops


@pytest.mark.parametrize(
    ("in_channel", "out_channel", "kernel_size", "groups", "data_x", "data_y"),
    [
        (2, 4, 3, 2, 3, 5),
        (3, 5, 1, 1, 2, 2),
        (4, 2, 3, 1, 1, 2),
    ],
)
def test_flops(in_channel, out_channel, kernel_size, groups, data_x, data_y) -> None:
    assert (kernel_size - 1) % 2 == 0
    pad = (kernel_size - 1) // 2
    conv_module = nn.Conv2d(
        in_channel, out_channel, kernel_size=kernel_size, groups=groups, bias=False, padding=pad
    )
    input_data_shape = (1, in_channel, data_x, data_y)
    out_elements = out_channel * data_x * data_y
    per_element_filters = in_channel * kernel_size * kernel_size // groups
    desired_output = 2 * out_elements * per_element_filters
    assert inference_flops(conv_module, data_shape=input_data_shape, unit=1) == desired_output
