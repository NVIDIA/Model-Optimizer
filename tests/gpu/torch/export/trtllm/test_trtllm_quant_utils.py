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

"""Unit tests for :mod:`modelopt.torch.export.trtllm.quant_utils`.

These helpers are reachable only from the TensorRT-LLM checkpoint export path
(``trtllm/postprocess.py``), so they are tested alongside it rather than with the
backend-agnostic helpers in ``modelopt.torch.export.quant_utils``.

Test modules in this directory carry a ``trtllm_`` prefix because pytest runs without
``__init__.py`` here and derives the module name from the bare filename, so a plain
``test_quant_utils.py`` would collide with the one a directory up.
"""

import pytest
import torch

from modelopt.torch.export.trtllm.quant_utils import get_scaling_factor_from_weight


@pytest.mark.parametrize(
    ("weight", "group_size", "expected"),
    [
        (
            torch.tensor([[0.0, 0.35, 0.28, 7.0], [0.49, 0.84, -0.77, 0.07]]),
            2,
            torch.tensor([[0.05, 1.0], [0.12, 0.11]]),
        ),  # group_size != 0 and divides weight.shape[1]
        (
            torch.tensor([[0.127, 0.0, 1.27, -12.7], [0.0, 127.0, 0.254, 2.54]]),
            0,
            torch.tensor([0.1, 1.0]),
        ),  # group_size = 0
        (
            torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, -0.127, 0.254, 2.54]]),
            0,
            torch.tensor([1.0, 0.02]),
        ),  # zero replaced with 1.0
        (
            torch.tensor([[0.0, 0.84, -0.77, 0.07], [0.0, 0.0, 0.0, 0.0]]),
            2,
            torch.tensor([[0.12, 0.11], [1.0, 1.0]]),
        ),  # zero replaced with 1.0
    ],
)
def test_get_scaling_factor_from_weight(weight, group_size, expected):
    scaling_factor = get_scaling_factor_from_weight(weight, group_size)
    # Check if shapes match
    if group_size != 0:
        assert list(scaling_factor.shape) == [weight.shape[0], weight.shape[1] // group_size]
    else:
        assert list(scaling_factor.shape) == [weight.shape[0]]

    assert torch.allclose(scaling_factor, expected, rtol=0.0, atol=0.0)
