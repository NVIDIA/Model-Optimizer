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

"""Tests of tensor quantizer."""

import torch
from _test_utils.torch.quantization.tensor_quantizer_common import (
    BlockQuantTester,
    SequentialQuantizerTester,
    TensorQuantizerTester,
)

from modelopt.torch.quantization.nn import GroupedQuantizer, TensorQuantizer


class TestTensorQuantizerCPU(TensorQuantizerTester):
    device = "cpu"


class TestBlockQuantCPU(BlockQuantTester):
    device = "cpu"


class TestSequentialQuantizerCPU(SequentialQuantizerTester):
    device = "cpu"


def test_grouped_quantizer_forward_uses_representative_quantizer():
    """Single-weight compatibility paths should dispatch to the first group."""
    representative = TensorQuantizer()
    other = TensorQuantizer()
    other.disable()
    grouped = GroupedQuantizer(representative, other)
    inputs = torch.tensor([0.1234, -0.5678])

    assert torch.equal(grouped(inputs), representative(inputs))
    assert not torch.equal(grouped(inputs), other(inputs))
