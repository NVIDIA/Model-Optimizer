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

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.tensor_quantizer_common import (
    BlockQuantTester,
    SequentialQuantizerTester,
    TensorQuantizerTester,
)

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import GroupedQuantizer, SequentialQuantizer, TensorQuantizer


class TestTensorQuantizerCPU(TensorQuantizerTester):
    device = "cpu"


class TestBlockQuantCPU(BlockQuantTester):
    device = "cpu"


class TestSequentialQuantizerCPU(SequentialQuantizerTester):
    device = "cpu"


def test_grouped_quantizer_forward_uses_representative_quantizer():
    """Single-weight compatibility paths should dispatch to the first group."""
    representative = TensorQuantizer(
        quant_attribute_cfg=QuantizerAttributeConfig(num_bits=8), amax=1.0
    )
    other = TensorQuantizer()
    other.disable()
    grouped = GroupedQuantizer(representative, other)
    inputs = torch.tensor([0.1234, -0.5678])

    assert torch.equal(grouped(inputs), representative(inputs))
    assert not torch.equal(grouped(inputs), other(inputs))


@pytest.mark.parametrize(
    ("container_cls", "pytorch_container_cls", "expected_disable_result"),
    [
        (SequentialQuantizer, nn.Sequential, None),
        (GroupedQuantizer, nn.ModuleList, [None, None]),
    ],
)
def test_quantizer_container_base_delegates_shared_contract(
    container_cls, pytorch_container_cls, expected_disable_result
):
    """Shared container mechanics should preserve each public container's behavior."""
    quantizers = (
        TensorQuantizer(quant_attribute_cfg=QuantizerAttributeConfig(num_bits=8), amax=1.0),
        TensorQuantizer(quant_attribute_cfg=QuantizerAttributeConfig(num_bits=8), amax=2.0),
    )
    container = container_cls(*quantizers)

    assert isinstance(container, pytorch_container_cls)
    assert torch.equal(container.amax, quantizers[0].amax)
    assert list(container.state_dict()) == ["0._amax", "1._amax"]

    container.amax = 0.5
    assert all(torch.equal(quantizer.amax, torch.tensor(0.5)) for quantizer in quantizers)

    disable_result = container.disable()
    assert disable_result == expected_disable_result
    assert not any(quantizer.is_enabled for quantizer in quantizers)

    container.enable()
    assert all(quantizer.is_enabled for quantizer in quantizers)

    rotate_quantizers = (
        TensorQuantizer(quant_attribute_cfg=QuantizerAttributeConfig(rotate={"enable": True})),
        TensorQuantizer(
            quant_attribute_cfg=QuantizerAttributeConfig(
                rotate={"enable": True, "mode": "rotate_back"}
            )
        ),
    )
    rotate_container = container_cls(*rotate_quantizers)
    assert rotate_container.disable_rotate() == expected_disable_result
    assert not any(quantizer.rotate_is_enabled for quantizer in rotate_quantizers)


@pytest.mark.parametrize("container_cls", [SequentialQuantizer, GroupedQuantizer])
def test_quantizer_container_base_sets_attribute_config(container_cls):
    """Scalar configs broadcast and list configs apply one-to-one for both containers."""
    container = container_cls(TensorQuantizer(), TensorQuantizer())

    container.set_from_attribute_config({"num_bits": 4})
    assert [quantizer.num_bits for quantizer in container] == [4, 4]

    container.set_from_attribute_config([{"num_bits": 8}, {"enable": False}])
    assert container[0].num_bits == 8
    assert container[1].is_enabled is False


def test_grouped_quantizer_preserves_nested_sequential_state_dict_layout():
    """Grouped quantizers may hold nested SequentialQuantizer members."""
    grouped = GroupedQuantizer(
        SequentialQuantizer(TensorQuantizer(amax=1.0), TensorQuantizer(amax=2.0)),
        TensorQuantizer(amax=3.0),
    )

    assert list(grouped.state_dict()) == ["0.0._amax", "0.1._amax", "1._amax"]
