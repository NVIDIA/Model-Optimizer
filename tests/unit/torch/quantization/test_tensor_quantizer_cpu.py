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

import modelopt.torch.quantization.nn.modules.tensor_quantizer as tensor_quantizer_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import GroupedQuantizer, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils.numeric_utils import E4M3_MAX, E4M3_MAX_46


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


@pytest.mark.parametrize(
    ("four_over_six", "expected_fp8_max"), [(False, E4M3_MAX), (True, E4M3_MAX_46)]
)
def test_unpromoted_nvfp4_static_quantizer_fake_quantizes(
    monkeypatch, four_over_six, expected_fp8_max
):
    """A plain TensorQuantizer with static NVFP4 attrs must fake-quantize via the
    static blockwise FP4 path, not the per-tensor FP8 (E4M3-only) path.

    Regression test for the training-forward crash hit when an NVFP4 weight
    quantizer was never promoted to StaticBlockScaleQuantizer (promotion happens
    in calibration finalize, which some training flows skip). The real
    implementation launches a triton kernel and cannot run on CPU, so it is
    replaced with a passthrough that records how it was called.
    """
    block_sizes = {-1: 16, "type": "static", "scale_bits": (4, 3)}
    if four_over_six:
        block_sizes["four_over_six"] = True
    quantizer = TensorQuantizer(
        quant_attribute_cfg=QuantizerAttributeConfig(num_bits=(2, 1), block_sizes=block_sizes)
    )
    assert type(quantizer) is TensorQuantizer
    assert quantizer.is_nvfp4_static

    weight = torch.randn(4, 64)
    per_block_amax = weight.view(4, 4, 16).abs().amax(dim=-1)
    quantizer.amax = per_block_amax.clone()

    captured = {}

    def spy(*args, **kwargs):
        # Call site: (inputs, amax, global_amax, quantize_block_scales,
        #             fp8_max_for_normalization, dtype, pass_through_bwd).
        captured["args"] = args
        return args[0]

    monkeypatch.setattr(tensor_quantizer_module, "static_blockwise_fp4_fake_quant", spy)

    output = quantizer(weight)

    assert output.shape == weight.shape
    args = captured["args"]
    assert torch.equal(args[1], per_block_amax)
    # global_amax=None lets the implementation derive the global scale from the
    # per-block amax, matching the fallback used when promoting the quantizer.
    assert args[2] is None
    assert args[3] is True
    assert args[4] == expected_fp8_max
    assert args[5] == weight.dtype
