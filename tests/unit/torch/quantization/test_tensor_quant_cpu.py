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

"""Tests of tensor quantization function and module."""

import numpy as np
import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear
from _test_utils.torch.quantization.tensor_quant_common import FakeTensorQuantTester

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer


class TestFakeTensorQuantCPU(FakeTensorQuantTester):
    device = "cpu"


class TestQuantizerAttributeConfig:
    def test_scaled_mode(self):
        num_bits = np.random.randint(1, 16)

        test_quant_attr_cfg = QuantizerAttributeConfig(num_bits=num_bits)
        assert test_quant_attr_cfg.num_bits == num_bits
        assert test_quant_attr_cfg.axis is None

        axis = (0, 1, 3)
        test_quant_attr_cfg = QuantizerAttributeConfig(axis=axis)
        assert test_quant_attr_cfg.num_bits == 8  # default value
        assert test_quant_attr_cfg.axis == axis

    def test_from_to_dict(self, verbose):
        quant_attr_cfg_1 = QuantizerAttributeConfig(
            num_bits=2,
            fake_quant=True,
            axis=(1, 2),
        )
        quant_attr_cfg_2 = QuantizerAttributeConfig(**quant_attr_cfg_1.dict())
        assert quant_attr_cfg_1 == quant_attr_cfg_2

        quant_attr_cfg_1 = QuantizerAttributeConfig(num_bits=2, unsigned=True)
        quant_attr_cfg_2 = QuantizerAttributeConfig(**quant_attr_cfg_1.dict())
        assert quant_attr_cfg_1 == quant_attr_cfg_2

    def test_num_bits(self):
        """Test num_bits for both integer and tuple cases."""

        with pytest.raises(
            ValueError,
            match="Invalid quantizer config: Cannot specify only {'enable': True}. "
            "Additional parameters are required when enabling quantization.",
        ):
            QuantizerAttributeConfig(enable=True)

        with pytest.raises(
            ValueError, match="num_bits must be a positive integer or a tuple of positive integers."
        ):
            QuantizerAttributeConfig(enable=True, num_bits=0)

        with pytest.raises(
            ValueError, match="num_bits must be a positive integer or a tuple of positive integers."
        ):
            QuantizerAttributeConfig(enable=True, num_bits=-1)

        # # Test positive tuple validation
        with pytest.raises(
            ValueError, match="num_bits must be a positive integer or a tuple of positive integers."
        ):
            QuantizerAttributeConfig(enable=True, num_bits=(0, 3))

        with pytest.raises(
            ValueError, match="num_bits must be a positive integer or a tuple of positive integers."
        ):
            QuantizerAttributeConfig(enable=True, num_bits=(-1, 2))


WINT4INT8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": 8, "axis": 0, "enable": True},
        ],
        "*input_quantizer": {"num_bits": 8, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "awq_full",
}


def test_set_quantizer_cxt():
    model = SimpleLinear()
    model.eval()
    inputs = model.get_input()
    mtq.quantize(model, WINT4INT8_CFG, lambda model: model(inputs))
    state_dict = model.state_dict()
    output_ref = model(inputs)

    mtq.set_quantizer_by_cfg(model, {"*output_quantizer": {"enable": True}})

    with mtq.set_quantizer_by_cfg_context(
        model, {"*": {"enable": False}, "*output_quantizer": {"enable": True}}
    ):
        for name, module in model.named_modules():
            if not isinstance(module, TensorQuantizer):
                continue
            if name.endswith("output_quantizer"):
                assert module.is_enabled
            else:
                assert not module.is_enabled
        mtq.calibrate(model, "max", lambda model: model(inputs * 10))

    mtq.set_quantizer_by_cfg(model, {"*output_quantizer": {"enable": False}})

    output_test = model(inputs)
    assert torch.allclose(output_ref, output_test)

    state_dict_test = model.state_dict()
    for k, v in state_dict_test.items():
        if "output_quantizer" in k:
            continue
        assert torch.allclose(v, state_dict[k])
