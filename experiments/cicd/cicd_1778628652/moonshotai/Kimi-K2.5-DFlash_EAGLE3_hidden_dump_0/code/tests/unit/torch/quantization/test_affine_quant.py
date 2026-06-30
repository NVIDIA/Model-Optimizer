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

"""Test of affine quantization for KV cache"""

import pytest
import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer


class TestAffineBMMQuantizer:
    def get_bias_shape(self, input_shape, axis):
        """Calculate bias shape after reduction.

        Args:
            input_shape: Input tensor shape (e.g. (2, 3, 4, 5))
            axis: Axes to reduce (e.g. (-1, -3))

        Returns:
            Expected shape tuple

        Example:
            shape=(2, 3, 4, 5), axis=(-2, -4) -> (1, 3, 1, 5)
        """
        if axis is None:
            return ()

        ndim = len(input_shape)
        pos_axes = {ax if ax >= 0 else ndim + ax for ax in axis}

        return tuple(input_shape[i] if i not in pos_axes else 1 for i in range(ndim))

    @pytest.mark.parametrize("num_bits", [(2, 1), (4, 3)])
    @pytest.mark.parametrize("type", ["static", "dynamic"])
    @pytest.mark.parametrize("method", ["mean", "max_min"])
    @pytest.mark.parametrize("axis", [(-1,), (-1, -3), (-1, -2, -3)])
    def test_bias_static(self, num_bits, type, method, axis):
        # Test static kv bias quantization

        x = torch.randn(2, 3, 4, 5)

        # reduce_axis conversion: (-1, -3) -> {-1: None, -3: None}
        reduce_axis = dict.fromkeys(axis)

        quant_cfg_dict = {
            "num_bits": num_bits,
            "bias": {**reduce_axis, "type": type, "method": method},
        }

        if num_bits == (2, 1):
            quant_cfg_dict["block_sizes"] = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}

        quant_cfg = QuantizerAttributeConfig(**quant_cfg_dict)
        kv_quantizer = TensorQuantizer(quant_attribute_cfg=quant_cfg)

        enable_stats_collection(kv_quantizer)
        y_quant = kv_quantizer(x)
        finish_stats_collection(kv_quantizer)

        assert y_quant.shape == x.shape
        expected_bias_shape = self.get_bias_shape(x.shape, axis)
        if type == "static":
            assert kv_quantizer._bias_calibrator._calib_bias.shape == expected_bias_shape
            assert torch.allclose(
                kv_quantizer._bias_calibrator._calib_bias, kv_quantizer.bias_value
            )
        elif type == "static":
            assert kv_quantizer.bias.shape == expected_bias_shape
