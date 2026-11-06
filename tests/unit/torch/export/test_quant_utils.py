# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import pytest
import torch

from modelopt.torch.export.quant_format import QUANTIZATION_FP8, QUANTIZATION_FP8_PC_PT
from modelopt.torch.export.quant_utils import to_quantized_weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("quantization", "weight_shape", "scale_shape", "broadcast_shape"),
    [
        (QUANTIZATION_FP8, (3, 8), (), ()),
        (QUANTIZATION_FP8, (3, 8), (1,), (1,)),
        (QUANTIZATION_FP8, (2, 3, 8), (2, 1), (2, 1, 1)),
        (QUANTIZATION_FP8_PC_PT, (3, 8), (3,), (3, 1)),
        (QUANTIZATION_FP8_PC_PT, (2, 3, 8), (2,), (2, 1, 1)),
        (QUANTIZATION_FP8_PC_PT, (2, 3, 8), (2, 8), (2, 1, 8)),
        (QUANTIZATION_FP8_PC_PT, (2, 3, 8), (2, 3), (2, 3, 1)),
    ],
    ids=["scalar", "tensor", "moe-tensor", "channel", "expert", "moe-input", "moe-output"],
)
def test_to_quantized_weight_fp8_saturates(
    dtype, quantization, weight_shape, scale_shape, broadcast_shape
):
    scale = (torch.arange(1, math.prod(scale_shape) + 1, dtype=torch.float32) / 4).reshape(
        scale_shape
    )
    values = torch.tensor([-512, -448, -224, -1, 0, 224, 448, 512], dtype=torch.float32)
    weight = (values.expand(weight_shape) * scale.reshape(broadcast_shape)).to(dtype)
    original_weight, original_scale = weight.clone(), scale.clone()

    packed = to_quantized_weight(weight, scale, quantization)

    expected = torch.tensor([-448, -448, -224, -1, 0, 224, 448, 448], dtype=torch.float32)
    assert packed.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(packed.float(), expected.expand(weight_shape), rtol=0, atol=0)
    torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
