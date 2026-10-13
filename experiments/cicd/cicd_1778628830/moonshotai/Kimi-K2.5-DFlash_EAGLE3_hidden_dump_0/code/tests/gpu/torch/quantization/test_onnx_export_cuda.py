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

"""Unit tests for ONNX export for CUDA quantization."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.onnx_export import TEST_MODELS, onnx_export_tester


@pytest.mark.parametrize("model_cls", TEST_MODELS)
@pytest.mark.parametrize(
    ("num_bits", "per_channel_quantization", "constant_folding"),
    [
        (8, True, True),
        (8, False, True),
        (8, True, False),
        (8, False, False),
        ((4, 3), False, True),
        ((4, 3), False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_onnx_export_cuda(model_cls, num_bits, per_channel_quantization, constant_folding, dtype):
    # TODO: ORT output correctness tests sometimes fails due to random seed.
    # It needs to be investigated closer
    torch.manual_seed(0)

    model = model_cls()
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and num_bits == (4, 3):
            pytest.skip("Conv2d with FP8 quantization is not supported yet")
    onnx_export_tester(model, "cuda", num_bits, per_channel_quantization, constant_folding, dtype)
