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

"""CUDA coverage for native checkpoint conversion."""

import pytest
import torch

from modelopt.torch.export.shard_cast_utils import (
    dequantize_fp8_ue8m0_to_bf16,
    quantize_bf16_to_fp8_ue8m0,
)


@pytest.mark.parametrize("device_index", [0, 1])
def test_native_fp8_roundtrip_uses_tensor_device(device_index):
    if device_index >= torch.cuda.device_count():
        pytest.skip("Needs a second CUDA device")
    weight = torch.ones(129, 131, device=f"cuda:{device_index}", dtype=torch.bfloat16).T
    with torch.cuda.device(0):
        encoded, scales = quantize_bf16_to_fp8_ue8m0(weight)
        restored = dequantize_fp8_ue8m0_to_bf16(encoded, scales.cpu())
        assert torch.cuda.current_device() == 0
    torch.testing.assert_close(restored, weight, rtol=0, atol=0)
    # Prevalidated checkpoint scales must also decode without a host synchronization.
    with torch.cuda.device(weight.device):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=weight.device)):
            captured = dequantize_fp8_ue8m0_to_bf16(encoded, scales, validate_scales=False)
        captured.zero_()
        graph.replay()
    torch.testing.assert_close(captured, weight, rtol=0, atol=0)
