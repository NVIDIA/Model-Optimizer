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

"""CUDA-only latent initialization formats and initialization memory/time."""

import copy

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize(
    "config",
    [
        mtq.NVFP4_DEFAULT_CFG,
        mtq.MXFP4_DEFAULT_CFG,
        mtq.MXFP6_DEFAULT_CFG,
        mtq.MXFP8_DEFAULT_CFG,
        mtq.W4A8_NVFP4_FP8_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    ],
)
def test_initialize_cuda_weight_formats(config):
    model = nn.Linear(256, 256, bias=False).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    mtq.quantize(model, copy.deepcopy(config), lambda m: m(inputs))
    weight = model.weight
    expected = model.weight_quantizer(weight).detach().clone()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    mtq.initialize_qdq_weights(model)
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    assert model.weight is weight and model.weight_quantizer.is_enabled
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    model(inputs).float().square().mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


def test_initialize_weight_memory_and_time(record_property, capsys):
    model = nn.Sequential(*(nn.Linear(4096, 4096, bias=False) for _ in range(4)))
    model = model.to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(inputs))
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    mtq.initialize_qdq_weights(model)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    extra_peak_bytes = torch.cuda.max_memory_allocated() - baseline
    record_property("initialization_ms", elapsed_ms)
    record_property("initialization_extra_peak_bytes", extra_peak_bytes)
    with capsys.disabled():
        print(
            f"QDQ initialization: device={torch.cuda.get_device_name()}, "
            f"format=NVFP4, dtype=BF16, weights=4x4096x4096, "
            f"elapsed_ms={elapsed_ms:.3f}, extra_peak_allocated_bytes={extra_peak_bytes}"
        )
