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

"""GPU memory and VJP coverage for native IQ scale replay."""

import pytest
import torch

from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.repair_formats import _ReplayScales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for peak memory")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_replay_peak_memory_and_vjp(dtype, record_property):
    """Measure fallback replay for a 4096-by-14336 projection, excluding packing/decoding."""
    torch.manual_seed(105)
    fmt = IQ_FORMAT_REGISTRY["iq1_s"]
    elements = 4096 * 14336
    shape = (
        elements // fmt.block_size,
        fmt.block_size // fmt.int8_group_size,
        fmt.int8_group_size,
    )
    steps = torch.randn(shape[:-1], device="cuda", requires_grad=True)
    _ReplayScales.apply(
        torch.ones(1, 1, shape[-1], device="cuda", dtype=torch.int8),
        torch.ones(1, 1, device="cuda"),
        dtype,
    )
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    levels = torch.randint(-16, 16, shape, device="cuda", dtype=torch.int8)
    start.record()
    output = _ReplayScales.apply(levels, steps, dtype)
    end.record()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - baseline
    budget_bytes = elements * (1 + output.element_size()) + 8 * 262144
    record_property("replay_dtype", str(dtype))
    record_property("replay_elements", elements)
    record_property("replay_forward_ms", start.elapsed_time(end))
    record_property("replay_forward_peak_bytes", peak_bytes)
    record_property("replay_forward_budget_bytes", budget_bytes)
    assert peak_bytes <= budget_bytes

    reference_steps = steps.detach().clone().requires_grad_()
    expected = (levels.float() * reference_steps.unsqueeze(-1)).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradient = torch.randn_like(output)
    (actual_vjp,) = torch.autograd.grad(output, steps, gradient)
    (expected_vjp,) = torch.autograd.grad(expected, reference_steps, gradient)
    torch.testing.assert_close(actual_vjp, expected_vjp, rtol=0, atol=0)
