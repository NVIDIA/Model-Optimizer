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

"""Chunk-64 FP32 Neumann inverse with actual-polynomial first-order backward."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

__all__ = ["neumann_inverse"]


@triton.jit
def _neumann_fwd(Lower, Output, DEGREE: tl.constexpr):
    index = tl.arange(0, 64)
    offsets = tl.program_id(0) * 4096 + index[:, None] * 64 + index[None, :]
    base = -tl.load(Lower + offsets)
    series = (index[:, None] == index[None, :]).to(tl.float32)
    power = base
    COUNT: tl.constexpr = (DEGREE + 1).bit_length() - 1
    for shift in tl.static_range(COUNT - 1, -1, -1):
        series = series + tl.dot(power, series, input_precision="ieee")
        if ((DEGREE + 1) >> shift) & 1 or shift > 0:
            power = tl.dot(power, power, input_precision="ieee")
        if ((DEGREE + 1) >> shift) & 1:
            series = series + power
            if shift > 0:
                power = tl.dot(power, base, input_precision="ieee")
    tl.store(Output + offsets, series)


class _NeumannInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, degree):
        ctx.save_for_backward(lower)
        ctx.degree = degree
        output = torch.empty_like(lower)
        _neumann_fwd[(lower.numel() // 4096,)](lower, output, degree, num_warps=4)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        from modelopt.torch.quantization.linear_attention.solve import _neumann_binary

        (lower,) = ctx.saved_tensors
        # Reconstruct the same polynomial, not the derivative of an exact inverse.
        with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
            value = lower.detach().requires_grad_()
            result = _neumann_binary(value, ctx.degree)
            gradient = torch.autograd.grad(result, value, grad_output)[0]
        return gradient, None


def neumann_inverse(lower: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate the configured polynomial without an implicit backend fallback."""
    if not lower.is_cuda or lower.dtype != torch.float32 or lower.shape[-2:] != (64, 64):
        raise ValueError("Triton Neumann requires CUDA FP32 matrices of shape [...,64,64]")
    if type(degree) is not int or not 0 <= degree <= 63:
        raise ValueError("degree must be an integer in [0,63]")
    return _NeumannInverse.apply(lower.contiguous(), degree)
