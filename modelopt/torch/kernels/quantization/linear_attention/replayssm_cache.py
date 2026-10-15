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

"""Indexed two-precision GDN decode using externally owned cache pages.

The engine must bind these buffers before decode and reset the counter after
prefill. No request state is allocated or retained outside the bound pages.
"""

from math import prod

import torch
import triton

__all__ = []


def _gdn_replay_state_spec(key_heads, value_heads, value_dim):
    """Return per-slot shapes/dtypes, starting with the existing FP32 GDN anchor."""
    if key_heads < 1 or value_heads < 1 or value_heads % key_heads or value_dim < 1:
        raise ValueError("GDN replay requires positive dimensions and grouped value heads")
    return (
        ((value_heads, value_dim, 128), torch.float32),  # recurrence checkpoint
        ((value_heads, value_dim, 128), torch.float32),  # FP8 read values in FP32 storage
        ((key_heads, 16, 128), torch.bfloat16),
        ((value_heads, 16, value_dim), torch.bfloat16),
        ((value_heads, 16), torch.float32),  # raw gate
        ((value_heads, 16), torch.float32),  # beta
        ((value_heads, 16, value_dim), torch.float16),  # read-side residuals
        ((4,), torch.int32),  # decode count + padding to preserve 16-byte page alignment
    )


class _GDNReplayCache:
    """Bind engine-owned buffers; slot order is independent of the current batch.

    Cache rows may have a padded stride, as in vLLM's Mamba pages. Inside a row,
    each tensor must be dense. The caller supplies unique valid slot indices;
    slot zero is reserved for inactive rows. Decode outputs for inactive rows
    are left untouched. A fresh prefill must write the anchor and call reset.
    """

    def __init__(self, buffers, key_heads, *, quantized_read=True):
        (
            self.anchor,
            self.read,
            self.keys,
            self.values,
            self.gates,
            self.betas,
            self.updates,
            steps,
        ) = buffers
        heads, values, keys = self.anchor.shape[1:]
        if keys != 128 or heads < 1:
            raise ValueError("Two-precision GDN serving requires K=128 and positive heads")
        spec = _gdn_replay_state_spec(key_heads, heads, values)
        for tensor, (shape, dtype) in zip(buffers, spec):
            if (
                tensor.shape != (len(self.anchor), *shape)
                or tensor.dtype != dtype
                or tensor.device != self.anchor.device
                or not tensor.is_cuda
                or tensor.stride(0) < prod(shape)
                or any(tensor.stride(i + 1) != prod(shape[i + 1 :]) for i in range(len(shape)))
            ):
                raise ValueError("Invalid ReplaySSM cache shape, dtype, device or row layout")
        if self.anchor.stride(0) != self.read.stride(0):
            raise ValueError("Recurrence and read checkpoints must have the same slot stride")
        self.steps = steps[:, 0]
        self.key_heads = key_heads
        self.quantized_read = quantized_read

    @torch.no_grad()
    def reset(self, indices):
        """Begin decode from newly written prefill anchors, including recycled slots.

        Old ring entries are masked by the window position. The first decode
        writes the read checkpoint, so clearing the large buffers is unnecessary.
        """
        self.steps.index_fill_(0, indices.long(), 0)

    @torch.no_grad()
    def step(self, qkv, a, b, rate, bias, indices, output):
        """Run one native token per active slot, then advance only those counters.

        QKV and raw gate projections must be BF16. Output is [batch,Hv,V], in
        FP32 or BF16. This inference-only entry point does not build an adjoint.
        """
        # Optional standalone ReplaySSM dependency, pinned by the example.
        from emulation.gdn_replay_unf_vf import _gdn_unf_vf_step_kernel
        from emulation.modes import MODES

        rows, heads, values = output.shape
        if (
            qkv.shape != (rows, 2 * self.key_heads * 128 + heads * values)
            or a.shape != (rows, heads)
            or b.shape != a.shape
            or rate.shape != (heads,)
            or bias.shape != rate.shape
            or output.shape[1:] != self.anchor.shape[1:3]
            or indices.shape != (rows,)
            or indices.dtype not in (torch.int32, torch.int64)
            or output.dtype not in (torch.float32, torch.bfloat16)
            or any(x.dtype != torch.bfloat16 for x in (qkv, a, b))
            or any(x.dtype != torch.float32 for x in (rate, bias))
            or any(x.device != self.anchor.device for x in (qkv, a, b, rate, bias, indices, output))
            or any(not x.is_contiguous() for x in (qkv, a, b, rate, bias, indices, output))
        ):
            raise ValueError("Invalid packed GDN replay inputs; QKV and raw gates must be BF16")
        if not rows:
            return
        steps = self.steps.index_select(0, indices)
        positions = steps.remainder(16)
        windows = torch.div(steps, 16, rounding_mode="floor")
        precision = MODES["tp8_hi32" if self.quantized_read else "s32_rs"]
        # The native FP8 path does not access amax; use the anchor as a dummy pointer.
        _gdn_unf_vf_step_kernel[(triton.cdiv(values, 64), rows, heads)](
            self.anchor,
            self.read,
            self.anchor,
            qkv,
            a,
            b,
            rate,
            bias,
            output,
            self.keys,
            self.values,
            self.gates,
            self.betas,
            self.updates,
            positions,
            positions == 15,
            windows,
            indices,
            0,
            1234,
            128**-0.5,
            qkv.stride(0),
            a.stride(0),
            b.stride(0),
            output.stride(0),
            self.anchor.stride(0),
            self.keys.stride(0),
            self.values.stride(0),
            self.gates.stride(0),
            self.betas.stride(0),
            self.updates.stride(0),
            0,
            rows,
            self.key_heads,
            heads,
            128,
            values,
            64,
            triton.next_power_of_2(triton.cdiv(values, 64)),
            precision.two_precision,
            precision.state,
            precision.stochastic,
            precision.read if precision.two_precision else precision.state,
            False,
        )
        self.steps.index_add_(0, indices, (indices != 0).to(torch.int32))
