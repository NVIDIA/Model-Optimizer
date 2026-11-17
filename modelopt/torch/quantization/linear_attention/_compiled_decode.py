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

"""Opt-in compilation of bounded GDN decode chunks with unchanged native forward values."""

from contextlib import contextmanager
from functools import cache

import torch

from .decode import _compile_recurrent_decode, _run_decode_chunk

__all__ = []


@contextmanager
def compiled_gdn_decode():
    """Compile GDN recurrence for this training context; retain eager execution by default."""
    token = _compile_recurrent_decode.set(True)
    try:
        yield
    finally:
        _compile_recurrent_decode.reset(token)


@cache
def compiled_chunk(device):
    # These cached buffers outlive a graph call, so allocate them outside its pool.
    from ...kernels.quantization.linear_attention.serving.forward import _decode_metadata

    compiled = torch.compile(
        _run_decode_chunk,
        fullgraph=True,
        options={"triton.cudagraphs": True, "emulate_precision_casts": True},
    )

    def run(state, q, k, v, g, beta, gates, *quantization):
        # A metadata-cache reset must not create persistent buffers inside a graph pool.
        _decode_metadata(device)
        # Bound compiler work without changing the 64-token recomputation boundary.
        outputs = []
        for start in range(0, len(q), 8):
            sl = slice(start, start + 8)
            chunk_gates = (gates[0][sl], gates[1][sl], *gates[2:]) if gates else None
            output, state = compiled(
                state, q[sl], k[sl], v[sl], g[sl], beta[sl], chunk_gates, *quantization
            )
            outputs.append(output)
        return torch.cat(outputs), state

    return run


@torch.library.custom_op("modelopt::gdn_native_decode_step", mutates_args=())
def native_gdn_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    normalize: bool,
    a: torch.Tensor,
    b: torch.Tensor,
    rate: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The optional serving backend stays opaque to Dynamo, preserving its rounding.
    from ...kernels.quantization.linear_attention.serving.forward import step

    return step(q, k, v, g, beta, state, scale, normalize, (a, b, rate, bias))


@native_gdn_step.register_fake
def _native_gdn_step_fake(q, k, v, g, beta, state, scale, normalize, a, b, rate, bias):
    # Native decode materializes both outputs, including for strided input views.
    return (
        torch.empty_like(v, memory_format=torch.contiguous_format),
        torch.empty_like(state, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op("modelopt::gdn_native_batched_step", mutates_args=())
def native_gdn_batched_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    normalize: bool,
    a: torch.Tensor,
    b: torch.Tensor,
    rate: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ...kernels.quantization.linear_attention.serving.forward import step_batched

    return step_batched(q, k, v, state, scale, normalize, (a, b, rate, bias))


@native_gdn_batched_step.register_fake
def _native_gdn_batched_step_fake(q, k, v, state, scale, normalize, a, b, rate, bias):
    return (
        torch.empty_like(v, memory_format=torch.contiguous_format),
        torch.empty_like(state, memory_format=torch.contiguous_format),
    )
