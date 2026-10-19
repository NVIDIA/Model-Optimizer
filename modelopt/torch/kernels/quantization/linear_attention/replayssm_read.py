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

"""NVFP4 read refresh using upstream ReplaySSM's codec, independent of anchor precision.

Imported only for stochastic reads. Request seeds and logical element offsets
keep the random stream independent of cache slot placement and CUDA graph replay.
"""

import triton
import triton.language as tl
from emulation.quant_tl import nvfp4_outer_scale, quantize_nvfp4

__all__ = []


@triton.jit
def _refresh_read(
    Anchor,
    Read,
    Amax,
    Seeds,
    Positions,
    Windows,
    Indices,
    NULL: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    AMAX_STRIDE: tl.constexpr,
    SEED_STRIDE: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    TILES: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Indices + batch)
    if slot == NULL:
        return
    pos, window = tl.load(Positions + batch), tl.load(Windows + batch)
    if not ((pos == 0 and window == 0) or pos == 15):
        return
    rows = tile * TILE + tl.arange(0, TILE)
    columns = tl.arange(0, K)
    logical = (head * V + rows[:, None]) * K + columns[None, :]
    offset = slot * STATE_STRIDE + logical
    state = tl.load(Anchor + offset, rows[:, None] < V, other=0)
    blocks = tl.arange(0, triton.next_power_of_2(H * TILES))
    amax = tl.max(tl.load(Amax + slot * AMAX_STRIDE + blocks, blocks < H * TILES, other=0))
    seed = tl.load(Seeds + slot * SEED_STRIDE)
    # Initial read is refresh 0; the first completed window is refresh 1.
    refresh = window + (pos == 15).to(tl.int32)
    uniform = tl.rand(seed + refresh * 1103515245, logical)
    rounded = quantize_nvfp4(state, uniform, nvfp4_outer_scale(amax), True, K // 16, 16)
    tl.store(Read + offset, rounded, rows[:, None] < V)


def refresh_nvfp4_read(anchor, read, amax, seeds, positions, windows, indices, *, null=-1):
    """Refresh only the read copy at seed/flush; the FP32 anchor is never rounded."""
    _, heads, values, keys = anchor.shape
    tiles = triton.next_power_of_2(triton.cdiv(values, 64))
    _refresh_read[(triton.cdiv(values, 64), len(indices), heads)](
        anchor,
        read,
        amax,
        seeds,
        positions,
        windows,
        indices,
        null,
        anchor.stride(0),
        amax.stride(0),
        seeds.stride(0),
        heads,
        values,
        keys,
        64,
        tiles,
    )
