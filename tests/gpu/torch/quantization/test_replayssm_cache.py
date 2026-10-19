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

"""ReplaySSM request slots must retain the training backend's native arithmetic."""

from math import prod

import pytest
import torch

from modelopt.torch.kernels.quantization.linear_attention.replayssm_cache import (
    _gdn_replay_state_spec,
    _GDNReplayCache,
)

native = pytest.importorskip("emulation", reason="Requires pinned upstream ReplaySSM emulation")


def _paged_buffers(slots, spec):
    sizes = [prod(shape) * dtype.itemsize for shape, dtype in spec]
    assert sum(sizes) % 16 == 0
    # Like vLLM Mamba pages: all states share a row stride larger than any one tensor.
    stride = ((sum(sizes) + 255) // 256) * 256
    pages = torch.full((slots, stride), 255, dtype=torch.uint8, device="cuda")
    buffers, offset = [], 0
    for (shape, dtype), size in zip(spec, sizes):
        buffers.append(pages[:, offset : offset + size].view(dtype).view(slots, *shape))
        offset += size
    return buffers


@pytest.mark.parametrize(
    ("read_mode", "rounding"),
    [
        ("tp8_hi32", "nearest"),
        ("tp4_hi32", "nearest"),
        ("s32_rs", "nearest"),
        ("tp4_hi32", "stochastic"),
    ],
)
@pytest.mark.parametrize("captured", [False, True])
def test_replay_cache_reordering_flush_and_reuse(read_mode, rounding, captured):
    torch.manual_seed(73)
    buffers = _paged_buffers(5, _gdn_replay_state_spec(1, 2, 144, read_mode=read_mode))
    cache = _GDNReplayCache(buffers, 1, read_mode=read_mode, read_rounding=rounding)
    pointers = [x.data_ptr() for x in buffers]
    references = {}
    rate = torch.zeros(2, device="cuda")
    bias = torch.randn(2, device="cuda") * 0.1

    def prefill(slot):
        state = torch.randn(1, 2, 144, 128, device="cuda") * 0.1
        buffers[0][slot].copy_(state[0])
        cache.reset(torch.tensor([slot], device="cuda", dtype=torch.int32))
        if rounding == "stochastic":
            from modelopt.torch.quantization.linear_attention._two_precision import _native_replay

            references[slot] = _native_replay(
                state[0].transpose(-1, -2), read_mode, rounding, key_heads=1
            )
            references[slot].read_seeds.copy_(cache.read_seeds[slot : slot + 1])
        else:
            references[slot] = native.GDNReplay(state, 1, read_mode)

    if captured:
        # Fixed addresses, changing request slots and GPU counters on every replay.
        static_qkv = torch.zeros(4, 544, device="cuda", dtype=torch.bfloat16)
        static_a = torch.zeros(4, 2, device="cuda", dtype=torch.bfloat16)
        static_b = torch.zeros_like(static_a)
        static_indices = torch.zeros(4, device="cuda", dtype=torch.int32)
        static_output = torch.empty(4, 2, 144, device="cuda")
        for _ in range(3):
            cache.step(static_qkv, static_a, static_b, rate, bias, static_indices, static_output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cache.step(static_qkv, static_a, static_b, rate, bias, static_indices, static_output)
    prefill(1)
    prefill(2)
    for iteration in range(40):
        if iteration == 8:
            prefill(3)
        if iteration == 17:
            # Replace a request with an unfinished window, leaving dirty rings/read values.
            prefill(2)
        slots = [1, 2] if iteration % 2 else [2, 1]
        if iteration % 3 == 0:
            slots.remove(2)
        if iteration >= 8:
            slots.insert(0, 3)
        if iteration % 4 == 0:
            slots.insert(1, 0)
        if captured:
            slots += [0] * (4 - len(slots))
        indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
        qkv = torch.randn(len(slots), 544, device="cuda", dtype=torch.bfloat16)
        a = (torch.randn(len(slots), 2, device="cuda") * 0.1 - 3).bfloat16()
        b = torch.randn_like(a)
        output = torch.full((len(slots), 2, 144), 17.0, device="cuda")
        if captured:
            for target, source in zip(
                (static_qkv, static_a, static_b, static_indices, static_output),
                (qkv, a, b, indices, output),
            ):
                target.copy_(source)
            graph.replay()
            output = static_output
        else:
            cache.step(qkv, a, b, rate, bias, indices, output)
        for row, slot in enumerate(slots):
            if slot == 0:
                assert torch.all(output[row] == 17)
                continue
            reference = references[slot]
            expected = reference.step(
                qkv[row : row + 1], a[row : row + 1].float(), b[row : row + 1].float(), rate, bias
            )
            torch.testing.assert_close(output[row], expected[0], rtol=0, atol=0)
            for actual, expected in (
                (cache.anchor, reference.state_hi),
                (cache.read, reference.state_lo),
            ):
                # With FP32 reads the low checkpoint is unused and intentionally uninitialized.
                if actual is cache.read and read_mode == "s32_rs":
                    continue
                torch.testing.assert_close(actual[slot], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(cache.steps[slot], reference.steps[0], rtol=0, atol=0)
    assert pointers == [x.data_ptr() for x in buffers]
    # The reserved slot and a never-used slot are untouched, despite poisoned initial contents.
    assert torch.isnan(buffers[0][[0, 4]]).all()
    assert torch.all(cache.steps[[0, 4]] == -1)


def test_stochastic_read_grid():
    from modelopt.torch.kernels.quantization.linear_attention.replayssm_read import (
        refresh_nvfp4_read,
    )

    anchor = torch.full((1, 1, 128, 128), 1.125, device="cuda")
    anchor[..., 1::2] = -1.125
    anchor[..., 15::16] = 6  # Exact global and E4M3 scales; adjacent codes are +/-1 and +/-1.5.
    read = torch.empty_like(anchor)
    amax = torch.full((1, 1, 2), 6.0, device="cuda")
    seed = torch.tensor([17], device="cuda", dtype=torch.int32)
    zero = torch.zeros_like(seed)
    refresh_nvfp4_read(anchor, read, amax, seed, zero, zero, zero)
    actual = read[..., :15].reshape(-1)
    expected = anchor[..., :15].reshape(-1)
    assert ((actual.abs() == 1) | (actual.abs() == 1.5)).all()
    assert (actual[expected > 0].mean() - 1.125).abs() < 0.02
    assert (actual[expected < 0].mean() + 1.125).abs() < 0.02
    assert torch.all(read[..., 15::16] == 6)
