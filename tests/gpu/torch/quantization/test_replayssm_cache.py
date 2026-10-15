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


@pytest.mark.parametrize("quantized_read", [True, False])
def test_replay_cache_reordering_flush_and_reuse(quantized_read):
    torch.manual_seed(73)
    buffers = _paged_buffers(5, _gdn_replay_state_spec(1, 2, 64))
    cache = _GDNReplayCache(buffers, 1, quantized_read=quantized_read)
    pointers = [x.data_ptr() for x in buffers]
    references = {}
    rate = torch.zeros(2, device="cuda")
    bias = torch.randn(2, device="cuda") * 0.1
    mode = "tp8_hi32" if quantized_read else "s32_rs"

    def prefill(slot):
        state = torch.randn(1, 2, 64, 128, device="cuda") * 0.1
        buffers[0][slot].copy_(state[0])
        cache.reset(torch.tensor([slot], device="cuda", dtype=torch.int32))
        references[slot] = native.GDNReplay(state, 1, mode)

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
        indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
        qkv = torch.randn(len(slots), 384, device="cuda", dtype=torch.bfloat16)
        a = (torch.randn(len(slots), 2, device="cuda") * 0.1 - 3).bfloat16()
        b = torch.randn_like(a)
        output = torch.full((len(slots), 2, 64), 17.0, device="cuda")
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
                if actual is cache.read and not quantized_read:
                    continue
                torch.testing.assert_close(actual[slot], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(cache.steps[slot], reference.steps[0], rtol=0, atol=0)
    assert pointers == [x.data_ptr() for x in buffers]
    # The reserved slot and a never-used slot are untouched, despite poisoned initial contents.
    assert torch.isnan(buffers[0][[0, 4]]).all()
    assert torch.all(cache.steps[[0, 4]] == -1)
