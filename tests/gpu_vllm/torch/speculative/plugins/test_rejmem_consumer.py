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

"""The memory consumer's prepare kernel against vLLM 0.33.0's own, on one synthetic DFlash step.

A memory row in the wrong KV slot or a draft seq_len off by one silently changes what the draft
attends to, so the test pins the whole difference: positions and slots of the rejected rows and
seq_lens, nothing else.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm

if vllm.__version__ != "0.33.0":
    pytest.skip("the memory consumer patches vLLM 0.33.0 only", allow_module_level=True)

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash

from modelopt.torch.speculative.plugins.rejmem import vllm_consumer

NS, BLOCK, MAX_REQS, MAX_TOKENS = 3, 4, 4, 64
NQ = NS + 1
# Per request: (first target position, target rows, rejected rows, sampled tokens).
REQUESTS = [(10, NQ, 3, 1), (20, NQ, 0, NQ), (0, 6, 0, 0)]  # memory, full accept, prefill chunk


def _run(prepare, requests, max_model_len):
    dev = torch.device("cuda")
    nr = len(requests)
    lens = [n for _, n, _, _ in requests]
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    positions = torch.tensor(
        [p + i for p, n, _, _ in requests for i in range(n)], dtype=torch.int64, device=dev
    )
    width = -(-max_model_len // BLOCK)
    block_table = 1 + torch.arange(MAX_REQS * width, dtype=torch.int32, device=dev).view(
        MAX_REQS, -1
    )
    buffers = InputBuffers(MAX_REQS, MAX_TOKENS, dev)
    out = SimpleNamespace(
        query_slots=torch.zeros(MAX_TOKENS, dtype=torch.int64, device=dev),
        ctx_positions=torch.full((MAX_TOKENS,), -7, dtype=torch.int64, device=dev),
        ctx_slots=torch.full((MAX_TOKENS,), -7, dtype=torch.int64, device=dev),
        sample=[torch.zeros(MAX_REQS * NS, dtype=torch.int64, device=dev) for _ in range(3)],
        temperature=torch.zeros(MAX_REQS, device=dev),
        seeds=torch.zeros(MAX_REQS, dtype=torch.int64, device=dev),
    )
    batch = SimpleNamespace(
        num_reqs=nr,
        num_scheduled_tokens=np.array(lens, dtype=np.int32),
        positions=positions,
        query_start_loc=torch.tensor(qsl, device=dev),
        idx_mapping=torch.arange(nr, dtype=torch.int64, device=dev),
    )
    prepare(
        buffers, out.query_slots, out.ctx_positions, out.ctx_slots, *out.sample, out.temperature,
        out.seeds, batch,
        torch.tensor([s for *_, s in requests], dtype=torch.int32, device=dev),
        torch.tensor([r for _, _, r, _ in requests], dtype=torch.int32, device=dev),
        torch.arange(100, 100 + MAX_REQS, device=dev),  # last sampled
        torch.arange(200, 200 + MAX_REQS, device=dev),  # next prefill tokens
        torch.ones(MAX_REQS, device=dev), torch.arange(MAX_REQS, device=dev),
        block_table, BLOCK, 0, 1, 1, 999, NQ, NS, MAX_REQS, MAX_TOKENS, max_model_len,
    )  # fmt: skip
    torch.cuda.synchronize()
    return buffers, out, block_table


def _slot(block_table, req, kv_pos):
    return int(block_table[req, kv_pos // BLOCK]) * BLOCK + kv_pos % BLOCK


def test_rejected_rows_become_memory_after_the_block_and_nothing_else_changes(monkeypatch):
    monkeypatch.setattr(vllm_consumer, "_dropped_rows", None)
    ref_buf, ref, _ = _run(dflash.prepare_dflash_inputs, REQUESTS, max_model_len=64)
    buf, out, block_table = _run(vllm_consumer.prepare_dflash_inputs, REQUESTS, max_model_len=64)

    for name in ("input_ids", "positions", "is_padding", "query_start_loc"):
        assert torch.equal(getattr(buf, name), getattr(ref_buf, name)), name
    assert torch.equal(out.query_slots, ref.query_slots)
    assert all(torch.equal(a, b) for a, b in zip(out.sample, ref.sample))
    num_rejected = torch.tensor([r for _, _, r, _ in REQUESTS] + [0], dtype=torch.int32).cuda()
    assert torch.equal(buf.seq_lens, ref_buf.seq_lens + num_rejected)

    row = 0
    for req, (first, n, r, _) in enumerate(REQUESTS):
        for i in range(n):
            pos = first + i
            if i < n - r:  # committed context: unchanged
                assert int(out.ctx_positions[row]) == int(ref.ctx_positions[row]) == pos
                assert (
                    int(out.ctx_slots[row])
                    == int(ref.ctx_slots[row])
                    == _slot(block_table, req, pos)
                )
            else:  # memory: real position, slot of logical index pos + 1 + NS
                assert int(ref.ctx_slots[row]) == PAD_SLOT_ID
                assert int(out.ctx_positions[row]) == pos
                assert int(out.ctx_slots[row]) == _slot(block_table, req, pos + NQ)
            row += 1
    assert torch.equal(out.ctx_positions[row:], ref.ctx_positions[row:])


def test_memory_rows_past_max_model_len_stop_the_worker(monkeypatch):
    monkeypatch.setattr(vllm_consumer, "_dropped_rows", None)
    # Request 0's memory rows 11, 12, 13 need logical indices 15, 16, 17: two are past 16.
    _, out, _ = _run(vllm_consumer.prepare_dflash_inputs, REQUESTS[:1], max_model_len=16)
    assert out.ctx_slots[1:4].tolist()[1:] == [PAD_SLOT_ID, PAD_SLOT_ID]
    assert vllm_consumer.dropped_memory_rows() == 2  # at once, before the next step raises
    with pytest.raises(RuntimeError, match="2 memory rows found no KV slot"):
        _run(vllm_consumer.prepare_dflash_inputs, REQUESTS[:1], max_model_len=16)
