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

import pytest
import torch

from modelopt.torch.speculative.plugins.rejmem.records import (
    SCHEMA,
    RolloutRecord,
    load_record,
    save_record,
    validate_record,
)


def _record() -> RolloutRecord:
    """Prompt [10, 11]; round 0 at anchor 2 drafts [20, 21, 22] and 1 is accepted; round 1 at anchor 4
    holds the 2 rejected drafts as memory and is never verified."""
    i32 = torch.int32
    tensors = {
        "token_ids": torch.tensor([10, 11, 12, 20, 13], dtype=i32),
        "ctx_aux_hidden": torch.randn(4, 4).bfloat16(),
        "ctx_last_hidden": torch.randn(4, 2).bfloat16(),
        "round_anchor_pos": torch.tensor([2, 4]),
        "round_index": torch.tensor([0, 1]),
        "round_num_accepted": torch.tensor([1, -1], dtype=i32),
        "round_draft_token_ids": torch.tensor([[20, 21, 22], [30, 31, 32]], dtype=i32),
        "round_mem_offset": torch.tensor([0, 0, 2]),
        "mem_token_ids": torch.tensor([21, 22], dtype=i32),
        "mem_aux_hidden": torch.randn(2, 4).bfloat16(),
    }
    meta = {
        "schema": SCHEMA,
        "tag": "t",
        "req_id": "r",
        "prompt_len": 2,
        "num_spec_tokens": 3,
        "aux_layer_ids": [1, 5],
        "hidden_size": 2,
        "weight_version_start": 0,
        "weight_version_end": 0,
        "memory_consumer": False,
        "num_rounds_total": 2,
        "round_keep": "all",
    }
    return RolloutRecord(tensors=tensors, meta=meta)


def test_save_load_roundtrip(tmp_path):
    rec = _record()
    save_record(str(tmp_path / "t.safetensors"), rec)
    loaded = load_record(str(tmp_path / "t.safetensors"))
    assert loaded.meta == rec.meta
    assert all(torch.equal(loaded.tensors[k], v) for k, v in rec.tensors.items())


@pytest.mark.parametrize(
    ("name", "index", "value", "error"),
    [
        ("round_draft_token_ids", (0, 0), 99, "accepted drafts differ"),
        ("round_num_accepted", (0,), 0, "anchor != previous anchor"),
        ("mem_token_ids", (1,), 99, "memory tokens are not the rejected drafts"),
    ],
)
def test_round_invariants(name, index, value, error):
    rec = _record()
    rec.tensors[name][index] = value
    with pytest.raises(ValueError, match=error):
        validate_record(rec)
