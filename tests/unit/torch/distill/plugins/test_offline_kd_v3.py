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

import base64
import json
from types import SimpleNamespace

import pytest
import torch

pa = pytest.importorskip("pyarrow")

from modelopt.torch.distill.plugins.offline_kd_v3 import (
    INPUT_COLUMNS,
    META_KEY,
    corpus_settings,
    publish_chunk,
    strip_row,
    topk_targets,
)


class _Tokenizer:
    def get_vocab(self):
        return {"a": 0, "b": 1}

    def __len__(self):
        return 2


def test_strip_row_shifts_labels_and_drops_unscored_tail():
    ids = torch.tensor([5, 6, 7, 8, 9, 10])
    sample = strip_row(ids, torch.tensor([-100, -100, 7, 8, -100, -100]))
    assert sample["tokens"].tolist() == [5, 6, 7]
    assert sample["labels"].tolist() == [6, 7, 8]
    assert sample["loss_mask"].tolist() == [False, True, True]
    assert sample["segments"] == [3]
    assert sample["positions"].tolist() == [1, 2]
    assert strip_row(ids, torch.full((6,), -100))["segments"] == []


def test_publish_chunk_round_trip(tmp_path):
    vocab, k = 70000, 4  # Above 2**16, so indices need the high byte.
    rows = [
        (torch.tensor([5, 6, 7, 8]), torch.tensor([-100, 6, 69000, -100])),
        (torch.tensor([1, 2, 3]), torch.full((3,), -100)),  # No loss: kept as an empty sample.
        (torch.tensor([4, 69000, 11]), torch.tensor([-100, 69000, 11])),
    ]
    torch.manual_seed(0)
    samples = []
    for ids, labels in rows:
        sample = strip_row(ids, labels)
        logits = torch.randn(len(sample["positions"]), vocab + 1)
        logits[:, vocab] = 100.0  # Output-layer padding is excluded from the distribution.
        values, indices, tail = topk_targets(logits, k, vocab)
        assert (indices < vocab).all()
        assert torch.allclose(values.exp().sum(-1) + tail.exp(), torch.ones(len(tail)))
        samples.append({**sample, "values": values.half(), "indices": indices, "tail": tail})
    config = SimpleNamespace(rope_parameters={"rope_type": "dynamic", "factor": 2.0})
    settings = corpus_settings(_Tokenizer(), config, 3, recorded={}, provenance={})
    assert settings["contract"]["prefix_stable"] is False

    chunk = tmp_path / "chunk-000000000000-000000000003"
    written = publish_chunk(
        str(tmp_path), samples, settings, start=0, vocab_size=vocab, block_bytes=1
    )
    assert written == str(chunk)
    index = json.loads((chunk / "index.json").read_text())
    assert index["settings"] == settings
    # Megatron-LM's planner rejects overlapping chunks through these fields.
    assert (index["producer"]["start_iteration"], index["producer"]["end_iteration"]) == (0, 3)
    data = (chunk / index["parts"][0]["path"]).read_bytes()
    assert index["parts"][0]["size"] == len(data)
    assert json.loads(pa.ipc.open_file(data).schema.metadata[META_KEY]) == settings
    schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(index["schemas"][0])))
    blocks = index["parts"][0]["blocks"]
    assert [(b["rows"], b["supervised"]) for b in blocks] == [(1, 2), (1, 0), (1, 2)]
    for block, sample in zip(blocks, samples):
        raw = data[block["offset"] : block["offset"] + block["length"]]
        row = pa.ipc.read_record_batch(pa.ipc.read_message(raw), schema).to_pylist()[0]
        for key in ("tokens", "labels", "position_ids", "loss_mask"):
            assert row[key] == sample[key].tolist()
        assert row["segments"] == sample["segments"]
        values = torch.tensor(row["values"], dtype=torch.float16).reshape(-1, k)
        assert torch.equal(values, sample["values"])
        low, high = (
            torch.tensor(row[key], dtype=torch.int64) for key in ("indices_low", "indices_high")
        )
        assert torch.equal((low | high << 16).reshape(-1, k), sample["indices"])
        assert torch.equal(torch.tensor(row["tail"]), sample["tail"])
        # Input-only reads take the block's input_length prefix; unread target bytes stay zero.
        stream = schema.serialize().to_pybytes() + raw[: block["input_length"]]
        stream += bytes(block["length"] - block["input_length"])
        options = pa.ipc.IpcReadOptions(included_fields=list(range(len(INPUT_COLUMNS))))
        inputs = pa.ipc.open_stream(stream, options=options).read_next_batch().to_pylist()[0]
        assert inputs == {key: row[key] for key in INPUT_COLUMNS}
