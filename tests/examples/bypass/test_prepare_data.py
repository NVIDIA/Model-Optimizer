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

"""Actual immutable packing, native chat rendering and token-weighted source mixing."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from examples.bypass.prepare_data import (
    PackedBatchSequence,
    blend_conversations,
    prepare_packed_corpus,
)


@pytest.mark.parametrize("calibration_length", [None, 8])
def test_packed_corpus_has_unique_cursor_and_disjoint_conversations(
    tmp_path, monkeypatch, calibration_length
):
    def conversations():
        for index in range(256):
            item = (f"conversation-{index}", list(range(1 + index * 23, 21 + index * 23)))
            yield item
            if index % 50 == 0:
                yield item

    sizes = {"train": 32, "validation": 4, "diagnostic": 4, "calibration": 4}
    manifest = prepare_packed_corpus(
        conversations(),
        tmp_path / "data",
        sequence_length=16,
        pack_counts=sizes,
        calibration_sequence_length=calibration_length,
    )
    original_iterdir = Path.iterdir
    # Filesystem enumeration order must not change the persisted resume identity.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "iterdir", lambda path: reversed(list(original_iterdir(path))))
        repeated = prepare_packed_corpus(
            conversations(),
            tmp_path / "repeated",
            sequence_length=16,
            pack_counts=sizes,
            calibration_sequence_length=calibration_length,
        )
    assert repeated == manifest
    assert (tmp_path / "repeated/manifest.json").read_bytes() == (
        tmp_path / "data/manifest.json"
    ).read_bytes()
    with pytest.raises(ValueError):
        prepare_packed_corpus(
            iter([("too-short", [1, 2])]),
            tmp_path / "exhausted",
            sequence_length=16,
            pack_counts=sizes,
        )
    assert not (tmp_path / "exhausted").exists()
    owners, intervals = {}, {}
    for partition in sizes:
        directory = tmp_path / "data" / partition
        for row in map(json.loads, (directory / "provenance.jsonl").read_text().splitlines()):
            for fragment in row["fragments"]:
                identity = fragment["conversation_id"]
                assert owners.setdefault(identity, partition) == partition
                assert fragment["start"] == intervals.get(identity, 0)
                assert fragment["stop"] > fragment["start"]
                intervals[identity] = fragment["stop"]
        ids = torch.from_numpy(np.load(directory / "input_ids.npy").copy())
        docs = torch.from_numpy(np.load(directory / "document_ids.npy").copy())
        positions = torch.from_numpy(np.load(directory / "position_ids.npy").copy())
        labels = torch.from_numpy(np.load(directory / "labels.npy").copy())
        assert (
            ids.shape[1] == (calibration_length or 16)
            if partition == "calibration"
            else ids.shape[1] == 16
        )
        same = docs[:, 1:] == docs[:, :-1]
        torch.testing.assert_close(labels[:, :-1][same], ids[:, 1:][same])
        assert torch.all(labels[:, :-1][~same] == -100)
        starts = torch.cat((torch.ones_like(docs[:, :1], dtype=torch.bool), ~same), dim=1)
        assert torch.all(positions[starts] == 0)
        assert torch.any((docs > 0) & (labels == -100))  # MSE retains document-final tokens.
    lanes = [
        PackedBatchSequence(
            tmp_path / "data", "train", global_batch_size=8, dp_rank=rank, dp_size=2
        )
        for rank in range(2)
    ]
    seen = [int(value) for step in range(4) for lane in lanes for value in lane[step]["sample_ids"]]
    assert seen == list(range(32))
    batch = lanes[0][0]
    assert batch["_packed_seq_ids"].max() > 1
    torch.testing.assert_close(batch["attention_mask"], torch.ones_like(batch["input_ids"]))
    torch.testing.assert_close(batch["valid_mask"], batch["attention_mask"] > 0)
    with pytest.raises(IndexError):
        lanes[0][4]
    smaller = {**sizes, "train": 8}
    prepare_packed_corpus(
        conversations(), tmp_path / "prefix", sequence_length=16, pack_counts=smaller
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "prefix/train/input_ids.npy"),
        np.load(tmp_path / "data/train/input_ids.npy")[:8],
    )


def test_native_cascade_rendering_mixing_and_source_exhaustion(tmp_path):
    weights = {
        "chat": 5620,
        "math": 2110,
        "science": 1090,
        "instruction_following": 330,
        "conversational_agent": 330,
        "terminal_agent": 330,
        "swe": 180,
        "safety": 2,
    }
    vocab = {
        "[UNK]": 0,
        "<analysis>": 1,
        "<tools>": 2,
        "answer": 3,
        "reason": 4,
        **{name: index + 5 for index, name in enumerate(weights)},
    }
    engine = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    engine.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% if tools %}<tools> {{ tools | tojson }} {% endif %}"
        "{% for m in messages %}{{ m['content'] }} "
        "{% if m['reasoning_content'] is defined %}<analysis> {{ m['reasoning_content'] }} {% endif %}"
        "{% if m['tool_calls'] is defined %}{{ m['tool_calls'] | tojson }} {% endif %}{% endfor %}"
    )
    sources, identities = {}, {}
    for name in weights:
        rows = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": name + " " + str(i) + " answer " * (12 if name == "chat" else 2),
                        "reasoning_content": "reason",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"key": 1}'},
                            }
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {"type": "object"}},
                    }
                ],
            }
            for i in range(1200)
        ]
        rows.insert(0, copy.deepcopy(rows[0]))
        directory = tmp_path / name
        Dataset.from_list(rows).save_to_disk(directory)
        sources[name] = load_from_disk(directory)
        for row in rows:
            native = copy.deepcopy(row)
            native["messages"][0]["tool_calls"][0]["function"]["arguments"] = {"key": 1}
            rendered = tokenizer.apply_chat_template(
                native["messages"],
                tools=native["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            identities[hashlib.sha256(rendered.encode()).hexdigest()] = name
    tokens = list(blend_conversations(sources, tokenizer, weights=weights, max_conversations=1000))
    assert len(tokens) == len({identity for identity, _ in tokens}) == 1000
    totals = Counter()
    for identity, ids in tokens:
        totals[identities[identity]] += len(ids)
        assert 1 in ids and 2 in ids  # Native reasoning and tool markers survived.
    for name, weight in weights.items():
        assert abs(totals[name] / totals.total() - weight / sum(weights.values())) < 0.01
    assert tokens == list(
        blend_conversations(sources, tokenizer, weights=weights, max_conversations=1000)
    )
    # Exhausting a requested source must never renormalize the remaining mixture.
    small = {name: dataset.select(range(1)) for name, dataset in sources.items()}
    with pytest.raises(ValueError):
        list(blend_conversations(small, tokenizer, weights=weights, max_conversations=1000))
