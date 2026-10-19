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

"""Offline tests for packed tokenized inputs."""

import json

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file

from modelopt.torch.utils.plugins.transformers_dataset import (
    PackedBlockDataCollator,
    load_packed_dataset,
)


def _save_shard(path, offsets=None, ids=None, mask=None):
    save_file(
        {
            "input_ids": torch.tensor(
                [11, 12, 13, 21, 22] if ids is None else ids, dtype=torch.int32
            ),
            "loss_mask": torch.tensor([0, 1, 1, 0, 1] if mask is None else mask, dtype=torch.uint8),
            "offsets": torch.tensor([0, 3, 5] if offsets is None else offsets),
            "source_dataset_ids": torch.tensor([4, 7]),
        },
        str(path),
    )


def test_falconer_shards_preserve_boundaries_masks_and_manifest_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    merged = tmp_path / "merged"
    for directory in (first, second, merged):
        directory.mkdir()
    _save_shard(first / "shard_00000.safetensors")
    _save_shard(second / "shard_00000.safetensors", ids=[31, 32, 33, 41, 42])
    (merged / "manifest.json").write_text(
        json.dumps(
            {"shards": ["../second/shard_00000.safetensors", "../first/shard_00000.safetensors"]}
        )
    )
    dataset = load_packed_dataset(merged)
    assert len(dataset) == 4
    assert dataset[0]["input_ids"].tolist() == [31, 32, 33]
    assert dataset[1]["input_ids"].tolist() == [41, 42]
    assert dataset[2]["input_ids"].tolist() == [11, 12, 13]
    assert dataset[-1]["labels"].tolist() == [-100, 22]
    assert dataset[0]["loss_mask"].tolist() == [0, 1, 1]
    with pytest.raises(IndexError):
        dataset[4]


@pytest.mark.parametrize(
    "offsets", [[0, 4, 3, 5], [1, 3, 5], [0, 3, 6], [0, 0, 5], [0.0, 3.0, 5.0]]
)
def test_falconer_shards_reject_invalid_offsets(tmp_path, offsets):
    _save_shard(tmp_path / "shard_00000.safetensors", offsets=offsets)
    with pytest.raises(ValueError, match="offsets"):
        load_packed_dataset(tmp_path)


def test_huggingface_packed_dataset_keeps_optional_columns(tmp_path):
    Dataset.from_dict({"input_ids": [[1, 2, 3]], "loss_mask": [[0, 1, 1]]}).save_to_disk(tmp_path)
    row = load_packed_dataset(tmp_path)[0]
    assert row["input_ids"].tolist() == [1, 2, 3]
    assert row["loss_mask"].tolist() == [0, 1, 1]


@pytest.mark.parametrize("shift_labels", [False, True])
def test_packed_collation_preserves_masks_padding_truncation_and_labels(shift_labels):
    rows = [
        {
            "input_ids": [11, 12, 13, 14, 15],
            "labels": [11, -100, 13, 14, 15],
            "loss_mask": [0, 1, 1, 1, 1],
        },
        {"input_ids": [21, 22], "loss_mask": [0, 1]},
    ]
    batch = PackedBlockDataCollator(train_len=4, pad_token_id=99, shift_labels=shift_labels)(rows)
    assert batch["input_ids"].tolist() == [[11, 12, 13, 14], [21, 22, 99, 99]]
    assert batch["loss_mask"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    expected = [[-100, -100, 13, 14], [-100, 22, -100, -100]]
    if shift_labels:
        expected = [[-100, 13, 14, -100], [22, -100, -100, -100]]
    assert batch["labels"].tolist() == expected


@pytest.mark.parametrize(
    ("row", "error"),
    [
        ({"input_ids": [1.0, 2.0]}, "integer"),
        ({"input_ids": [1, -2]}, "nonnegative"),
        ({"input_ids": [1, 2], "loss_mask": [0, 0.5]}, "binary"),
        ({"input_ids": [1, 2], "loss_mask": [1]}, "match"),
        ({"input_ids": [1, 2], "loss_mask": [1, 2]}, "binary"),
    ],
)
def test_packed_collation_rejects_invalid_tokens_and_masks(row, error):
    with pytest.raises(ValueError, match=error):
        PackedBlockDataCollator()([row])
