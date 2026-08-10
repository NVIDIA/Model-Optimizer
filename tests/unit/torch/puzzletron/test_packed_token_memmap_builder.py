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

"""Tests for packed-token cache reuse and rebuild behavior."""

import json
import sys
from types import SimpleNamespace

import pytest

from examples.puzzletron.tools import build_packed_token_memmap


@pytest.mark.parametrize(
    "option",
    ["--num-samples", "--seq-length", "--workers", "--tokenize-batch-size"],
)
def test_cli_rejects_non_positive_numeric_inputs(tmp_path, monkeypatch, option):
    arguments = [
        "build_packed_token_memmap.py",
        "--dataset-path",
        "dataset",
        "--tokenizer-path",
        "tokenizer",
        "--output",
        str(tmp_path / "tokens.bin"),
        "--num-samples",
        "1",
        "--seq-length",
        "1",
        "--workers",
        "1",
        "--tokenize-batch-size",
        "1",
    ]
    arguments[arguments.index(option) + 1] = "0"
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as error:
        build_packed_token_memmap.main()

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("recorded_dataset", "rebuilds"),
    [("dataset-old", True), ("dataset-new", False)],
    ids=("stale-metadata", "reusable-cache"),
)
def test_cache_reuse_requires_current_complete_metadata(
    tmp_path,
    monkeypatch,
    recorded_dataset,
    rebuilds,
):
    output = tmp_path / "train.tokens"
    dataset_path = tmp_path / "dataset-new"
    tokenizer_path = tmp_path / "tokenizer"
    expected_bytes = 3 * 4
    output.write_bytes(b"\0" * expected_bytes)
    metadata_path = output.with_suffix(output.suffix + ".json")
    stale_metadata = {
        "status": "complete",
        "version": 1,
        "dataset_path": str(tmp_path / recorded_dataset),
        "tokenizer_path": str(tokenizer_path),
        "split": "train",
        "content_field": "messages",
        "num_samples": 1,
        "seq_length": 2,
        "shuffle_seed": 7,
        "dtype": "uint32",
        "bytes": expected_bytes,
        "workers": 1,
    }
    metadata_path.write_text(json.dumps(stale_metadata))

    pool_calls = []

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def imap_unordered(self, _worker, tasks):
            pool_calls.append(tuple(tasks))
            return [
                {
                    "worker": task["worker"],
                    "rows": task["stop"] - task["start"],
                    "examples": 1,
                }
                for task in tasks
            ]

    def fake_pool(*, processes):
        assert processes == 1
        return FakePool()

    monkeypatch.setattr(
        build_packed_token_memmap.mp,
        "get_context",
        lambda _method: SimpleNamespace(Pool=fake_pool),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_packed_token_memmap.py",
            "--dataset-path",
            str(dataset_path),
            "--tokenizer-path",
            str(tokenizer_path),
            "--output",
            str(output),
            "--num-samples",
            "1",
            "--seq-length",
            "2",
            "--workers",
            "1",
            "--shuffle-seed",
            "7",
        ],
    )

    build_packed_token_memmap.main()

    assert bool(pool_calls) is rebuilds
    rebuilt_metadata = json.loads(metadata_path.read_text())
    assert rebuilt_metadata == {
        **stale_metadata,
        "dataset_path": str(dataset_path.resolve()),
    }
