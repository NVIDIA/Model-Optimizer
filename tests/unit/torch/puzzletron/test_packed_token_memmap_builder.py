# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for packed-token cache reuse and rebuild behavior."""

import json
import sys
from types import SimpleNamespace

import pytest

from examples.puzzletron.tools import build_packed_token_memmap


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
