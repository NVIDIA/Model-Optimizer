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

"""The blend cache holds the same rows, in the same order, however many ranks build it."""

import sys
import threading
from pathlib import Path

import datasets
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "llm_qat"))

import dataset_utils
from dataset_utils import (
    BlendConfig,
    DatasetSourceConfig,
    ParallelConfig,
    _build_cache_path,
    _merge_distributed_shards,
)

# Rows of three sources; the last holds fewer rows than the largest rank count, so some ranks load none.
_SOURCES = [list(range(10)), list(range(100, 107)), list(range(200, 203))]


def _loaded(rows, rank, world_size):
    """The consecutive slice of ``rows`` a rank loads, the last rank taking the remainder."""
    per_rank = len(rows) // world_size
    offset = rank * per_rank
    return rows[offset:] if rank == world_size - 1 else rows[offset : offset + per_rank]


def _tokenized(rows):
    if not rows:
        return datasets.Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
    return datasets.Dataset.from_dict(
        {
            "input_ids": [[r] for r in rows],
            "attention_mask": [[1]] * len(rows),
            "labels": [[r] for r in rows],
        }
    )


def _blend(tmp_path, monkeypatch, world_size):
    """What rank 0 caches when ``world_size`` ranks, run as threads, each load their slices."""
    barrier = threading.Barrier(world_size, timeout=120)
    monkeypatch.setattr(dataset_utils, "dist_barrier", barrier.wait)
    splits = {"train": 0.6, "eval": 0.2, "test": 0.2}
    results, errors = {}, []

    def rank_job(rank):
        parts = [_tokenized(_loaded(rows, rank, world_size)) for rows in _SOURCES]
        parallel = ParallelConfig(num_proc=1, rank=rank, world_size=world_size)
        try:
            results[rank] = _merge_distributed_shards(
                str(tmp_path), parts, parallel, splits, seed=7
            )
        except BaseException as error:  # a rank that stops would leave the others at the barrier
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=rank_job, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    return {name: [row[0] for row in results[0][name]["input_ids"]] for name in splits}


def test_the_blend_is_the_same_at_any_number_of_ranks(tmp_path, monkeypatch):
    """Each rank holds a consecutive slice of every source, so merging source by source gives the
    sources' own order whatever the rank count; merging rank by rank would not."""
    blends = {w: _blend(tmp_path / f"ranks{w}", monkeypatch, w) for w in (1, 2, 4)}

    assert blends[1] == blends[2] == blends[4]
    assert sorted(row for rows in blends[1].values() for row in rows) == sorted(
        row for rows in _SOURCES for row in rows
    )


class _Tokenizer:
    vocab_size = 100
    eos_token_id = 2
    pad_token_id = 0


@pytest.mark.parametrize(
    "change",
    [{"seed": 7}, {"shuffle": False}, {"shuffle_buffer": 5}],
    ids=["seed", "shuffle", "shuffle-buffer"],
)
def test_the_cache_key_follows_what_decides_the_rows(change):
    """A cache built with another seed or shuffle setting holds other rows, so it is another cache."""
    config = BlendConfig(sources=[DatasetSourceConfig(hf_path="x", ratio=1.0, split="train")])
    settings = {"seed": 42, "shuffle": True, "shuffle_buffer": 10000}

    same = _build_cache_path(config, _Tokenizer(), 128, "cache", **settings)
    changed = _build_cache_path(config, _Tokenizer(), 128, "cache", **{**settings, **change})

    assert same == _build_cache_path(config, _Tokenizer(), 128, "cache", **settings)
    assert changed != same
