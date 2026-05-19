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

"""Tests for StreamingDataset's DDP sharding contract.

We do not spin up real torch.distributed; instead we monkeypatch the helper that
reads rank/world_size. The dataset only needs to see the right tuple to compute
its shard.
"""

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import safetensors.torch
import torch

from modelopt.torch.speculative.plugins import hf_streaming_dataset
from modelopt.torch.speculative.plugins.hf_streaming_dataset import (
    EagleVllmStreamingConfig,
    EagleVllmStreamingDataset,
    StreamingConfig,
    StreamingDataset,
)


def _entries(n: int) -> list[dict]:
    """Just enough shape for __init__; we never iterate so tokenizer is unused."""
    return [{"id": i} for i in range(n)]


@pytest.fixture
def patch_dist(monkeypatch):
    """Return a setter; tests call it with (rank, world) to simulate a DDP rank.

    Patches ``modelopt.torch.utils.distributed.rank/size`` as imported into the
    streaming dataset module (``dist_utils``); the dataset calls these once in
    ``__init__`` to compute its shard.
    """

    def _set(rank: int, world: int):
        monkeypatch.setattr(hf_streaming_dataset.dist_utils, "rank", lambda: rank)
        monkeypatch.setattr(hf_streaming_dataset.dist_utils, "size", lambda: world)

    return _set


def _entry_ids(ds: StreamingDataset) -> list[int]:
    return [e["id"] for e in ds.entries]


@pytest.mark.parametrize("world", [1, 2, 3, 8])
def test_shards_partition_corpus(patch_dist, world):
    """Union of all ranks' entries == full corpus, with no overlap."""
    corpus = _entries(100)
    cfg = StreamingConfig(seed=42)

    seen: list[int] = []
    counts: list[int] = []
    for rank in range(world):
        patch_dist(rank, world)
        ds = StreamingDataset(corpus, tokenizer=MagicMock(), config=cfg)
        ids = _entry_ids(ds)
        counts.append(len(ids))
        seen.extend(ids)

    assert sorted(seen) == list(range(100)), "shards must cover the corpus exactly once"
    assert max(counts) - min(counts) <= 1, f"shard sizes unbalanced: {counts}"


def test_same_seed_same_partition(patch_dist):
    """Two constructions with the same seed must yield identical shards on a given rank."""
    corpus = _entries(50)
    cfg = StreamingConfig(seed=7)
    patch_dist(1, 4)
    a = _entry_ids(StreamingDataset(corpus, tokenizer=MagicMock(), config=cfg))
    b = _entry_ids(StreamingDataset(corpus, tokenizer=MagicMock(), config=cfg))
    assert a == b


def test_different_seed_different_order(patch_dist):
    """Sanity: changing the seed actually reshuffles (else the shared-seed contract
    would be vacuous)."""
    corpus = _entries(50)
    patch_dist(0, 1)
    a = _entry_ids(StreamingDataset(corpus, tokenizer=MagicMock(), config=StreamingConfig(seed=1)))
    b = _entry_ids(StreamingDataset(corpus, tokenizer=MagicMock(), config=StreamingConfig(seed=2)))
    assert a != b
    assert sorted(a) == sorted(b)


def test_empty_shard_raises(patch_dist):
    """world_size > len(entries) leaves some ranks empty -> would deadlock DDP. Fail loud.

    With 3 entries striped across 5 ranks, rank 4 always gets 0 entries regardless
    of which permutation the seed produces.
    """
    patch_dist(4, 5)
    with pytest.raises(ValueError, match="got 0 entries after sharding"):
        StreamingDataset(_entries(3), tokenizer=MagicMock(), config=StreamingConfig(seed=0))


def test_iter_rejects_dataloader_workers(patch_dist, monkeypatch):
    """Iterating from within a DataLoader worker must raise — multiple workers would
    each spawn an asyncio loop and N× the request load on the server."""
    patch_dist(0, 1)
    ds = StreamingDataset(_entries(4), tokenizer=MagicMock(), config=StreamingConfig(seed=0))
    # Pretend we're inside a DataLoader worker.
    monkeypatch.setattr(hf_streaming_dataset, "get_worker_info", lambda: MagicMock())
    with pytest.raises(RuntimeError, match="dataloader_num_workers=0"):
        next(iter(ds))


def test_empty_corpus_raises(patch_dist):
    patch_dist(0, 1)
    with pytest.raises(ValueError, match="entries is empty"):
        StreamingDataset([], tokenizer=MagicMock(), config=StreamingConfig())


def test_set_resume_position_skips_entries_without_fetching(patch_dist):
    """Resume should fast-forward inside the dataset without invoking _fetch.

    Verifies the contract relied on by StreamingResumeCallback: skipped entries
    are not sent to the server, so resume costs nothing on the inference side.
    """
    patch_dist(0, 1)
    fetched_ids: list[int] = []

    class _Track(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        async def _fetch(self, client, sample):
            fetched_ids.append(int(sample["cid"]))

    corpus = _entries(10)
    ds = _Track(corpus, tokenizer=MagicMock(), config=StreamingConfig(seed=0, prefetch=2))
    ds.set_resume_position(5)
    list(ds)

    expected = {e["id"] for e in ds.entries[5:]}
    assert set(fetched_ids) == expected
    # _resume_skip is one-shot
    assert ds._resume_skip == 0


def test_circuit_breaker_trips_on_consecutive_fetch_failures(patch_dist):
    """When _fetch keeps failing, the producer raises after the threshold so the
    trainer sees a clear error instead of a silent empty epoch."""
    patch_dist(0, 1)
    threshold = 3

    class _AlwaysFails(StreamingDataset):
        # Bypass tokenization so we don't need a real tokenizer.
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        async def _fetch(self, client, sample):
            raise RuntimeError("simulated server failure")

    ds = _AlwaysFails(
        _entries(20),
        tokenizer=MagicMock(),
        config=StreamingConfig(seed=0, prefetch=2, fail_after_consecutive_skips=threshold),
    )
    with pytest.raises(RuntimeError, match="consecutive _fetch failures"):
        list(ds)


# ---------------------------------------------------------------------------
# Integration test: EagleVllmStreamingDataset against a mocked vLLM server.
# Exercises the full pipeline (HTTP → safetensors → align → format) without
# launching real vLLM. The transport is mocked but the safetensors IO,
# tensor reshaping, and dict assembly are real.
# ---------------------------------------------------------------------------


def _write_canned_safetensors(path: Path, seq: int, n_layers: int, hidden: int) -> None:
    """Mimic what vLLM's ExampleHiddenStatesConnector writes per request."""
    safetensors.torch.save_file(
        {
            "token_ids": torch.arange(seq, dtype=torch.int64),
            "hidden_states": torch.randn(seq, n_layers, hidden),
        },
        str(path),
    )


def _tokenizer_returning(seq: int) -> MagicMock:
    """Tokenizer mock whose apply_chat_template yields a fixed seq-len output."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = {
        "input_ids": torch.arange(seq, dtype=torch.long).unsqueeze(0),
    }
    return tok


def test_eagle_vllm_dataset_end_to_end(tmp_path, patch_dist, monkeypatch):
    """Drive EagleVllmStreamingDataset against an in-process mocked server.

    Verifies that the wire-format → tensor → batch-dict chain produces dicts
    matching what EagleOfflineDataCollator expects, and that scratch files
    are cleaned up after each fetch.
    """
    patch_dist(0, 1)
    seq, n_layers, hidden = 8, 3, 16  # n_layers = 1 final + 2 aux
    scratch = tmp_path / "vllm_scratch"
    scratch.mkdir()

    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        path = scratch / f"req_{counter['n']}.safetensors"
        _write_canned_safetensors(path, seq, n_layers, hidden)
        return httpx.Response(
            200,
            json={"kv_transfer_params": {"hidden_states_path": str(path)}},
        )

    real_async_client = httpx.AsyncClient

    def mock_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(hf_streaming_dataset.httpx, "AsyncClient", mock_async_client)

    n_entries = 4
    entries = [
        {
            "conversation_id": f"c-{i}",
            "messages": [{"role": "user", "content": "x"}],
        }
        for i in range(n_entries)
    ]
    ds = EagleVllmStreamingDataset(
        entries=entries,
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_url="http://mock:8000",
            model="mock-model",
            prefetch=2,
            seed=0,
        ),
    )

    batches = list(ds)

    assert len(batches) == n_entries
    expected_keys = {
        "input_ids",
        "base_model_hidden_states",
        "aux_hidden_states",
        "attention_mask",
        "loss_mask",
        "labels",
    }
    for b in batches:
        assert set(b) == expected_keys
        assert b["input_ids"].shape == (seq,)
        assert b["input_ids"].dtype == torch.int64
        assert b["base_model_hidden_states"].shape == (seq, hidden)
        # 2 aux layers * hidden, flattened
        assert b["aux_hidden_states"].shape == (seq, 2 * hidden)
        assert b["attention_mask"].shape == (seq,)
        assert b["loss_mask"].shape == (seq,)
        assert b["labels"].shape == (seq,)
        # labels are input_ids shifted by 1, last position is IGNORE
        assert torch.equal(b["labels"][:-1], b["input_ids"][1:])
        assert b["labels"][-1].item() == hf_streaming_dataset.IGNORE_TOKEN_ID

    # Scratch files must be unlinked after fetch
    assert list(scratch.iterdir()) == []
