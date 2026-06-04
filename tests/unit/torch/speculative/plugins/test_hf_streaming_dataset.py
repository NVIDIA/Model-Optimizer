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

"""Tests for the map-style StreamingDataset.

The dataset is a plain ``torch.utils.data.Dataset``: DDP sharding is HF Trainer's
job (``DistributedSampler``), so there is no rank/dispatch logic to test here.
These tests cover the ``__getitem__`` contract: resample-on-miss, the
consecutive-failure circuit breaker, and the vLLM wire-format -> batch-dict chain.
"""

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import safetensors.torch
import torch

# hf_streaming_dataset imports LabelSmoother at module scope.
pytest.importorskip("transformers")

from modelopt.torch.speculative.plugins import hf_streaming_dataset
from modelopt.torch.speculative.plugins.hf_streaming_dataset import (
    EagleVllmStreamingConfig,
    EagleVllmStreamingDataset,
    StreamingConfig,
    StreamingDataset,
)


def _entries(n: int) -> list[dict]:
    """Minimal entry shape; ``id`` is the only field tests read back."""
    return [{"id": i} for i in range(n)]


def test_empty_corpus_raises():
    with pytest.raises(ValueError, match="entries is empty"):
        StreamingDataset([], tokenizer=MagicMock(), config=StreamingConfig())


def test_len_matches_corpus():
    ds = StreamingDataset(_entries(37), tokenizer=MagicMock(), config=StreamingConfig())
    assert len(ds) == 37


def test_getitem_resamples_past_unfit_entries():
    """An unfit entry (tokenize -> None) must not be returned; __getitem__ probes
    forward to the next fetchable index and returns that instead."""
    fetched_cids: list[int] = []

    class _Track(StreamingDataset):
        def _tokenize_entry(self, entry):
            # Even ids are "unfit" (e.g. truncated away / missing fields).
            if entry["id"] % 2 == 0:
                return None
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            fetched_cids.append(int(sample["cid"]))
            return {"ok": True}

        def _format(self, fetched):
            return {"sentinel": fetched_cids[-1]}

    ds = _Track(_entries(10), tokenizer=MagicMock(), config=StreamingConfig())
    # idx 0 is unfit -> resamples forward to idx 1.
    out = ds[0]
    assert out == {"sentinel": 1}
    assert fetched_cids == [1]
    # An already-fit index is returned directly.
    assert ds[3] == {"sentinel": 3}


def test_circuit_breaker_trips_on_consecutive_failures():
    """When _fetch keeps hitting transient errors (server down), __getitem__ raises
    after the threshold instead of silently resampling the whole corpus."""
    threshold = 3

    class _AlwaysFails(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            # A down server surfaces as a transport error, which the breaker counts.
            raise httpx.ConnectError("simulated server down")

    ds = _AlwaysFails(
        _entries(20),
        tokenizer=MagicMock(),
        config=StreamingConfig(fail_after_consecutive_skips=threshold),
    )
    with pytest.raises(RuntimeError, match="consecutive _fetch failures"):
        ds[0]


def test_contract_violation_propagates_not_swallowed():
    """A non-transient error from _fetch (e.g. a contract violation / bug) must
    surface immediately, not be masked as a fetch miss and silently resampled."""

    class _BadContract(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            raise RuntimeError("server token_ids drift")

    ds = _BadContract(
        _entries(20),
        tokenizer=MagicMock(),
        # High threshold: if the error were (wrongly) swallowed, the breaker wouldn't
        # fire, so a leaked breaker message would mask the regression.
        config=StreamingConfig(fail_after_consecutive_skips=100),
    )
    with pytest.raises(RuntimeError, match="server token_ids drift"):
        ds[0]


def test_fetch_returning_none_exhausts_then_raises():
    """If every entry's fetch yields None (e.g. all rejected), __getitem__ raises a
    clear 'no fetchable sample' error rather than hanging or returning junk."""

    class _AllNone(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            return None

    ds = _AllNone(
        _entries(4),
        tokenizer=MagicMock(),
        config=StreamingConfig(fail_after_consecutive_skips=100),
    )
    with pytest.raises(RuntimeError, match="no fetchable sample"):
        ds[0]


def test_resume_skips_consumed_samples_without_refetching():
    """Map-style resume contract: HF Trainer skips consumed batches via
    accelerate.skip_first_batches, which drops their indices at the batch-sampler
    level so __getitem__ (and thus _fetch) is never called for them. This is why
    main.py leaves ignore_data_skip at its default (False) for streaming -- resume
    lands at the exact position with no re-fetch. Guards against a regression that
    would re-fetch (or re-stream) already-consumed samples on resume."""
    pytest.importorskip("accelerate")
    from accelerate import skip_first_batches
    from torch.utils.data import DataLoader, RandomSampler

    fetched: list[int] = []

    class _Recording(StreamingDataset):
        def _tokenize_entry(self, entry):
            return {"cid": str(entry["id"]), "token_ids": [1], "loss_mask": None}

        def _fetch(self, sample):
            cid = int(sample["cid"])
            fetched.append(cid)  # stands in for the HTTP fetch
            return {"cid": cid}

        def _format(self, payload):
            return torch.tensor(payload["cid"])

    n, batch_size, skip_batches = 20, 2, 3
    ds = _Recording(_entries(n), tokenizer=MagicMock(), config=StreamingConfig())

    def make_dl():
        # Fresh, identically-seeded sampler -> identical permutation across runs.
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(0)),
        )

    # Full pass -> ground-truth consumption order (cid == requested index here).
    full_order = [int(x) for batch in make_dl() for x in batch]
    fetched.clear()

    # Resume: skip the first `skip_batches` batches.
    tail_order = [int(x) for batch in skip_first_batches(make_dl(), skip_batches) for x in batch]

    consumed = full_order[: skip_batches * batch_size]
    expected_tail = full_order[skip_batches * batch_size :]
    assert tail_order == expected_tail, "resume must continue at the exact data position"
    assert set(fetched).isdisjoint(consumed), "skipped (consumed) samples must not be re-fetched"
    assert fetched == expected_tail, "only the un-consumed tail is fetched after resume"


def test_server_urls_normalization():
    """server_urls accepts a single string, a comma-separated string, or a list, and
    strips trailing slashes."""

    def _urls(v):
        cfg = EagleVllmStreamingConfig(
            server_urls=v, model="m", shared_storage_root=str(Path.cwd())
        )
        return cfg.server_urls

    assert _urls("http://a:8000/") == ["http://a:8000"]
    assert _urls("http://a:8000, http://b:8000/") == ["http://a:8000", "http://b:8000"]
    assert _urls(["http://a:8000", "http://b:8000"]) == ["http://a:8000", "http://b:8000"]
    with pytest.raises(ValueError, match="at least one non-empty URL"):
        EagleVllmStreamingConfig(server_urls="", model="m", shared_storage_root=".")


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


def _patch_sync_client(monkeypatch, handler):
    """Route the dataset's per-process httpx.Client through a MockTransport handler."""
    real_client = httpx.Client

    def mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(hf_streaming_dataset.httpx, "Client", mock_client)


def test_eagle_vllm_dataset_end_to_end(tmp_path, monkeypatch):
    """Drive EagleVllmStreamingDataset against an in-process mocked server.

    Verifies the wire-format -> tensor -> batch-dict chain produces dicts matching
    what EagleOfflineDataCollator expects, and that scratch files are cleaned up.
    """
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

    _patch_sync_client(monkeypatch, handler)

    n_entries = 4
    entries = [
        {"conversation_id": f"c-{i}", "messages": [{"role": "user", "content": "x"}]}
        for i in range(n_entries)
    ]
    ds = EagleVllmStreamingDataset(
        entries=entries,
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            shared_storage_root=str(scratch),
        ),
    )

    batches = [ds[i] for i in range(n_entries)]

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

    assert list(scratch.iterdir()) == [], "scratch files must be unlinked after fetch"


def test_fetch_round_robins_across_server_urls(tmp_path, monkeypatch):
    """With multiple server_urls, consecutive fetches alternate across endpoints so
    load is spread over replicas rather than pinned to the first one."""
    seq, n_layers, hidden = 8, 3, 16
    scratch = tmp_path / "vllm_scratch"
    scratch.mkdir()

    hosts: list[str] = []
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        counter["n"] += 1
        path = scratch / f"req_{counter['n']}.safetensors"
        _write_canned_safetensors(path, seq, n_layers, hidden)
        return httpx.Response(
            200,
            json={"kv_transfer_params": {"hidden_states_path": str(path)}},
        )

    _patch_sync_client(monkeypatch, handler)

    n_entries = 4
    entries = [
        {"conversation_id": f"c-{i}", "messages": [{"role": "user", "content": "x"}]}
        for i in range(n_entries)
    ]
    ds = EagleVllmStreamingDataset(
        entries=entries,
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls=["http://a:8000", "http://b:8000"],
            model="mock-model",
            shared_storage_root=str(scratch),
        ),
    )

    for i in range(n_entries):
        ds[i]

    # Per-process round-robin cursor: a, b, a, b -- one request each, alternating.
    assert hosts == ["a", "b", "a", "b"]


def test_path_outside_shared_storage_root_is_rejected(tmp_path, monkeypatch):
    """Out-of-root path from the server is not opened or unlinked; the fetch yields
    None, so the single-entry corpus is exhausted and __getitem__ raises."""
    seq, n_layers, hidden = 8, 3, 16
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    forbidden = outside / "secret.safetensors"
    _write_canned_safetensors(forbidden, seq, n_layers, hidden)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"kv_transfer_params": {"hidden_states_path": str(forbidden)}},
        )

    _patch_sync_client(monkeypatch, handler)

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            shared_storage_root=str(allowed),
            fail_after_consecutive_skips=100,
        ),
    )

    with pytest.raises(RuntimeError, match="no fetchable sample"):
        ds[0]
    assert forbidden.exists(), "rejected path must not be unlinked"


def test_load_safetensors_retries_past_writer_race(tmp_path, monkeypatch):
    """The connector writes asynchronously, so an immediate read can race it;
    _load_safetensors must retry past the transient FileNotFound/Safetensor error."""
    seq, n_layers, hidden = 4, 2, 8
    path = tmp_path / "late.safetensors"
    _write_canned_safetensors(path, seq, n_layers, hidden)

    calls = {"n": 0}
    real_safe_open = hf_streaming_dataset.safe_open

    def flaky_safe_open(p, framework):
        calls["n"] += 1
        if calls["n"] < 3:  # first 2 reads race the writer (file not ready yet)
            raise FileNotFoundError(f"No such file or directory: {p}")
        return real_safe_open(p, framework=framework)

    monkeypatch.setattr(hf_streaming_dataset, "safe_open", flaky_safe_open)
    monkeypatch.setattr(hf_streaming_dataset.time, "sleep", lambda *_: None)  # no real backoff

    token_ids, hidden_states = EagleVllmStreamingDataset._load_safetensors(str(path))
    assert calls["n"] == 3
    assert hidden_states.shape == (seq, n_layers, hidden)
