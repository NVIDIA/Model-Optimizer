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
consecutive-failure circuit breaker, and the vLLM RDMA wire-format -> batch-dict
chain. Hidden states move over NIXL RDMA; the end-to-end tests inject a fake
``nixl`` agent (the library is not a test dependency) and route the sidecar HTTP
through ``httpx.MockTransport`` -- the RDMA transfer itself is a no-op, so they
exercise the orchestration + format chain, not real byte movement.
"""

import base64
import sys
import types
from unittest.mock import MagicMock

import httpx
import pytest
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
            fetched.append(cid)  # stands in for the RDMA fetch
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
        return EagleVllmStreamingConfig(server_urls=v, model="m", max_seq_len=128).server_urls

    assert _urls("http://a:8000/") == ["http://a:8000"]
    assert _urls("http://a:8000, http://b:8000/") == ["http://a:8000", "http://b:8000"]
    assert _urls(["http://a:8000", "http://b:8000"]) == ["http://a:8000", "http://b:8000"]
    with pytest.raises(ValueError, match="at least one non-empty URL"):
        EagleVllmStreamingConfig(server_urls="", model="m", max_seq_len=128)


def test_max_seq_len_is_required():
    """max_seq_len is optional on the base config but required for the RDMA backend (it
    pre-sizes the recv buffer), so a missing/non-positive value must fail at construction
    rather than crashing later in _fetch."""
    with pytest.raises(ValueError, match="max_seq_len"):
        EagleVllmStreamingConfig(server_urls="http://a:8000", model="m")
    with pytest.raises(ValueError, match=r"max_seq_len|greater than 0"):
        EagleVllmStreamingConfig(server_urls="http://a:8000", model="m", max_seq_len=0)


class _FakeNixlAgent:
    """Stand-in for a ``nixl`` agent: the RDMA transfer is a no-op (returns DONE
    immediately), so the recv buffer is left as-is. End-to-end tests assert shapes
    (driven by the sidecar's ``hs_shape``) and the format chain, not transferred bytes.
    """

    def register_memory(self, tensors):
        return MagicMock()

    def deregister_memory(self, reg):
        return None

    def get_agent_metadata(self):
        return b"agent-meta"

    def add_remote_agent(self, meta):
        return "remote-agent"

    def get_xfer_descs(self, views):
        return MagicMock()

    def deserialize_descs(self, blob):
        return MagicMock()

    def initialize_xfer(self, op, ldescs, rdescs, remote):
        return "xfer-handle"

    def transfer(self, handle):
        return None

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        return None


def _mock_rdma(monkeypatch, handler):
    """Inject a fake ``nixl._api`` (not installed in CI) and route the dataset's
    per-process ``httpx.Client`` (sidecar metadata/descriptor calls + the completions
    POST) through a ``MockTransport`` handler."""
    fake_api = types.ModuleType("nixl._api")
    fake_api.nixl_agent = lambda *a, **k: _FakeNixlAgent()
    fake_api.nixl_agent_config = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "nixl", types.ModuleType("nixl"))
    monkeypatch.setitem(sys.modules, "nixl._api", fake_api)

    real_client = httpx.Client

    def mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(hf_streaming_dataset.httpx, "Client", mock_client)


def _tokenizer_returning(seq: int) -> MagicMock:
    """Tokenizer mock whose apply_chat_template yields a fixed seq-len output."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = {
        "input_ids": torch.arange(seq, dtype=torch.long).unsqueeze(0),
    }
    return tok


def _rdma_sidecar_handler(seq, n_layers, hidden, *, on_completion=None, done_valid=True):
    """Build an httpx handler emulating the RdmaHiddenStatesConnector sidecar:
    POST /v1/completions -> {hs_req_id}; GET /meta -> agent metadata;
    GET /desc -> ready descriptor (shape/dtype/token_ids); GET /done -> ack + ``valid``
    (False emulates the ring lapping the slot mid-read).
    ``token_ids`` mirrors the client prompt verbatim (no drift)."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/completions":
            if on_completion is not None:
                on_completion(request)
            return httpx.Response(200, json={"kv_transfer_params": {"hs_req_id": "req-1"}})
        if path == "/meta":
            return httpx.Response(200, json={"agent_metadata": base64.b64encode(b"m").decode()})
        if path == "/desc":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "hs_descs": base64.b64encode(b"d").decode(),
                    "hs_shape": [seq, n_layers, hidden],
                    "hs_dtype": "float32",
                    "token_ids": list(range(seq)),
                    "slot": 0,
                },
            )
        if path == "/done":
            return httpx.Response(200, json={"freed": "req-1", "valid": done_valid})
        return httpx.Response(404, json={"error": "not found"})

    return handler


def test_eagle_vllm_dataset_end_to_end(monkeypatch):
    """Drive EagleVllmStreamingDataset against an in-process mocked RDMA server.

    Verifies the RDMA fetch -> tensor -> batch-dict chain produces dicts matching
    what EagleOfflineDataCollator expects.
    """
    seq, n_layers, hidden = 8, 3, 16  # n_layers = 1 final + 2 aux
    _mock_rdma(monkeypatch, _rdma_sidecar_handler(seq, n_layers, hidden))

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
            max_seq_len=seq,
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


def test_fetch_round_robins_across_server_urls(monkeypatch):
    """With multiple server_urls, consecutive fetches alternate across endpoints so
    load is spread over replicas rather than pinned to the first one."""
    seq, n_layers, hidden = 8, 3, 16
    hosts: list[str] = []
    handler = _rdma_sidecar_handler(
        seq, n_layers, hidden, on_completion=lambda req: hosts.append(req.url.host)
    )
    _mock_rdma(monkeypatch, handler)

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
            max_seq_len=seq,
        ),
    )

    for i in range(n_entries):
        ds[i]

    # Per-process round-robin cursor: a, b, a, b -- one completions POST each, alternating.
    assert hosts == ["a", "b", "a", "b"]


def test_lapped_slot_is_treated_as_miss(monkeypatch):
    """If /done reports valid=False (the ring overwrote the slot mid-read), the fetched
    bytes are stale and must be discarded as a miss -- not returned as training data.
    Here every read is reported lapped, so __getitem__ exhausts and raises rather than
    silently yielding corrupt hidden states."""
    seq, n_layers, hidden = 8, 3, 16
    _mock_rdma(monkeypatch, _rdma_sidecar_handler(seq, n_layers, hidden, done_valid=False))

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
            fail_after_consecutive_skips=100,
        ),
    )
    with pytest.raises(RuntimeError, match="no fetchable sample"):
        ds[0]


def test_oversize_server_response_raises(monkeypatch):
    """If the server captured more tokens than max_seq_len (its connector max_tokens >
    our recv buffer), reading would silently truncate the slice; fail loud instead so the
    size mismatch is configured away rather than trained on misaligned hidden states."""
    seq, n_layers, hidden = 8, 3, 16
    # Sidecar advertises a longer sequence than the recv buffer is sized for.
    _mock_rdma(monkeypatch, _rdma_sidecar_handler(seq + 4, n_layers, hidden))

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
        ),
    )
    with pytest.raises(RuntimeError, match="max_seq_len"):
        ds[0]


def test_sidecar_port_taken_from_response(monkeypatch):
    """The connector advertises its sidecar port per completions response; the dataset
    must address the /meta, /desc and /done sidecar calls at that port (not a hardcoded
    default), so a non-default connector sidecar_port works without an env override."""
    seq, n_layers, hidden = 8, 3, 16
    advertised_port = 23456
    seen_ports: set[int] = set()
    base = _rdma_sidecar_handler(seq, n_layers, hidden)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/completions":
            return httpx.Response(
                200,
                json={
                    "kv_transfer_params": {"hs_req_id": "req-1", "hs_sidecar_port": advertised_port}
                },
            )
        seen_ports.add(request.url.port)
        return base(request)

    _mock_rdma(monkeypatch, handler)

    ds = EagleVllmStreamingDataset(
        entries=[{"conversation_id": "c-0", "messages": [{"role": "user", "content": "x"}]}],
        tokenizer=_tokenizer_returning(seq),
        config=EagleVllmStreamingConfig(
            server_urls="http://mock:8000",
            model="mock-model",
            max_seq_len=seq,
        ),
    )
    ds[0]
    assert seen_ports == {advertised_port}
