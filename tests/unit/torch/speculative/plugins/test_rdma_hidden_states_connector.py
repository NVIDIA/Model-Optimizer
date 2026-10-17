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

"""Regressions for vLLM hidden-state cache layout and request addressing."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from modelopt.torch.speculative.plugins.rdma_hidden_states_connector import (
    RdmaConnMeta,
    RdmaHiddenStatesConnector,
    ReqMeta,
    build_request_slot_mapping,
    extract_from_kv_cache,
)


def test_extract_from_bhnc_cache_preserves_capture_planes():
    """The KV block axis must not be mistaken for the hidden-state plane axis."""
    num_blocks, num_planes, block_size, hidden_size = 10, 6, 4, 3
    cache = torch.arange(
        num_blocks * num_planes * block_size * hidden_size, dtype=torch.float32
    ).reshape(num_blocks, num_planes, block_size, hidden_size)
    slot_mapping = torch.tensor([1, block_size + 2])

    actual = extract_from_kv_cache(cache, slot_mapping, num_tokens=2)
    expected = torch.stack((cache[0, :, 1, :], cache[1, :, 2, :]))

    assert actual.shape == (2, num_planes, hidden_size)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("block_ids", "num_tokens"),
    [([7, 2], 6), ([4], 3), ([9, 1], 8)],
)
def test_request_addressing_is_independent_of_batch_order(block_ids, num_tokens):
    """Each request follows its own noncontiguous blocks, not a batch offset."""
    block_size = 4
    mapping = build_request_slot_mapping(block_ids, num_tokens, block_size, "cpu")
    expected = torch.tensor(
        [
            block_ids[position // block_size] * block_size + position % block_size
            for position in range(num_tokens)
        ]
    )
    torch.testing.assert_close(mapping, expected)


def test_request_addressing_rejects_insufficient_blocks():
    with pytest.raises(ValueError, match="do not cover"):
        build_request_slot_mapping([1], 5, 4, "cpu")


def test_scheduler_metadata_carries_request_specific_capture_blocks():
    """The worker receives the capture group's blocks keyed to each request."""
    connector = object.__new__(RdmaHiddenStatesConnector)
    connector._slot_ctr = 0
    connector._pool_slots = 4
    connector._capture_group_id = 1
    request = SimpleNamespace(
        req_id="request-b",
        prompt_token_ids=list(range(6)),
        num_computed_tokens=0,
        block_ids=[[0, 1], [7, 2]],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[request],
        num_scheduled_tokens={"request-b": 6},
    )

    metadata = connector.build_connector_meta(scheduler_output)

    assert len(metadata.requests) == 1
    assert metadata.requests[0].req_id == "request-b"
    assert metadata.requests[0].block_ids == [7, 2]


def test_save_kv_layer_captures_each_requests_own_noncontiguous_blocks(monkeypatch):
    """Production capture must not associate scheduler order with cache position."""
    from vllm.model_executor.models import extract_hidden_states

    class AttentionMetadata:
        pass

    class Event:
        def record(self, stream=None):
            pass

    class Stream:
        def wait_event(self, event):
            pass

    class Nixl:
        def get_xfer_descs(self, tensors):
            return tensors

        def get_serialized_descs(self, descriptors):
            return b"descriptor"

    monkeypatch.setattr(extract_hidden_states, "CacheOnlyAttentionMetadata", AttentionMetadata)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    block_size = 4
    kv_layer = torch.arange(8 * 2 * block_size * 3, dtype=torch.float32).reshape(
        8, 2, block_size, 3
    )
    requests = [
        ReqMeta.make("request-b", range(6), slot=1, block_ids=[5, 1]),
        ReqMeta.make("request-a", range(5), slot=0, block_ids=[3, 0]),
    ]
    connector = object.__new__(RdmaHiddenStatesConnector)
    connector._owner = True
    connector.cache_layers = ["capture"]
    connector._get_connector_metadata = lambda: RdmaConnMeta(requests=requests)
    connector._cs = lambda: Stream()
    connector._per_token_elems = 2 * 3
    connector._feat_shape = (2, 3)
    connector._dtype = kv_layer.dtype
    connector._slot_elems = 6 * connector._per_token_elems
    connector._pool = torch.full((2, connector._slot_elems), -1.0)
    connector._nixl = Nixl()
    connector._lock = threading.Lock()
    connector._slot_gen = {}
    connector._bufs = {}
    connector._oversize_warned = False
    connector._max_tokens = 6

    connector.save_kv_layer("capture", kv_layer, AttentionMetadata())

    for request in requests:
        mapping = build_request_slot_mapping(
            request.block_ids, len(request.token_ids), block_size, "cpu"
        )
        expected = extract_from_kv_cache(kv_layer, mapping, len(request.token_ids)).flatten()
        captured = connector._pool[request.slot, : expected.numel()]
        torch.testing.assert_close(captured, expected)
        assert connector._bufs[request.req_id]["slot"] == request.slot


def test_extract_from_kv_cache_rejects_unknown_layout():
    """Fail explicitly if a future vLLM release changes the per-layer cache view."""
    with pytest.raises(ValueError, match=r"\[blocks, heads, block_size, head_size\]"):
        extract_from_kv_cache(torch.empty(2, 16, 4), torch.tensor([0]), num_tokens=1)
