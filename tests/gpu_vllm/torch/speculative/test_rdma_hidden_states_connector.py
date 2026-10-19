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

"""Regressions for hidden-state cache layout and request addressing."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

extract_hidden_states = pytest.importorskip("vllm.model_executor.models.extract_hidden_states")

from vllm.config import CacheConfig, KVTransferConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1 import kv_cache_interface
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from modelopt.torch.speculative.plugins.rdma_hidden_states_connector import (
    RdmaHiddenStatesConnector,
    build_request_slot_mapping,
    extract_from_kv_cache,
    get_cache_layout,
)


@pytest.mark.parametrize("layout", ["BHNC", "BNHC"])
def test_extract_preserves_capture_planes(layout):
    num_blocks, num_planes, block_size, hidden_size = 10, 6, 4, 3
    cache = torch.arange(
        num_blocks * num_planes * block_size * hidden_size, dtype=torch.float32
    ).reshape(num_blocks, num_planes, block_size, hidden_size)
    expected = torch.stack((cache[0, :, 1, :], cache[1, :, 2, :]))
    if layout == "BNHC":
        cache = cache.transpose(1, 2).contiguous()

    actual = extract_from_kv_cache(cache, torch.tensor([1, 6]), 2, num_planes)

    assert actual.shape == (2, num_planes, hidden_size)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("shape", "planes"),
    [((2, 16, 4), 6), ((2, 6, 6, 3), 6), ((2, 4, 5, 3), 6)],
)
def test_rejects_unknown_or_ambiguous_cache_layout(shape, planes):
    with pytest.raises(ValueError):
        get_cache_layout(torch.empty(shape), planes)


@pytest.mark.parametrize("block_ids", [[-1, 2], [8, 2], [1]])
def test_rejects_invalid_request_blocks(block_ids):
    with pytest.raises(ValueError):
        build_request_slot_mapping(block_ids, 5, 4, "cpu", 8)


class _NixlAgent:
    def __init__(self, *args, **kwargs):
        pass

    def register_memory(self, tensors):
        return tensors

    def get_agent_metadata(self):
        return b"agent"

    def get_xfer_descs(self, tensors):
        return tensors

    def get_serialized_descs(self, descriptors):
        return b"descriptor"


@pytest.fixture
def connector_config(cuda_parallel_state):
    config = VllmConfig(
        cache_config=CacheConfig(block_size=16, cache_dtype="bfloat16"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="RdmaHiddenStatesConnector",
            kv_connector_module_path="modelopt.torch.speculative.plugins.rdma_hidden_states_connector",
            kv_role="kv_producer",
            kv_connector_extra_config={"pool_slots": 2, "max_tokens": 24},
        ),
    )
    config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 3, 5, 7, 9, 11])
        )
    )
    with set_current_vllm_config(config):
        layer = extract_hidden_states.CacheOnlyAttentionLayer(
            num_heads=6, head_size=3, prefix="capture"
        )
    capture_spec = layer.get_kv_cache_spec(config)
    target_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=3, dtype=torch.bfloat16
    )
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["target"], kv_cache_spec=target_spec),
            KVCacheGroupSpec(layer_names=["capture"], kv_cache_spec=capture_spec),
        ],
    )
    return config, cache_config, capture_spec


@pytest.fixture
def isolated_transport(monkeypatch):
    # Isolate RDMA registration and the HTTP listener; retain CUDA copies and events.
    nixl_api = ModuleType("nixl._api")
    nixl_api.nixl_agent = _NixlAgent
    nixl_api.nixl_agent_config = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "nixl", ModuleType("nixl"))
    monkeypatch.setitem(sys.modules, "nixl._api", nixl_api)
    monkeypatch.setattr(RdmaHiddenStatesConnector, "_start_sidecar", lambda self: None)


@pytest.fixture
def cuda_parallel_state(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the capture and pinned-memory copy")
    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path / 'distributed'}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


def _allocate_capture_cache(spec, num_blocks):
    backend = extract_hidden_states.CacheOnlyAttentionBackend
    if hasattr(backend, "get_kv_cache_shape"):
        shape = backend.get_kv_cache_shape(
            num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size
        )
    else:
        shape_bytes = kv_cache_interface.compute_layer_kv_cache_shape_bytes(spec, num_blocks)
        element_size = torch.empty((), dtype=spec.dtype).element_size()
        shape = (*shape_bytes[:-1], shape_bytes[-1] // element_size)
    return torch.full(shape, -1, device="cuda", dtype=spec.dtype)


def test_cuda_capture_preserves_request_and_plane_order(connector_config, isolated_transport):
    config, cache_config, spec = connector_config
    connector = RdmaHiddenStatesConnector(config, KVConnectorRole.WORKER, cache_config)
    cache = _allocate_capture_cache(spec, cache_config.num_blocks)
    connector.register_kv_caches({"capture": cache})
    assert connector._capture_group_id == 1

    requests = [
        SimpleNamespace(
            req_id="request-a",
            prompt_token_ids=list(range(21)),
            num_computed_tokens=0,
            block_ids=[[0, 1], [3, 6]],
        ),
        SimpleNamespace(
            req_id="request-b",
            prompt_token_ids=list(range(18)),
            num_computed_tokens=0,
            block_ids=[[2, 3], [5, 1]],
        ),
    ]
    # Scheduler order differs from the runner's order and cache block order.
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=requests[::-1],
        num_scheduled_tokens={
            request.req_id: len(request.prompt_token_ids) for request in requests
        },
    )
    metadata = connector.build_connector_meta(scheduler_output)
    connector.bind_connector_metadata(metadata)
    inputs = {}
    mappings = []
    for index, request in enumerate(requests):
        count = len(request.prompt_token_ids)
        values = (torch.arange(count * 6 * 3, device="cuda").reshape(count, 6, 3) + index * 500).to(
            spec.dtype
        )
        inputs[request.req_id] = values.cpu()
        positions = torch.arange(count, device="cuda")
        blocks = torch.tensor(request.block_ids[1], device="cuda")
        mapping = (
            blocks[positions // spec.block_size] * spec.block_size + positions % spec.block_size
        )
        mappings.append(mapping)

    slot_mapping = torch.cat(mappings)
    values = torch.cat([inputs[request.req_id] for request in requests]).cuda()
    extract_hidden_states.basic_cache(values, cache, slot_mapping)
    connector.save_kv_layer(
        "capture", cache, extract_hidden_states.CacheOnlyAttentionMetadata(slot_mapping)
    )
    connector._cs().synchronize()

    for request in metadata.requests:
        expected = inputs[request.req_id]
        actual = connector._pool[request.slot, : expected.numel()].reshape(expected.shape)
        torch.testing.assert_close(actual, expected)
        assert connector._bufs[request.req_id]["shape"] == tuple(expected.shape)
        assert connector._bufs[request.req_id]["slot"] == request.slot


def test_registration_rejects_missing_eagle_layer_ids(connector_config, isolated_transport):
    config, cache_config, spec = connector_config
    config.speculative_config.draft_model_config.hf_config = SimpleNamespace()
    connector = RdmaHiddenStatesConnector(config, KVConnectorRole.WORKER, cache_config)
    cache = _allocate_capture_cache(spec, cache_config.num_blocks)
    with pytest.raises(ValueError, match="EAGLE"):
        connector.register_kv_caches({"capture": cache})
