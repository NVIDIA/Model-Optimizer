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

"""How the streaming hidden-state connector finds a request's hidden states in vLLM's cache.

CPU-only, but here rather than in tests/unit because the connector imports vLLM at module
level. Every mistake these tests guard against is silent: the draft trains on another
token's or another request's hidden states, and nothing raises.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    UniformTypeKVCacheSpecs,
)

from modelopt.torch.speculative.plugins.rdma_hidden_states_connector import (
    _hidden_state_group_index,
    _request_slots,
    extract_from_kv_cache,
    planes_dim,
)

NUM_BLOCKS, BLOCK_SIZE, N_PLANES, HIDDEN = 5, 4, 3, 8

ATTENTION = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
CAPTURE = HiddenStateCacheSpec(
    block_size=1, num_kv_heads=N_PLANES, head_size=HIDDEN, dtype=torch.bfloat16
)


@pytest.mark.parametrize("layout", ["BNHC", "BHNC"])
def test_a_request_reads_its_own_tokens_from_either_layout(layout):
    """vLLM #51718 (0.29) moved the captured planes from dim 2 of the cache view to dim 1."""
    tokens = torch.randn(NUM_BLOCKS * BLOCK_SIZE, N_PLANES, HIDDEN)  # row i = slot i
    cache = tokens.view(NUM_BLOCKS, BLOCK_SIZE, N_PLANES, HIDDEN)
    if layout == "BHNC":
        cache = cache.transpose(1, 2).contiguous()
    # Holding blocks 3 then 1, the request's 6 tokens sit in slots 12-15 and 4-5.
    slots = _request_slots((3, 1), 6, BLOCK_SIZE)
    got = extract_from_kv_cache(cache, slots, 6, planes_dim(cache, N_PLANES))
    assert torch.equal(got, tokens[[12, 13, 14, 15, 4, 5]])


@pytest.mark.parametrize("dims", [(N_PLANES, N_PLANES), (BLOCK_SIZE, BLOCK_SIZE)])
def test_a_view_that_does_not_show_where_the_planes_are_is_refused(dims):
    with pytest.raises(RuntimeError):
        planes_dim(torch.empty(NUM_BLOCKS, *dims, HIDDEN), N_PLANES)


def _groups(*specs):
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=s, layer_names=["layer"]) for s in specs]
    )


@pytest.mark.parametrize(
    ("config", "index"),
    [
        # Mixed attention types: the capture layer gets a group of its own.
        (_groups(ATTENTION, CAPTURE), 1),
        # One attention type, worker side: the capture layer is merged into its group.
        (
            _groups(
                UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs={"attn": ATTENTION, "capture": CAPTURE}
                )
            ),
            0,
        ),
        # The same group as the scheduler sees it, flattened to one representative spec.
        (_groups(ATTENTION), 0),
    ],
)
def test_the_capture_group_is_found_however_vllm_groups_it(config, index):
    assert _hidden_state_group_index(config) == index


def test_no_capture_group_among_several_is_refused():
    with pytest.raises(RuntimeError):
        _hidden_state_group_index(_groups(ATTENTION, ATTENTION))
