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

Every mistake these tests guard against is silent: the draft trains on another token's or
another request's hidden states, and nothing raises.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    UniformTypeKVCacheSpecs,
)

from modelopt.torch.speculative.plugins import rdma_hidden_states_connector
from modelopt.torch.speculative.plugins.hf_streaming_dataset import (
    EagleVllmStreamingConfig,
    MegatronEagleVllmStreamingDataset,
)
from modelopt.torch.speculative.plugins.rdma_hidden_states_connector import (
    RdmaHiddenStatesConnector,
    _hidden_state_group_index,
    planes_dim,
)

NUM_BLOCKS, BLOCK_SIZE, N_PLANES, HIDDEN = 5, 4, 3, 8

ATTENTION = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
CAPTURE = HiddenStateCacheSpec(
    block_size=1, num_kv_heads=N_PLANES, head_size=HIDDEN, dtype=torch.bfloat16
)


def _groups(*specs):
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=s, layer_names=["layer"]) for s in specs]
    )


def _worker_connector(kv):
    """A worker-side connector over ``kv``, as ``register_kv_caches`` leaves it, minus NIXL."""
    extra = {"pool_slots": "2", "max_tokens": "8"}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda k, d: extra.get(k, d))
    )
    conn = RdmaHiddenStatesConnector(config, KVConnectorRole.WORKER, _groups(ATTENTION, CAPTURE))
    conn.cache_layers = ["capture"]
    conn._planes_dim = planes_dim(kv, N_PLANES)
    conn._cache_block_size, conn._num_blocks = BLOCK_SIZE, NUM_BLOCKS
    conn._feat_shape, conn._dtype = (N_PLANES, HIDDEN), kv.dtype
    conn._per_token_elems = N_PLANES * HIDDEN
    conn._slot_elems = conn._max_tokens * conn._per_token_elems
    conn._pool = torch.zeros(conn._pool_slots, conn._slot_elems, dtype=kv.dtype).pin_memory()
    conn._nixl = SimpleNamespace(get_xfer_descs=lambda t: None, get_serialized_descs=lambda d: b"")
    return conn


@pytest.mark.parametrize("layout", ["BNHC", "BHNC"])
def test_concurrent_requests_each_capture_their_own_hidden_states(layout, monkeypatch):
    """Two prompts prefilled in one step, batched behind a request that is already decoding.

    Their capture blocks are neither contiguous nor in batch order, and the attention group
    holds other block ids, so reading the step's slot_mapping by position, or the wrong
    group's blocks, hands a request another request's rows. vLLM 0.29 (#51718) moved the
    cache view from [B, N, H, C] to [B, H, N, C].
    """
    tokens = torch.randn(NUM_BLOCKS * BLOCK_SIZE, N_PLANES, HIDDEN, device="cuda")  # row = slot
    kv = tokens.view(NUM_BLOCKS, BLOCK_SIZE, N_PLANES, HIDDEN)
    if layout == "BHNC":
        kv = kv.transpose(1, 2).contiguous()
    conn = _worker_connector(kv)

    # block_ids holds one list per group: attention first, then the capture layer.
    new_reqs = [
        SimpleNamespace(req_id="A", prompt_token_ids=list(range(6)), block_ids=([7, 8], [3, 1])),
        SimpleNamespace(req_id="B", prompt_token_ids=list(range(3)), block_ids=([9], [0])),
    ]
    scheduled = SimpleNamespace(scheduled_new_reqs=new_reqs, num_scheduled_tokens={"A": 6, "B": 3})
    conn.bind_connector_metadata(conn.build_connector_meta(scheduled))
    # The step's batch: the decoding request (slot 19) first, then B, then A.
    slot_mapping = torch.tensor([19, 0, 1, 2, 12, 13, 14, 15, 4, 5], device="cuda")
    monkeypatch.setattr(
        rdma_hidden_states_connector,
        "get_forward_context",
        lambda: SimpleNamespace(slot_mapping={"capture": slot_mapping}),
    )
    conn.save_kv_layer("capture", kv, CacheOnlyAttentionMetadata(slot_mapping))
    torch.cuda.synchronize()

    # Pool slots go to requests in scheduling order: A's 6 tokens, then B's 3.
    for pool_slot, slots in [(0, [12, 13, 14, 15, 4, 5]), (1, [0, 1, 2])]:
        want = tokens[slots].flatten().cpu()
        assert torch.equal(conn._pool[pool_slot, : want.numel()], want)


@pytest.mark.parametrize("dims", [(N_PLANES, N_PLANES), (BLOCK_SIZE, BLOCK_SIZE)])
def test_a_view_that_does_not_show_where_the_planes_are_is_refused(dims):
    with pytest.raises(RuntimeError):
        planes_dim(torch.empty(NUM_BLOCKS, *dims, HIDDEN), N_PLANES)


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


def test_live_megatron_stream_at_capture_limit():
    """Exercise real vLLM HTTP/NIXL capture, padding, and advertised slot capacity."""
    url = os.getenv("EAGLE_STREAMING_TEST_URL")
    model_path = os.getenv("EAGLE_STREAMING_TEST_TEACHER")
    if not url or not model_path:
        pytest.skip("start the Qwen3 teacher server in examples/speculative_decoding/README.md")

    cap = 256
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    entries = [
        {
            "conversation_id": name,
            "messages": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": "A drafter proposes tokens and a teacher verifies them.",
                },
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for name, question in (
            ("short", "What does the teacher do?"),
            ("capped", "Explain speculative decoding. " * cap),
        )
    ]
    config = EagleVllmStreamingConfig(
        server_urls=url, model="qwen3-teacher", max_seq_len=cap, request_timeout=15
    )
    dataset = MegatronEagleVllmStreamingDataset(entries, tokenizer, config)

    oversized = config.model_copy(update={"max_seq_len": cap + 1})
    with pytest.raises(
        RuntimeError, match="serve connector max_tokens=256 < trainer max_seq_len=257"
    ):
        MegatronEagleVllmStreamingDataset(entries[:1], tokenizer, oversized)[0]

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .cuda()
    )
    norm_inputs = []
    before_norm = model.model.norm.register_forward_pre_hook(
        lambda _module, args: norm_inputs.append(args[0])
    )
    try:
        for index, entry in enumerate(entries):
            prompt = tokenizer.apply_chat_template(
                entry["messages"], tokenize=True, return_dict=False, enable_thinking=False
            )
            assert (len(prompt) > cap) == (index == 1)
            ids = torch.tensor(prompt[:cap], dtype=torch.long)
            length = ids.numel()
            sample = dataset[index]

            assert sample["input_ids"].shape == sample["loss_mask"].shape == (cap + 1,)
            torch.testing.assert_close(sample["input_ids"][:length], ids)
            assert (sample["input_ids"][length:] == tokenizer.eos_token_id).all()
            assert (sample["loss_mask"][:length] == 1).all()
            assert not sample["loss_mask"][length:].any()

            with torch.inference_mode():
                reference = model(ids[None].cuda(), output_hidden_states=True)
            hidden = model.config.hidden_size
            assert sample["aux_hidden_states"].shape == (cap, 3 * hidden)
            assert sample["hidden_states"].shape == (cap, hidden)
            aux = sample["aux_hidden_states"].reshape(cap, 3, hidden)
            expected = [
                (aux[:length, plane], reference.hidden_states[depth][0])
                for plane, depth in enumerate((2, 14, 25))
            ]
            expected.append((sample["hidden_states"][:length], norm_inputs.pop()[0]))
            for actual, want in expected:
                assert actual.dtype == torch.bfloat16
                similarity = torch.nn.functional.cosine_similarity(
                    actual.cuda().float().flatten(), want.float().flatten(), dim=0
                )
                assert similarity > 0.99
            assert not aux[length:].any()
            assert not sample["hidden_states"][length:].any()
    finally:
        before_norm.remove()
