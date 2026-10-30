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

"""The expert-parallel sizing of transformers' load-time caching-allocator warmup.

``from_pretrained`` reserves the bytes ``get_total_byte_count`` reports for each device in one
allocation. Under the EP load it would count every expert at full size (the experts are most of a
large MoE), so ``_ep_aware_caching_allocator_warmup`` swaps in a sizing that counts one EP shard of
each expert. The sizing is called directly here, and once through ``caching_allocator_warmup`` with
the CUDA queries and the allocation faked.
"""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import modeling_utils

from modelopt.torch.utils.plugins.model_load_utils import _ep_aware_caching_allocator_warmup

if not hasattr(modeling_utils, "get_total_byte_count"):
    pytest.skip("transformers has no get_total_byte_count to size", allow_module_level=True)

_GPU = torch.device("cuda", 0)

# The plan load_and_prepare_ep shards with after the a2a swap: base-relative keys (no ``model.``).
_A2A_PLAN = {
    "layers.*.mlp.gate": "torch_a2a_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "torch_a2a_experts",
}

# _TinyMoe's fp32 bytes: the fused experts; the dense params (embedding, attention weight + bias,
# router); the router's persistent ``e_score_correction_bias`` buffer. The tied head adds none.
_EXPERT_BYTES = 4 * (4 * 16 * 8 + 4 * 8 * 8)
_DENSE_BYTES = 4 * (16 * 8 + 8 * 8 + 8 + 4 * 8)
_BUFFER_BYTES = 4 * 4
_ATTN_BYTES = 4 * (8 * 8 + 8)


class _TinyMoe(nn.Module):
    """One MoE decoder layer, a head tied to the embedding, and a persistent router buffer."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 8)
        layer = nn.Module()
        layer.self_attn = nn.Linear(8, 8)
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(8, 4, bias=False)
        layer.mlp.gate.register_buffer("e_score_correction_bias", torch.zeros(4))
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.gate_up_proj = nn.Parameter(torch.zeros(4, 16, 8))
        layer.mlp.experts.down_proj = nn.Parameter(torch.zeros(4, 8, 8))
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.all_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def get_parameter_or_buffer(self, name):
        """``PreTrainedModel``'s lookup, which the sizing goes through."""
        try:
            return self.get_parameter(name)
        except AttributeError:
            return self.get_buffer(name)

    def device_map(self):
        """Every param and persistent buffer on one GPU, as ``from_pretrained`` expands it."""
        names = [name for name, _ in self.named_parameters(remove_duplicate=False)]
        names += [name for name, _ in self.named_buffers()]
        return dict.fromkeys(names, _GPU)


class _OneBytePerElement:
    def param_element_size(self, model, name, param):
        return 1


_ALL_BYTES = _EXPERT_BYTES + _DENSE_BYTES + _BUFFER_BYTES


@pytest.mark.parametrize(
    ("dense_on_meta", "quantizer", "stock_bytes", "per_rank_bytes"),
    [
        # Experts count one EP shard; the router is replicated, so it counts in full like the rest.
        pytest.param(
            False,
            None,
            _ALL_BYTES,
            _EXPERT_BYTES // 4 + _DENSE_BYTES + _BUFFER_BYTES,
            id="dense-replicated",
        ),
        # Sharded dense: every dense .weight / .bias stays on meta during from_pretrained, while
        # persistent buffers still load onto the device.
        pytest.param(
            True, None, _ALL_BYTES, _EXPERT_BYTES // 4 + _BUFFER_BYTES, id="dense-on-meta"
        ),
        # The quantizer sizes the elements: one byte each where fp32 takes four.
        pytest.param(
            False,
            _OneBytePerElement(),
            _ALL_BYTES // 4,
            (_EXPERT_BYTES // 4 + _DENSE_BYTES + _BUFFER_BYTES) // 4,
            id="quantizer",
        ),
    ],
)
def test_transformers_reserves_only_the_per_rank_footprint(
    monkeypatch, dense_on_meta, quantizer, stock_bytes, per_rank_bytes
):
    """End to end: the one up-front allocation ``caching_allocator_warmup`` makes shrinks from the
    whole model (stock sizing, which also checks the byte counts above) to what this EP rank
    holds."""
    reserved = []

    def fake_empty(numel, *, dtype, device, **kwargs):
        reserved.append((numel * torch.tensor([], dtype=dtype).element_size(), device))

    model = _TinyMoe()
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "mem_get_info", lambda index: (80 * 1024**3, 80 * 1024**3))
        m.setattr(torch.cuda, "memory_reserved", lambda index: 0)
        m.setattr(torch.cuda, "memory_allocated", lambda index: 0)
        m.setattr(torch, "empty", fake_empty)
        modeling_utils.caching_allocator_warmup(model, model.device_map(), quantizer)
        with _ep_aware_caching_allocator_warmup(4, _A2A_PLAN, dense_on_meta=dense_on_meta):
            modeling_utils.caching_allocator_warmup(model, model.device_map(), quantizer)

    assert reserved == [(stock_bytes, _GPU), (per_rank_bytes, _GPU)]


def test_a_tp_planned_dense_param_keeps_its_tp_share_while_the_rest_stays_on_meta(monkeypatch):
    """transformers places a TP-planned param itself, never asking where the dense part goes, so it
    is on the device at its TP share even when the rest of the dense part stays on meta. The share
    is of the ep submesh it is placed over, not of the world, which holds two of them here."""
    model = _TinyMoe()
    model._tp_plan = {"model.layers.*.self_attn": "colwise"}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    with _ep_aware_caching_allocator_warmup(4, _A2A_PLAN, dense_on_meta=True):
        total = modeling_utils.get_total_byte_count(model, model.device_map())

    assert total == {_GPU: _EXPERT_BYTES // 4 + _ATTN_BYTES // 4 + _BUFFER_BYTES}


def test_an_error_while_sizing_falls_back_to_the_stock_sizing(monkeypatch):
    """The warmup only sizes an allocator reservation, so whatever the EP-aware sizing cannot
    handle gets transformers' own number instead of failing the load, with whatever else a later
    release passes the hook, and a warning: that number can reserve most of the device."""
    stock_calls = []

    def stock_sizing(model, accelerator_device_map, hf_quantizer=None, *args, **kwargs):
        stock_calls.append((model, accelerator_device_map, hf_quantizer, args, kwargs))
        return {_GPU: 123}

    class _UnsupportedQuantizer:
        def param_element_size(self, model, name, param):
            raise TypeError("unsupported param")

    monkeypatch.setattr(modeling_utils, "get_total_byte_count", stock_sizing)
    model, quantizer = _TinyMoe(), _UnsupportedQuantizer()
    device_map = model.device_map()

    with _ep_aware_caching_allocator_warmup(4, _A2A_PLAN):
        with pytest.warns(UserWarning, match="EP-aware warmup sizing failed"):
            sizing = modeling_utils.get_total_byte_count(model, device_map, quantizer, 1, flag=True)
        assert sizing == {_GPU: 123}

    assert stock_calls == [(model, device_map, quantizer, (1,), {"flag": True})]


def test_a_sizing_hook_with_other_parameters_keeps_the_stock_sizing(monkeypatch):
    """A release whose hook takes other leading parameters keeps transformers' own sizing: the
    replacement would not know how the hook is called. It says so, since that sizing may reserve
    most of the device."""

    def stock_sizing(model, device_map, quantizer=None):
        return {_GPU: 123}

    monkeypatch.setattr(modeling_utils, "get_total_byte_count", stock_sizing)

    with (
        pytest.warns(UserWarning, match="using its full-model sizing"),
        _ep_aware_caching_allocator_warmup(4, _A2A_PLAN),
    ):
        assert modeling_utils.get_total_byte_count is stock_sizing


@pytest.mark.parametrize("ep_degree", [1, 0])
def test_without_expert_parallelism_the_stock_sizing_stays(ep_degree):
    stock = modeling_utils.get_total_byte_count

    with _ep_aware_caching_allocator_warmup(ep_degree, _A2A_PLAN):
        assert modeling_utils.get_total_byte_count is stock


def test_a_transformers_without_the_sizing_hook_is_left_alone(monkeypatch):
    monkeypatch.delattr(modeling_utils, "get_total_byte_count")

    with _ep_aware_caching_allocator_warmup(4, _A2A_PLAN):
        assert not hasattr(modeling_utils, "get_total_byte_count")
