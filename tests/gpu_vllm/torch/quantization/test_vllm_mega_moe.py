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

"""Dispatch/combine fake quantization of vLLM's DeepGEMM Mega-MoE experts, with stand-ins for the
kernel and its symmetric buffer."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins import vllm_mega_moe
from modelopt.torch.utils.distributed import ParallelState

# The first fake-quant call may build ModelOpt's CUDA extension.
pytestmark = pytest.mark.timeout(300)

NUM_TOPK, MAX_TOKENS, HIDDEN, PREFIX_BYTES = 3, 8, 16, 64
DEVICE = "cuda"


class _StandInSymmBuffer:
    """DeepGEMM's symmetric buffer: inputs first, the BF16 combine rows ``[slots, T, H]`` last."""

    def __init__(self, num_shared_experts):
        self.group = SimpleNamespace(size=lambda: 2)
        self.num_experts, self.intermediate_hidden = 8, 32
        self.num_topk, self.num_max_tokens_per_rank, self.hidden = NUM_TOPK, MAX_TOKENS, HIDDEN
        self.num_shared_experts = num_shared_experts
        self.mma_type, self.activation = "fp8xfp4", "swiglu"
        num_slots = NUM_TOPK + (1 if num_shared_experts else 0)
        self.size = PREFIX_BYTES + num_slots * MAX_TOKENS * HIDDEN * 2
        self.buffer = torch.zeros(
            self.size + 32, dtype=torch.int8, device=DEVICE
        )  # reused buffers can be larger
        self.topk_idx = torch.full((MAX_TOKENS, NUM_TOPK), -1, dtype=torch.int64, device=DEVICE)


_deep_gemm_mega = ModuleType("_stand_in_deep_gemm_mega")
_deep_gemm_mega._C = SimpleNamespace(
    get_symm_buffer_size_for_mega_moe=lambda world, experts, tokens, topk, hidden, inter, mma, act, shared: (
        (PREFIX_BYTES + (topk + (1 if shared else 0)) * tokens * hidden * 2),
        None,
    )
)
_StandInSymmBuffer.__module__ = _deep_gemm_mega.__name__


class _NativeMegaMoEExperts(torch.nn.Module):
    """Writes per-slot rows like the kernel's L2 epilogue and sums them like its combine."""

    def forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp=None):
        self.dispatched = hidden_states
        num_tokens = hidden_states.shape[0]
        slots = vllm_mega_moe._combine_slots(self.symm_buffer, num_tokens)
        self.symm_buffer.topk_idx[:num_tokens] = topk_ids
        for k in range(slots.shape[0]):
            slots[k].copy_(hidden_states * (k + 1) + self.corruption)
        valid = (topk_ids >= 0).T.unsqueeze(-1)
        return vllm_mega_moe._ordered_sum(slots, valid, NUM_TOPK) + self.corruption

    def get_symm_buffer(self):
        return self.symm_buffer


class _TestQuantMegaMoEExperts(vllm_mega_moe._QuantVLLMMegaMoEExperts, _NativeMegaMoEExperts):
    pass


@pytest.fixture(autouse=True)
def _stand_in_deep_gemm(monkeypatch):
    monkeypatch.setitem(sys.modules, _deep_gemm_mega.__name__, _deep_gemm_mega)
    monkeypatch.setattr(vllm_mega_moe, "create_parallel_state", lambda: ParallelState(None))
    monkeypatch.setattr(vllm_mega_moe, "_checked_buffers", set())


def _mega_moe(num_shared_experts=0, constant_amax=1.0, corruption=0.0):
    moe = object.__new__(_TestQuantMegaMoEExperts)
    torch.nn.Module.__init__(moe)
    moe.prefix = "model.layers.0.ffn.experts"
    moe.symm_buffer = _StandInSymmBuffer(num_shared_experts)
    moe.has_fused_shared_experts = bool(num_shared_experts)
    moe.corruption = corruption
    moe._setup()
    for quantizer in (moe.dispatch_quantizer, moe.combine_quantizer):
        quantizer.set_from_attribute_config({"enable": True, "constant_amax": constant_amax})
    return moe.to(DEVICE)  # like vLLM, which loads the model on the GPU


def _forward(moe, num_tokens=5):
    hidden_states = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    topk_ids = torch.randint(0, 8, (num_tokens, NUM_TOPK), device=DEVICE)
    topk_ids[0, 1] = topk_ids[3, 0] = -1  # unused slots
    topk_ids[4] = -1  # a padding token
    return hidden_states, topk_ids, moe(hidden_states, None, topk_ids, activation_clamp=None)


@pytest.mark.parametrize("num_shared_experts", [0, 1], ids=("routed_only", "fused_shared"))
def test_mega_moe_quantizes_dispatch_and_combine(num_shared_experts):
    moe = _mega_moe(num_shared_experts)
    if num_shared_experts:
        moe.dispatch_quantizer.disable()  # the fused shared expert reads the dispatched tokens
    hidden_states, topk_ids, output = _forward(moe)

    dispatched = moe.dispatch_quantizer(hidden_states)
    assert torch.equal(moe.dispatched, dispatched)
    valid = (topk_ids >= 0).T.unsqueeze(-1)
    total = torch.zeros(hidden_states.shape, device=DEVICE)
    for k in range(NUM_TOPK):  # each communicated row is quantized; the shared one is not
        total += moe.combine_quantizer(torch.where(valid[k], dispatched * (k + 1), 0)).float()
    if num_shared_experts:
        total += (dispatched * (NUM_TOPK + 1)).float()
    assert torch.equal(output, total.to(torch.bfloat16))
    assert not torch.equal(output, vllm_mega_moe._ordered_sum(*_slots_and_valid(moe, topk_ids)))
    if not num_shared_experts:
        assert not output[4].any()  # padding tokens get no routed rows


def _slots_and_valid(moe, topk_ids):
    slots = vllm_mega_moe._combine_slots(moe.symm_buffer, topk_ids.shape[0])
    return slots, (topk_ids >= 0).T.unsqueeze(-1), NUM_TOPK


@pytest.mark.parametrize(
    ("num_shared_experts", "constant_amax", "corruption", "match"),
    [
        (1, 1.0, 0.0, "fused shared expert reads the dispatched tokens"),
        (0, None, 0.0, "needs a calibrated .* or constant"),
        (0, 1.0, 1.0, "do not reproduce the kernel output"),
    ],
    ids=("fused_shared_dispatch", "per_call_combine_scale", "changed_buffer_layout"),
)
def test_mega_moe_rejects_unsupported_setups(num_shared_experts, constant_amax, corruption, match):
    moe = _mega_moe(num_shared_experts, constant_amax, corruption)
    if constant_amax is None:
        moe.dispatch_quantizer.disable()
    with pytest.raises(RuntimeError, match=match):
        _forward(moe)


def test_mega_moe_experts_are_registered():
    for cls in (vllm_mega_moe.DeepseekV4MegaMoEExperts, vllm_mega_moe.KimiK3MegaMoEExperts):
        if cls is None:
            pytest.skip("this vLLM has no DeepGEMM Mega-MoE experts")
        assert cls in vllm_mega_moe.QuantModuleRegistry
    assert isinstance(_mega_moe().combine_quantizer, TensorQuantizer)


def test_mega_moe_layout_check_waits_for_routed_rows():
    moe = _mega_moe(corruption=1.0)  # the rows no longer reproduce the kernel output
    moe.dispatch_quantizer.disable()
    hidden_states = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    padding = torch.full((4, NUM_TOPK), -1, device=DEVICE)  # like vLLM's warmup batches
    moe(hidden_states, None, padding, activation_clamp=None)  # nothing to compare yet
    assert not vllm_mega_moe._checked_buffers
    with pytest.raises(RuntimeError, match="do not reproduce the kernel output"):
        _forward(moe)


@pytest.mark.parametrize("corruption", [0.0, 1.0], ids=("same_layout", "changed_buffer_layout"))
def test_mega_moe_graph_checks_layout_on_replay(monkeypatch, corruption):
    """A CUDA graph captured after padding-only warmup checks the rows on every replay."""
    reproduced = torch.full((), bool(corruption), device=DEVICE)  # wrong until the replay checks
    # Record the device-side check instead of letting it kill the CUDA context.
    monkeypatch.setattr(
        torch, "_assert_async", lambda condition, message: reproduced.copy_(condition)
    )
    moe = _mega_moe(corruption=corruption)
    moe.dispatch_quantizer.disable()
    hidden_states = torch.randn(5, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    topk_ids = torch.full((5, NUM_TOPK), -1, device=DEVICE)  # like vLLM's warmup batches
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        moe(hidden_states, None, topk_ids, activation_clamp=None)  # eager warmup checks nothing
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = moe(hidden_states, None, topk_ids, activation_clamp=None)
    assert not vllm_mega_moe._checked_buffers

    topk_ids.copy_(torch.randint(0, 8, topk_ids.shape, device=DEVICE))
    graph.replay()
    assert reproduced.item() == (not corruption)
    if not corruption:
        slots, valid, num_topk = _slots_and_valid(moe, topk_ids)
        assert valid.all()
        assert torch.equal(
            output, vllm_mega_moe._ordered_sum(slots, valid, num_topk, moe.combine_quantizer)
        )
