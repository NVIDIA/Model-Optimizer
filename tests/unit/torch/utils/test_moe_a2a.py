# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Forward + backward correctness of the ``torch_a2a`` expert-parallel MoE transport (``a2a_moe``).

Each rank owns ``E // world`` experts; ``a2a_moe`` routes every ``(token, expert)`` pair to the
owning rank by NCCL/gloo all-to-all, runs the local experts, and combines back. transformers' fused
experts, split over the ranks, run through it, and we compare the forward AND the gradients (w.r.t.
the input hidden states, each rank's experts, and the **routing weights**) against the whole module
on one process.

The routing-weight gradient check is the regression guard for the autograd-aware routing-weight hop:
if the routing weights were transported with a plain (non-autograd) collective, they would receive
no gradient and the comparison would fail.
"""

import copy
import warnings

import pytest
import torch
import torch.distributed as dist

import modelopt.torch.utils.moe as moe
from modelopt.torch.utils.moe import a2a_moe

# Small, fully-deterministic problem. E must be divisible by every world size under test.
E, TOP_K, H, T = 4, 2, 8, 6
# The experts' dtype in the jobs below: bf16, as in the models EP calibrates. a2a_moe computes in the
# dtype it is given, and the whole-module reference runs in it too, so tolerances stay tight.
COMPUTE_DTYPE = torch.bfloat16


def _identity_experts(sorted_tok, offsets):
    return sorted_tok


def _keeps_dtype_job(rank, size, dtype, tol):
    """Every rank its own one-rank group: what comes back is the dense result, in the input's dtype.

    The model's residual stream adds it to the hidden states, so any other dtype breaks the next
    layer (fp16 + bf16 promotes to fp32) or silently rounds the model (fp32 through bf16).
    """
    group, _ = dist.new_subgroups(1)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(T, H, generator=gen).to(dtype)
    w = (torch.randn(E, H, H, generator=gen) / H**0.5).to(dtype)
    ti = torch.randint(0, E, (T, TOP_K), generator=gen)
    tw = torch.rand(T, TOP_K, generator=gen).to(dtype)

    def experts(sorted_tok, offsets):
        ends = offsets.tolist()
        return torch.cat(
            [sorted_tok[s:e] @ w[i] for i, (s, e) in enumerate(zip([0, *ends[:-1]], ends))]
        )

    with torch.no_grad():
        out = a2a_moe(group, x, ti, tw, E, experts)
    want = sum(
        tw[:, k, None].float() * torch.einsum("th,thj->tj", x.float(), w[ti[:, k]].float())
        for k in range(TOP_K)
    )
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), want, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    ("dtype", "tol"), [(torch.float32, 1e-5), (torch.float16, 1e-2)], ids=["fp32", "fp16"]
)
def test_a2a_moe_keeps_the_dtype_it_is_given(cpu_dist_workers, dtype, tol):
    cpu_dist_workers.run(_keeps_dtype_job, dtype=dtype, tol=tol)


def _chunked_hops_job(rank, size):
    """Shrinking the per-hop element limit splits every 2-D hop along the hidden dim into slices
    within it -- the guard against int32 offset overflow, at a size a unit test can afford --
    without changing the result."""
    gen = torch.Generator().manual_seed(rank)
    x = torch.randn(T, H, generator=gen)
    # Every rank sends and receives T * TOP_K = 12 rows. A limit of 34 holds 2 of their H = 8
    # columns; splitting H evenly into ceil(12 * 8 / 34) = 3 slices would make them 3 wide (36).
    ti = (torch.arange(T * TOP_K) % E).reshape(T, TOP_K)
    tw = torch.rand(T, TOP_K, generator=gen)
    limit = 34
    hops = []  # the elements each 2-D hop moves
    real = dist.all_to_all_single

    def recording(output, input_, *args, **kwargs):
        if input_.dim() == 2:
            hops.append(max(output.numel(), input_.numel()))
        return real(output, input_, *args, **kwargs)

    dist.all_to_all_single = recording
    try:
        with torch.no_grad():
            whole = a2a_moe(dist.group.WORLD, x, ti, tw, E // size, _identity_experts)
            whole_hops = hops.copy()
            hops.clear()
            moe._A2A_MAX_HOP_ELEMENTS = limit
            chunked = a2a_moe(dist.group.WORLD, x, ti, tw, E // size, _identity_experts)
    finally:
        dist.all_to_all_single = real
        moe._A2A_MAX_HOP_ELEMENTS = 2**31 - 1
    assert max(hops) <= limit < max(whole_hops)
    torch.testing.assert_close(chunked, whole, atol=0, rtol=0)


def test_a2a_moe_chunks_hops_past_the_element_limit(cpu_dist_workers):
    cpu_dist_workers.run(_chunked_hops_job)


def _skewed_routing_job(rank, size):
    """Rank 1 sends four times rank 0's tokens, all to an expert rank 0 owns: rank 0 receives five
    times what it sent and warns before allocating for it; rank 1, which receives nothing, does not."""
    tokens = T if rank == 0 else 4 * T
    x = torch.randn(tokens, H)
    ti = torch.zeros(tokens, 1, dtype=torch.int64)
    tw = torch.ones(tokens, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            a2a_moe(dist.group.WORLD, x, ti, tw, E // size, _identity_experts)
    imbalance = [w for w in caught if "imbalance" in str(w.message)]
    assert len(imbalance) == (1 if rank == 0 else 0), [str(w.message) for w in caught]


def test_a2a_moe_warns_when_a_rank_receives_far_more_than_it_sent(cpu_dist_workers):
    cpu_dist_workers.run(_skewed_routing_job)


def _split_qwen3_moe_experts(rank, size):
    """transformers' fused Qwen3-MoE experts, whole and this rank's contiguous share of them."""
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    config = Qwen3MoeConfig(
        hidden_size=H,
        moe_intermediate_size=2 * H,
        intermediate_size=4 * H,
        num_experts=E,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )
    torch.manual_seed(0)
    whole = Qwen3MoeExperts(config).to(COMPUTE_DTYPE)
    with torch.no_grad():
        whole.gate_up_proj.normal_(std=0.3)
        whole.down_proj.normal_(std=0.3)
    local_config = copy.deepcopy(config)
    local_config.num_experts = E // size
    local = Qwen3MoeExperts(local_config).to(COMPUTE_DTYPE)
    owned = slice(rank * (E // size), (rank + 1) * (E // size))
    with torch.no_grad():
        local.gate_up_proj.copy_(whole.gate_up_proj[owned])
        local.down_proj.copy_(whole.down_proj[owned])
    return whole, local, owned


def _routed_tokens(size):
    """Every rank's tokens and their top-k routing, identical on all ranks."""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(size * T, H, generator=gen).to(COMPUTE_DTYPE)
    weights, index = torch.randn(size * T, E, generator=gen).softmax(-1).topk(TOP_K, dim=-1)
    return x, weights.to(COMPUTE_DTYPE), index


def _assert_experts_backward_matches(rank, whole, local, owned, size):
    """``local`` runs this rank's tokens through the torch_a2a forward; ``whole`` runs all tokens on one
    process. The forward, and the gradients of this rank's hidden states and routing weights and of
    this rank's experts, must agree."""
    from transformers.integrations.moe import grouped_mm_experts_forward

    x, weights, index = _routed_tokens(size)
    tok = slice(rank * T, (rank + 1) * T)

    whole.config._experts_implementation = "grouped_mm"
    x_ref = x.clone().requires_grad_(True)
    w_ref = weights.clone().requires_grad_(True)
    out_ref = whole(x_ref, index, w_ref)
    out_ref.float().pow(2).sum().backward()

    local.config._experts_implementation = "torch_a2a"
    x_ep = x[tok].clone().requires_grad_(True)
    w_ep = weights[tok].clone().requires_grad_(True)
    # The torch_a2a_experts parallel style passes the EP group as a fourth argument.
    out_ep = local(x_ep, index[tok], w_ep, dist.group.WORLD)
    out_ep.float().pow(2).sum().backward()

    assert grouped_mm_experts_forward is not None  # the reference took the grouped_mm path
    close = {"atol": 3e-2, "rtol": 3e-2}
    torch.testing.assert_close(out_ep.float(), out_ref[tok].detach().float(), **close)
    torch.testing.assert_close(x_ep.grad.float(), x_ref.grad[tok].float(), **close)
    torch.testing.assert_close(w_ep.grad.float(), w_ref.grad[tok].float(), **close)
    for name in ("gate_up_proj", "down_proj"):
        # Each rank's experts collect the gradient of every rank's tokens routed to them.
        local_grad = getattr(local, name).grad
        assert local_grad is not None and local_grad.abs().max() > 0, name
        torch.testing.assert_close(
            local_grad.float(), getattr(whole, name).grad[owned].float(), **close
        )


def _real_experts_backward_job(rank, size):
    moe.register_torch_a2a()
    whole, local, owned = _split_qwen3_moe_experts(rank, size)
    _assert_experts_backward_matches(rank, whole, local, owned, size)


def test_fused_experts_backward_through_torch_a2a_matches_the_whole_module(cpu_dist_workers):
    """transformers' fused experts, split over two ranks, train through the torch_a2a forward exactly
    as the whole module does on one process: same output, and same gradients for the hidden states,
    the routing weights and each rank's experts."""
    transformers = pytest.importorskip("transformers")
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    cpu_dist_workers.run(_real_experts_backward_job)
