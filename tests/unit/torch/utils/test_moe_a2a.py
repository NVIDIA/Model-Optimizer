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
owning rank by NCCL/gloo all-to-all, runs the local experts, and combines back. We compare its
forward AND its gradients (w.r.t. the input hidden states, the per-expert weights, and the
**router/gate weights**) against a single-process DENSE reference that routes every token to its
top-k experts directly.

The router/gate-gradient check is the regression guard for the autograd-aware routing-weight hop:
if the routing weights were transported with a plain (non-autograd) collective, the gate would
receive ~zero gradient and the ``assert ... > 0`` below (and the all-reduced comparison) would fail.
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
COMPUTE_DTYPE = (
    torch.bfloat16
)  # a2a_moe computes in bf16 internally; mirror it so tolerances stay tight.


def _dense_moe(x, gate_w, w, top_k):
    """Reference dense MoE: out[t] = sum_k softmax(x@gate)[t, e_k] * (x[t] @ w[e_k]).

    Matmuls run in ``COMPUTE_DTYPE`` (like ``a2a_moe``); routing/softmax stay fp32 for a stable top-k.
    Fully differentiable w.r.t. ``x`` (hidden), ``w`` (experts) and ``gate_w`` (router).
    """
    logits = x.float() @ gate_w.float()  # [N, E]
    probs = logits.softmax(dim=-1)
    tw, ti = probs.topk(top_k, dim=-1)  # [N, top_k]
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    xc = x.to(COMPUTE_DTYPE)
    for k in range(top_k):
        w_k = w[ti[:, k]].to(COMPUTE_DTYPE)  # [N, H, H]
        contrib = torch.einsum("nh,nhj->nj", xc, w_k).float()
        out = out + contrib * tw[:, k : k + 1]
    return out, tw, ti


def _router(x, gate_w, top_k):
    """The same routing a2a_moe consumes: global expert ids + (grad-carrying) top-k weights."""
    probs = (x.float() @ gate_w.float()).softmax(dim=-1)
    tw, ti = probs.topk(top_k, dim=-1)
    return ti.to(torch.int64), tw


def _a2a_moe_fwd_bwd_job(rank, size, ep_size):
    # Every ``ep_size`` consecutive ranks form one expert-parallel group.
    group, _ = dist.new_subgroups(ep_size)
    rank, size = dist.get_rank(group), ep_size
    device = torch.device("cpu")
    assert E % size == 0, "E must be divisible by the expert-parallel size"
    n_local = E // size
    n_total = size * T

    gen = torch.Generator(device=device).manual_seed(0)
    x_full = torch.randn(n_total, H, generator=gen, device=device)
    gate0 = torch.randn(H, E, generator=gen, device=device) * 0.1
    w0 = torch.randn(E, H, H, generator=gen, device=device) / (H**0.5)

    # ---------- dense reference (single-process, full batch, all experts) ----------
    x_ref = x_full.clone().requires_grad_(True)
    gate_ref = gate0.clone().requires_grad_(True)
    w_ref = w0.clone().requires_grad_(True)
    out_ref, _, _ = _dense_moe(x_ref, gate_ref, w_ref, TOP_K)
    out_ref.pow(2).sum().backward()

    # ---------- a2a (this rank owns experts [rank*n_local : (rank+1)*n_local]) ----------
    x_a = x_full.clone().requires_grad_(True)
    gate_a = gate0.clone().requires_grad_(True)
    w_a = w0.clone().requires_grad_(True)

    tok = slice(rank * T, (rank + 1) * T)
    x_r = x_a[tok]  # this rank's tokens (data-parallel)
    ti, tw = _router(x_r, gate_a, TOP_K)  # global ids + grad-carrying weights
    w_local = w_a[rank * n_local : (rank + 1) * n_local]  # [n_local, H, H]

    def expert_fn(sorted_tok, offsets):
        # sorted_tok [Q, H] grouped by local expert; offsets = cumsum of per-local-expert counts.
        segs, start = [], 0
        for i in range(n_local):
            end = int(offsets[i])
            seg = sorted_tok[start:end]
            segs.append(seg @ w_local[i].to(seg.dtype))
            start = end
        return torch.cat(segs, dim=0) if segs else sorted_tok.new_zeros((0, H))

    out_a = a2a_moe(group, x_r, ti, tw, E, n_local, expert_fn)  # [T, H]
    out_a.pow(2).sum().backward()  # collective backward across ranks (a2a reverse all-to-all)

    atol, rtol = 3e-2, 3e-2
    # (1) forward matches the dense reference on this rank's tokens.
    torch.testing.assert_close(out_a.float(), out_ref[tok].detach().float(), atol=atol, rtol=rtol)
    # (2) hidden-state grad (this rank's tokens) — routes back through the a2a dispatch.
    torch.testing.assert_close(x_a.grad[tok].float(), x_ref.grad[tok].float(), atol=atol, rtol=rtol)
    # (3) local expert-weight grad = grad from ALL ranks' tokens routed here (a2a already gathered them).
    torch.testing.assert_close(
        w_a.grad[rank * n_local : (rank + 1) * n_local].float(),
        w_ref.grad[rank * n_local : (rank + 1) * n_local].float(),
        atol=atol,
        rtol=rtol,
    )
    # (4) router/gate grad: THE regression guard for the autograd routing-weight hop. Each rank's grad
    #     comes from its own tokens; summed across ranks it must equal the full dense reference. Without
    #     the autograd hop the gate grad is ~0 and both checks below fail.
    assert gate_a.grad.abs().max() > 1e-3, (
        "router/gate grad is ~0 -> routing-weight all-to-all is not autograd-aware (recv_w detached)."
    )
    gate_grad_sum = gate_a.grad.clone()
    dist.all_reduce(gate_grad_sum, op=dist.ReduceOp.SUM, group=group)
    torch.testing.assert_close(gate_grad_sum.float(), gate_ref.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_a2a_moe_forward_backward(cpu_dist_workers, ep_size):
    """a2a_moe forward + backward (hidden, expert, router grads) match a dense reference."""
    cpu_dist_workers.run(_a2a_moe_fwd_bwd_job, ep_size=ep_size)


def _identity_experts(sorted_tok, offsets):
    return sorted_tok


def _chunked_hops_job(rank, size):
    """Shrinking the per-hop element limit splits every 2-D hop along the hidden dim -- the guard
    against int32 offset overflow, at a size a unit test can afford -- without changing the result."""
    gen = torch.Generator().manual_seed(rank)
    x = torch.randn(T, H, generator=gen)
    ti = torch.randint(0, E, (T, TOP_K), generator=gen)
    tw = torch.rand(T, TOP_K, generator=gen)
    hops = []
    real = dist.all_to_all_single

    def counting(*args, **kwargs):
        hops.append(1)
        return real(*args, **kwargs)

    dist.all_to_all_single = counting
    try:
        with torch.no_grad():
            whole = a2a_moe(dist.group.WORLD, x, ti, tw, E, E // size, _identity_experts)
            n_whole = len(hops)
            hops.clear()
            moe._A2A_MAX_HOP_ELEMENTS = 7
            chunked = a2a_moe(dist.group.WORLD, x, ti, tw, E, E // size, _identity_experts)
            n_chunked = len(hops)
    finally:
        dist.all_to_all_single = real
        moe._A2A_MAX_HOP_ELEMENTS = 2**31 - 1
    assert n_chunked > n_whole
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
            a2a_moe(dist.group.WORLD, x, ti, tw, E, E // size, _identity_experts)
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
