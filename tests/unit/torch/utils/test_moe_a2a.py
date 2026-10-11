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

import warnings

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job

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


def _a2a_moe_fwd_bwd_job(rank, size):
    device = torch.device("cpu")
    assert E % size == 0, "E must be divisible by world size"
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

    out_a = a2a_moe(dist.group.WORLD, x_r, ti, tw, E, n_local, expert_fn)  # [T, H]
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
    dist.all_reduce(gate_grad_sum, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(gate_grad_sum.float(), gate_ref.grad.float(), atol=atol, rtol=rtol)

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_a2a_moe_forward_backward(world_size, skip_on_windows):
    """a2a_moe forward + backward (hidden, expert, router grads) match a dense reference."""
    spawn_multiprocess_job(world_size, _a2a_moe_fwd_bwd_job, backend="gloo")


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
    dist.destroy_process_group()


def test_a2a_moe_chunks_hops_past_the_element_limit(skip_on_windows):
    spawn_multiprocess_job(2, _chunked_hops_job, backend="gloo")


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
    dist.destroy_process_group()


def test_a2a_moe_warns_when_a_rank_receives_far_more_than_it_sent(skip_on_windows):
    spawn_multiprocess_job(2, _skewed_routing_job, backend="gloo")
