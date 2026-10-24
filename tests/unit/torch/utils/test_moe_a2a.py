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

"""Forward + backward correctness of the ``torch_a2a`` expert-parallel MoE transport (``a2a_moe``).

Each rank owns ``E // world`` experts; ``a2a_moe`` routes every ``(token, expert)`` pair to the
owning rank by NCCL/gloo all-to-all, runs the local experts, and combines back. transformers' fused
experts, split over the ranks, run through it, and we compare the forward AND the gradients (w.r.t.
the input hidden states, each rank's experts, and the **routing weights**) against the whole module
on one process.

The routing-weight gradient check guards that the routing weights keep their gradient: they never
travel, and weight their tokens' expert results on the rank that holds them.
"""

import copy
import warnings
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version

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

    group, _ = dist.new_subgroups(1)
    try:
        with torch.no_grad():
            out = a2a_moe(group, x, ti, tw, E, experts)
    finally:
        dist.destroy_process_group(group)
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


def _bounded_combine_job(rank, size):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class LargestFp32(TorchDispatchMode):
        """Records the largest fp32 tensor any op produces."""

        numel = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_leaves(out):
                if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                    self.numel = max(self.numel, t.numel())
            return out

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(T, H, generator=gen).to(COMPUTE_DTYPE)
    ti = torch.randint(0, E, (T, TOP_K), generator=gen)
    tw = torch.rand(T, TOP_K, generator=gen)
    default = moe._COMBINE_SLICE_ELEMENTS
    group, _ = dist.new_subgroups(1)
    try:
        with torch.no_grad():
            whole = a2a_moe(group, x, ti, tw, E, _identity_experts)
            moe._COMBINE_SLICE_ELEMENTS = TOP_K * H  # one token a slice
            with LargestFp32() as largest:
                sliced = a2a_moe(group, x, ti, tw, E, _identity_experts)
    finally:
        moe._COMBINE_SLICE_ELEMENTS = default
        dist.destroy_process_group(group)
    # The fp32 routing weights make the weighted results fp32, one token's at a time.
    assert largest.numel <= TOP_K * H
    torch.testing.assert_close(sliced, whole, rtol=0, atol=0)


def test_the_combine_works_a_slice_of_tokens_at_a_time(cpu_dist_workers):
    """A whole fp32 copy of the (token, expert) results reaches GiBs at calibration batch sizes; the
    combine weights and sums them a slice of tokens at a time, with the same result."""
    cpu_dist_workers.run(_bounded_combine_job)


def test_a2a_moe_chunks_hops_past_the_element_limit(cpu_dist_workers):
    cpu_dist_workers.run(_chunked_hops_job)


def _routing_skew_job(rank, size, routing):
    if routing == "collapsed":
        # Rank 1 sends four times rank 0's tokens, all to an expert rank 0 owns: rank 0 receives
        # more than any rank sent, and warns before allocating for it.
        tokens = T if rank == 0 else 4 * T
        ti = torch.zeros(tokens, 1, dtype=torch.int64)
    else:
        # Rank 1's batch is ten times rank 0's, routed evenly: rank 0 receives over five times what
        # it sent, but less than rank 1 sent.
        tokens = T if rank == 0 else 10 * T
        ti = (torch.arange(tokens) % 2 * (E // size)).unsqueeze(1)
    x = torch.randn(tokens, H)
    tw = torch.ones(tokens, 1)
    # Two ranks never receive twice the most any rank sent; a ratio of 1 lets them warn at all.
    ratio, moe._A2A_SKEW_WARN_RATIO = moe._A2A_SKEW_WARN_RATIO, 1
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                a2a_moe(dist.group.WORLD, x, ti, tw, E // size, _identity_experts)
    finally:
        moe._A2A_SKEW_WARN_RATIO = ratio
    collapsed = [w for w in caught if "collapsed" in str(w.message)]
    warns = routing == "collapsed" and rank == 0
    assert len(collapsed) == int(warns), [str(w.message) for w in caught]


@pytest.mark.parametrize("routing", ["collapsed", "unequal-batches"])
def test_a2a_moe_warns_when_a_rank_receives_more_than_any_rank_sent(cpu_dist_workers, routing):
    cpu_dist_workers.run(_routing_skew_job, routing=routing)


def test_the_router_style_registers_under_transformers_5_16_0s_name(monkeypatch):
    """transformers 5.16.0 exports the router style only as ``EpRouterParallel``; 5.16.1 restored
    ``RouterParallel`` as an alias. The registration takes either."""
    tp = pytest.importorskip("transformers.integrations.tensor_parallel")
    if not hasattr(tp, "EpRouterParallel"):
        pytest.skip("this transformers has no EpRouterParallel")
    # 5.16.0 itself has no RouterParallel to delete.
    monkeypatch.delattr(tp, "RouterParallel", raising=False)
    monkeypatch.setattr(moe, "_TORCH_A2A_REGISTERED", False)

    moe.register_torch_a2a()

    assert isinstance(tp.ALL_PARALLEL_STYLES["torch_a2a_router"], tp.EpRouterParallel)


def test_experts_whose_biases_the_plan_left_whole_are_refused():
    """A plan that shards GPT-OSS's expert weights but not their biases leaves every rank all the
    experts' biases, which the forward indexes by local expert id; it refuses them instead."""
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    config = GptOssConfig(
        hidden_size=H,
        intermediate_size=2 * H,
        num_local_experts=E,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    whole = GptOssExperts(config)
    local_config = copy.deepcopy(config)
    local_config.num_local_experts = E // 2
    local = GptOssExperts(local_config)
    local.gate_up_proj_bias = whole.gate_up_proj_bias
    x, weights, index = torch.randn(T, H), torch.rand(T, TOP_K), torch.randint(0, E, (T, TOP_K))
    # The refusal comes before any collective, so it needs no process group to reach.
    with pytest.raises(ValueError):
        moe.torch_a2a_experts_forward(local, x, index, weights, process_group=object())


def _no_tokens_for_rank1_job(rank, size):
    moe.register_torch_a2a()
    whole, local, owned = _split_qwen3_moe_experts(rank, size)
    _assert_experts_backward_matches(rank, whole, local, owned, size, to_rank0=True)


def test_experts_that_receive_no_tokens_run_on_an_empty_batch(cpu_dist_workers):
    """Every token routed to rank 0's experts: rank 1 runs transformers' fused experts on an empty
    batch, and the forward and gradients still match the whole module's on every rank."""
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    cpu_dist_workers.run(_no_tokens_for_rank1_job)


def _no_routes_from_rank1_job(rank, size, rank1_tokens):
    tokens = 2 if rank == 0 else rank1_tokens
    torch.manual_seed(0)
    x = torch.randn(tokens, H).requires_grad_(True)
    routes = torch.tensor([[0, 1], [0, 1]]) if rank == 0 else torch.full((tokens, 2), -1)
    # A rank left waiting in the backward's all-to-all gives up in seconds on this group.
    group = dist.new_group(timeout=timedelta(seconds=10))
    try:
        out = a2a_moe(group, x, routes, torch.ones(tokens, 2), 1, _identity_experts)
        (out + x).square().sum().backward()
    finally:
        dist.destroy_process_group(group)
    # Rank 0's tokens come back from both ranks' identity experts, out = 2x: d/dx (3x)^2 = 18x.
    # Rank 1 routes nothing, out = 0: d/dx x^2 = 2x.
    torch.testing.assert_close(x.grad, (18 if rank == 0 else 2) * x.detach())


@pytest.mark.parametrize("rank1_tokens", [2, 0], ids=["unrouted-slots", "no-tokens"])
def test_a_rank_that_routes_no_tokens_still_joins_the_backward(cpu_dist_workers, rank1_tokens):
    """Rank 1 sends nothing, its routing slots all -1 or no tokens at all, but it serves rank 0's
    tokens. Its backward must still run the reverse all-to-all, or rank 0 waits in its own."""
    cpu_dist_workers.run(_no_routes_from_rank1_job, rank1_tokens=rank1_tokens)


def _split_qwen3_moe_experts(rank, size):
    """transformers' fused Qwen3-MoE experts, whole and this rank's contiguous share of them."""
    # transformers is optional: CI's transformers-free unit lane collects this file.
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


def _routed_tokens(size, to_rank0=False, weights_dtype=COMPUTE_DTYPE):
    """Every rank's tokens and their top-k routing, identical on all ranks; ``to_rank0`` routes every
    token to the experts rank 0 owns."""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(size * T, H, generator=gen).to(COMPUTE_DTYPE)
    weights, index = torch.randn(size * T, E, generator=gen).softmax(-1).topk(TOP_K, dim=-1)
    if to_rank0:
        index = torch.arange(TOP_K).expand(size * T, TOP_K) % (E // size)
    return x, weights.to(weights_dtype), index


def _assert_experts_backward_matches(rank, whole, local, owned, size, to_rank0=False):
    """``local`` runs this rank's tokens through the torch_a2a forward; ``whole`` runs all tokens on one
    process. The forward, and the gradients of this rank's hidden states and routing weights and of
    this rank's experts, must agree."""
    x, weights, index = _routed_tokens(size, to_rank0)
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

    close = {"atol": 3e-2, "rtol": 3e-2}
    torch.testing.assert_close(out_ep.float(), out_ref[tok].detach().float(), **close)
    torch.testing.assert_close(x_ep.grad.float(), x_ref.grad[tok].float(), **close)
    torch.testing.assert_close(w_ep.grad.float(), w_ref.grad[tok].float(), **close)
    for name in ("gate_up_proj", "down_proj"):
        # Each rank's experts collect the gradient of every rank's tokens routed to them, and only
        # that: experts no token reached get none.
        param = getattr(local, name)
        local_grad = param.grad if param.grad is not None else torch.zeros_like(param)
        assert (local_grad.abs().max() > 0) == (not to_rank0 or rank == 0), name
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
    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    cpu_dist_workers.run(_real_experts_backward_job)


def _fp32_routing_weights_job(rank, size):
    moe.register_torch_a2a()
    whole, local, _ = _split_qwen3_moe_experts(0, 1)
    x, weights, index = _routed_tokens(1, weights_dtype=torch.float32)
    whole.config._experts_implementation = "grouped_mm"
    local.config._experts_implementation = "torch_a2a"
    group, _ = dist.new_subgroups(1)
    try:
        with torch.no_grad():
            want = whole(x, index, weights)
            got = local(x, index, weights, group)
    finally:
        dist.destroy_process_group(group)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fp32_routing_weights_give_grouped_mms_result_at_ep1(cpu_dist_workers):
    """A router may return fp32 weights for bf16 experts, as Mixtral's does. grouped_mm weights the
    experts' results in fp32 then; at EP=1 the torch_a2a forward must give its result bit for bit."""
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    cpu_dist_workers.run(_fp32_routing_weights_job)
