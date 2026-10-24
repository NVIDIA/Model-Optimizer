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

"""HuggingFace-coupled MoE expert-parallel (EP) helpers.

The DP=EP all-to-all expert transport (``torch_a2a``).

The all-to-all transport is plain ``torch.distributed.all_to_all_single`` (NCCL) wrapped around
transformers' ``_grouped_linear``. ``register_torch_a2a()`` registers it into
transformers' ``ExpertsInterface`` lazily -- never at import time, so importing this module has no
transformers-``ExpertsInterface`` side effect.
"""

import warnings

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single as autograd_all_to_all_single

# ---------------------------------------------------------------------------------------------------
# DP=EP all-to-all expert transport (`torch_a2a`)
#
# The throughput-correct MoE parallelism for calibration: dense+router run data-parallel (each rank its
# own batch), experts are EP-sharded, and tokens are routed to the expert-owning rank by an all-to-all
# (dispatch) / reduce (combine). The expert matmul stays on transformers' `_grouped_linear`, so each
# expert sees the tokens routed to it from every rank.
# Transport is plain `torch.distributed.all_to_all_single` (NCCL) -- works cross-node at any rank count.
# ---------------------------------------------------------------------------------------------------


# all_to_all_single addresses its flattened buffers with int32 element offsets, so a hop larger than
# this overflows them; a2a_moe chunks such transfers along the hidden dim.
_A2A_MAX_HOP_ELEMENTS = 2**31 - 1
# The combine weights and sums each token's top_k results a slice of tokens at a time, so its
# temporaries stay within this many elements however many tokens the layer routes.
_COMBINE_SLICE_ELEMENTS = 2**25
# A rank that receives more than this many times the most tokens any rank sent warns that routing has
# collapsed. A rank can receive at most world size times that, so it fires only above 4 ranks.
_A2A_SKEW_WARN_RATIO = 4


def a2a_moe(group, hidden_states, topk_idx, topk_weights, num_local_experts, expert_fn):
    """Route one MoE layer's expert stage by plain `torch.distributed.all_to_all_single` (NCCL).

    Works cross-node at any rank count, with explicit token permute/unpermute.

    Flow: expand (token, expert) pairs -> route each to dest_rank = global_expert // num_local ->
    all_to_all the tokens (+ local-expert id) -> local grouped_mm -> all_to_all back ->
    weight by the routing weights and sum each token's top_k results.

    Tokens travel and the result returns in ``hidden_states``' dtype -- the model's, so its residual
    stream keeps one dtype. The routing weights never travel: the rank that holds a token weights its
    experts' results in the weights' own dtype and sums them as transformers' grouped_mm forward does.
    """
    world = group.size()
    x = hidden_states.contiguous()
    H = x.size(-1)  # noqa: N806
    T, top_k = topk_idx.shape  # noqa: N806
    dev = x.device

    def _one_all_to_all(output, input_, out_split, in_split):
        """One all_to_all_single hop (hidden or expert output).

        The autograd-aware collective, whose backward is the reverse all-to-all, when a graph is being
        built and the input needs grad, so gradients reach the routed experts (AutoQuantize's gradient
        scores); the plain collective otherwise.
        """
        if torch.is_grad_enabled() and input_.requires_grad:
            # Deprecated in torch>=2.13 for _functional_collectives.all_to_all_single; until that
            # migration, silence its per-call FutureWarning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, message=".*all_to_all_single.*"
                )
                return autograd_all_to_all_single(output, input_, out_split, in_split, group=group)
        dist.all_to_all_single(output, input_, out_split, in_split, group=group)
        return output

    def _all_to_all(output, input_, out_split, in_split, step=None):
        """All-to-all in hidden-column slices ``step`` wide; in one hop if ``step`` is None or covers H.

        all_to_all_single addresses the flattened ``[rows, H]`` buffers with int32 element offsets, so a
        hop past 2**31 elements overflows them and hangs. Every slice has the same row splits, and
        concatenating the slices is autograd-safe.
        """
        hdim = input_.shape[-1]
        if step is None or step >= hdim:
            return _one_all_to_all(output, input_, out_split, in_split)
        if torch.is_grad_enabled() and input_.requires_grad:
            # An in-place copy_ would break the backward; cat's higher peak is fine for the few
            # tokens gradient scoring uses.
            slices = [
                _one_all_to_all(
                    output[..., c0 : c0 + step].contiguous(),
                    input_[..., c0 : c0 + step].contiguous(),
                    out_split,
                    in_split,
                )
                for c0 in range(0, hdim, step)
            ]
            return torch.cat(slices, dim=-1)
        # Forward only: write each slice into ``output`` in place; torch.cat would hold a second
        # [rows, H] copy of a buffer that is already the largest in calibration.
        for c0 in range(0, hdim, step):
            c1 = min(c0 + step, hdim)
            tmp = _one_all_to_all(
                output[..., c0:c1].contiguous(),
                input_[..., c0:c1].contiguous(),
                out_split,
                in_split,
            )
            output[..., c0:c1].copy_(tmp)
        return output

    # (token, slot) pairs in token-major order; a -1 "no selection" slot sends nothing.
    exp = topk_idx.reshape(-1).to(torch.int64)
    pair_rows = (exp >= 0).nonzero().squeeze(1)  # each sent pair's row in [T * top_k]
    exp = exp[pair_rows]
    dest = (exp // num_local_experts).to(torch.int64)  # owner rank of each pair
    local_eid = exp % num_local_experts

    order = torch.argsort(dest, stable=True)  # group pairs by dest rank
    pair_rows_s, local_eid_s = pair_rows[order], local_eid[order]
    send_hidden = x[pair_rows_s // top_k].contiguous()  # [P, H]
    in_counts = torch.bincount(dest[order], minlength=world).to(torch.int64)  # rows -> each rank

    out_counts = torch.empty_like(in_counts)  # rows <- each rank
    # The host reads the counts right below to size the dispatch: wait for the exchange, or a read that
    # races the collective's stream sizes the buffers from garbage.
    _cnt_work = dist.all_to_all_single(out_counts, in_counts, group=group, async_op=True)
    _cnt_work.wait()
    in_list, out_list = in_counts.tolist(), out_counts.tolist()
    Q = int(sum(out_list))  # noqa: N806

    P_sent = int(send_hidden.size(0))  # noqa: N806
    # One all-reduce, two maxima. Every rank must issue the same number of hops, but the rows a rank
    # moves depend on its routing, so the hops' column slices are sized from the most rows any rank
    # moves. And a rank receives about the average share, so the most tokens any rank sent, not this
    # rank's own, tells a collapsed router from batches of different sizes.
    most = torch.tensor([max(P_sent, Q), P_sent], device=dev, dtype=torch.int64)
    dist.all_reduce(most, op=dist.ReduceOp.MAX, group=group)
    most_rows, most_sent = most.tolist()
    # Warn before the O(Q x H) allocations below: a collapsed router can run out of memory there.
    if _A2A_SKEW_WARN_RATIO * max(most_sent, 1) < Q:
        warnings.warn(
            f"[a2a] rank {dist.get_rank()} received over {_A2A_SKEW_WARN_RATIO}x the most tokens any "
            "rank sent: routing has probably collapsed onto a few experts (check the router and the "
            "hidden states for NaN or blow-up), and the expert stage, which allocates O(received "
            "tokens x hidden), will likely run out of memory.",
            stacklevel=2,
        )
    # A slice as wide as that many rows allow; splitting H evenly instead rounds each slice up, which
    # can carry it past the limit.
    step = _A2A_MAX_HOP_ELEMENTS // max(most_rows, 1)
    # dispatch hidden (autograd-aware when scoring) + local-expert id (same split sizes)
    recv_hidden = torch.empty(Q, H, device=dev, dtype=x.dtype)
    recv_hidden = _all_to_all(recv_hidden, send_hidden, out_list, in_list, step)
    # Each [rows, H] buffer goes as soon as its stage is done; kept to the end, six of them were the
    # layer's peak memory.
    del send_hidden
    recv_eid = torch.empty(Q, device=dev, dtype=torch.int64)
    dist.all_to_all_single(recv_eid, local_eid_s.contiguous(), out_list, in_list, group=group)

    # group received tokens by local expert -> grouped_mm; weighted back home
    rorder = torch.argsort(recv_eid, stable=True)
    counts = torch.bincount(recv_eid[rorder], minlength=num_local_experts)
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    eo = expert_fn(recv_hidden[rorder], offsets).to(x.dtype)
    del recv_hidden
    eo_recv = torch.empty(Q, H, device=dev, dtype=x.dtype)
    eo_recv[rorder] = eo  # back to recv order
    del eo

    # combine: all_to_all back (swap counts, autograd-aware when scoring) -> sum each token's results
    send_back = torch.empty(P_sent, H, device=dev, dtype=x.dtype)
    send_back = _all_to_all(send_back, eo_recv, in_list, out_list, step)
    del eo_recv
    # Back to (token, slot) rows, a -1 slot's row zero, so each token's top_k results are adjacent.
    out_rows = send_back.new_zeros(T * top_k, H)
    out_rows[pair_rows_s] = send_back
    del send_back
    # Weight each result in its routing weight's dtype and sum each token's top_k as transformers'
    # grouped_mm forward does: in a fixed order, so the result is deterministic and the same at any EP
    # degree, which index_add_'s atomics on CUDA are not. A slice of tokens at a time bounds the
    # temporaries.
    w = topk_weights.reshape(-1)
    out = torch.empty(T, H, device=dev, dtype=x.dtype)
    slice_tokens = max(1, _COMBINE_SLICE_ELEMENTS // max(top_k * H, 1))
    # At least one slice, empty on a rank without tokens: writing it ties the result to the combine's
    # graph, so backward still runs this rank's reverse all-to-all, which its peers wait in.
    for t0 in range(0, max(T, 1), slice_tokens):
        t1 = min(t0 + slice_tokens, T)
        rows = slice(t0 * top_k, t1 * top_k)
        weighted = out_rows[rows] * w[rows, None]
        out[t0:t1] = weighted.view(t1 - t0, top_k, H).sum(1)
    return out


def torch_a2a_experts_forward(self, hidden_states, top_k_index, top_k_weights, process_group=None):
    """All-to-all expert-parallel forward for PTQ calibration (DP=EP).

    Registered as the `"torch_a2a"` experts implementation on transformers' `ExpertsInterface`.

    Dense+router run data-parallel; experts are EP-sharded; tokens are routed to the expert-owning
    rank by the NCCL all-to-all above. Local experts run through transformers' `_grouped_linear` on
    the tokens dispatched to them from every rank. `process_group` (the EP group) is injected by the
    `torch_a2a_experts` TP style's `_prepare_input_fn`; the router must emit GLOBAL ids unmasked
    (`torch_a2a_router`).

    ``self.num_experts`` is this rank's share of the experts: the expert-parallel loader loads them
    through transformers' expert-parallel sharding (the ``grouped_gemm`` / packed styles), which slices
    the experts' weights along dim 0 and sets ``num_experts`` to the per-rank count.
    """
    from transformers.integrations.moe import _grouped_linear

    if process_group is None:
        raise ValueError(
            "torch_a2a experts forward requires the EP `process_group` (use the `torch_a2a_experts` "
            "TP style, which injects it via _prepare_input_fn)."
        )
    num_local = self.num_experts
    # Per-expert biases (e.g. GPT-OSS ``gate_up_proj_bias`` / ``down_proj_bias``) are indexed by local
    # expert id below, so the plan must shard them over dim 0 like the weights (grouped_gemm): left
    # whole, every rank but the first would add other experts' biases. transformers'
    # ``_grouped_linear`` adds them after each expert's grouped GEMM.
    has_bias = getattr(self, "has_bias", False)
    if has_bias:
        for name in ("gate_up_proj_bias" if self.has_gate else "up_proj_bias", "down_proj_bias"):
            rows = getattr(self, name).shape[0]
            if rows != num_local:
                raise ValueError(
                    f"torch_a2a: {name} holds {rows} experts' biases for this rank's {num_local} "
                    "experts; the expert-parallel plan must shard the experts' biases with their "
                    "weights (grouped_gemm)."
                )

    def expert_fn(sorted_tok, offsets):
        # ``sorted_tok`` rows are grouped by local expert with cumulative ``offsets``; the per-ROW local
        # expert id is arange(num_local) repeated by each expert's token count (= diff of offsets), so gather
        # the per-expert bias to a per-row [S, out] bias for _grouped_linear's out += bias.
        row_eid = None
        if has_bias:
            counts = torch.diff(offsets, prepend=offsets.new_zeros(1)).to(torch.long)
            row_eid = torch.repeat_interleave(
                torch.arange(num_local, device=offsets.device), counts
            )
        w1 = self.gate_up_proj if self.has_gate else self.up_proj
        b1 = (
            (self.gate_up_proj_bias if self.has_gate else self.up_proj_bias)[row_eid]
            if has_bias
            else None
        )
        proj = _grouped_linear(sorted_tok, w1, offsets, bias=b1, is_transposed=self.is_transposed)
        proj = self._apply_gate(proj) if self.has_gate else self.act_fn(proj)
        b2 = self.down_proj_bias[row_eid] if has_bias else None
        return _grouped_linear(
            proj, self.down_proj, offsets, bias=b2, is_transposed=self.is_transposed
        )

    return a2a_moe(process_group, hidden_states, top_k_index, top_k_weights, num_local, expert_fn)


_TORCH_A2A_REGISTERED = False


def register_torch_a2a():
    """Register everything the torch_a2a path needs into stock transformers (idempotent).

    Done lazily from ``load_and_prepare_ep``, not at import time, so importing this module has no
    transformers side effect:

    - the ``torch_a2a`` experts forward into ``ExpertsInterface`` (``ALL_EXPERTS_FUNCTIONS``);
    - two EP parallel styles into ``ALL_PARALLEL_STYLES``:

      * ``torch_a2a_router`` -- emit global, unmasked expert ids (skip ``RouterParallel``'s
        per-EP-rank masking) so the a2a dispatch can route each (token, expert) to its owning rank;
      * ``torch_a2a_experts`` -- inject the EP ``process_group`` into the experts forward (4th arg)
        and skip the all-reduce (the a2a combine does the reduction).

    These two styles are functionally transformers' ``megamoe_router`` / ``megamoe_experts`` (added
    in 5.10.4 for DeepGEMM Mega-MoE), reimplemented here so the transport works below 5.10.4 too.
    The base classes (``RouterParallel``, ``TensorParallelLayer``) are stock wherever the EP infra
    exists (transformers >= 5.6, the framework floor enforced in ``load_and_prepare_ep``), so this
    works on any such version; ``grouped_gemm`` (expert weight sharding) is stock too.
    """
    global _TORCH_A2A_REGISTERED
    if _TORCH_A2A_REGISTERED:
        return
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES, TensorParallelLayer

    try:
        from transformers.integrations.tensor_parallel import RouterParallel
    except ImportError:
        # transformers 5.16.0 exports it only as EpRouterParallel; 5.16.1 restored the alias.
        from transformers.integrations.tensor_parallel import EpRouterParallel as RouterParallel

    ALL_EXPERTS_FUNCTIONS.register("torch_a2a", torch_a2a_experts_forward)

    class TorchA2ARouter(RouterParallel):
        """torch_a2a router style: emit global, unmasked expert ids (== stock ``megamoe_router``).

        The base ``RouterParallel._prepare_output_fn`` remaps to local ids + masks non-local; the a2a
        dispatch needs the global ids, so this just passes the router outputs through unchanged.
        """

        def _prepare_output_fn(self, mod, outputs, device_mesh):
            return outputs

        # transformers >=5.16 renamed the hook; the _prepare_* form above is never called there.
        def transform_output_post_forward(self, module, output, mesh):
            return output

    class TorchA2AExperts(TensorParallelLayer):
        """torch_a2a experts style: inject the EP ``process_group`` (4th arg) + skip the all-reduce.

        (== stock ``megamoe_experts``.) Expert weights shard via packed_colwise/rowwise +
        grouped_gemm, so ``shard_tensor`` is a no-op here. ``_prepare_input_fn``'s 4th return is what
        ``torch_a2a_experts_forward`` receives as ``process_group``.
        """

        def _prepare_input_fn(self, mod, inputs, device_mesh):
            hidden_states, top_k_index, top_k_weights = inputs[0], inputs[1], inputs[2]
            return hidden_states, top_k_index, top_k_weights, device_mesh.get_group()

        def _prepare_output_fn(self, mod, outputs, device_mesh):
            return outputs

        # transformers >= 5.16 calls the methods below and never the _prepare_* pair above; without
        # them the EP process_group is not injected. Older versions ignore them.
        def should_use_local_tensors(self, module):
            # The a2a forward runs the local expert shard through _grouped_linear itself, and
            # aten._grouped_mm rejects DTensor params next to plain activations.
            return True

        def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
            if len(args) < 3:
                return args, kwargs
            hidden_states, top_k_index, top_k_weights = args[0], args[1], args[2]
            return (hidden_states, top_k_index, top_k_weights, mesh.get_group()), kwargs

        def transform_output_post_forward(self, module, output, mesh):
            return output

        def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
            return param[...].to(device=device, dtype=dtype)

    ALL_PARALLEL_STYLES.register("torch_a2a_router", TorchA2ARouter())
    ALL_PARALLEL_STYLES.register("torch_a2a_experts", TorchA2AExperts())
    _TORCH_A2A_REGISTERED = True
