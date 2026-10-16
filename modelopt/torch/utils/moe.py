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

"""HuggingFace-coupled MoE expert-parallel (EP) helpers.

Finding a model's routed experts, choosing the EP degree they are sharded over, the DP=EP
all-to-all expert transport (``torch_a2a``). The expert-parallel model loader is
:func:`~modelopt.torch.utils.plugins.model_load_utils.load_and_prepare_ep`.

The all-to-all transport is plain ``torch.distributed.all_to_all_single`` (NCCL) wrapped around
transformers' quantizable ``_grouped_linear``. ``register_torch_a2a()`` registers
it into transformers' ``ExpertsInterface`` (lazily, from ``load_and_prepare_ep``'s a2a path -- never at
import time, so importing this module has no transformers-``ExpertsInterface`` side effect).
"""

import warnings

import torch
import torch.distributed as dist
import torch.nn as nn

from .distributed import distribution_plan


def require_transformers(min_version: str, feature: str) -> None:
    """Fail loudly if the installed transformers is older than ``min_version`` (the floor for ``feature``).

    Callers enforce the version where the needed APIs first exist, so a too-old transformers errors
    clearly instead of silently mis-behaving (e.g. transformers 5.5 silently shards nothing for EP).
    See each call site for the empirically-determined floor.
    """
    import transformers
    from packaging.version import parse

    have = transformers.__version__
    if parse(have) < parse(min_version):
        raise RuntimeError(
            f"{feature} requires transformers >= {min_version}, but {have} is installed. "
            f"Upgrade with: pip install -U 'transformers>={min_version}'."
        )


def _ep_config(hf_config):
    """Return the config that carries the EP plan / expert count.

    For a plain MoE causal LM this is the root config. For a VLM whose nested text submodel
    is the MoE (e.g. ``KimiK25.text_config`` -> deepseek_v3, or ``NemotronH_Omni.llm_config``
    -> nemotron_h), this is that nested config.
    """
    if getattr(hf_config, "base_model_ep_plan", None):
        return hf_config
    for _attr in ("text_config", "llm_config"):
        sub = getattr(hf_config, _attr, None)
        if sub is not None and getattr(sub, "base_model_ep_plan", None):
            return sub
    # Fall through: no plan anywhere; let callers decide.
    return hf_config


def num_experts_from_config(hf_config) -> int:
    """Return the MoE expert count from an HF config (0 if the model is not MoE).

    Recurses into ``.text_config`` / ``.llm_config`` so VLMs whose nested text submodel is the MoE
    (Kimi-K2.7, Llama-4 multimodal, NemotronH-Omni, ...) report the LM's expert count.
    """
    for cfg in (
        hf_config,
        getattr(hf_config, "text_config", None),
        getattr(hf_config, "llm_config", None),
    ):
        if cfg is None:
            continue
        for attr in ("num_experts", "num_local_experts", "n_routed_experts", "num_routed_experts"):
            value = getattr(cfg, attr, None)
            if value:
                return int(value)
    return 0


def choose_ep_dp_degrees(world_size: int, num_experts: int) -> tuple[int, int]:
    """Pick ``(expert_parallel, expert_data_parallel)`` degrees with ``EP * EDP == world_size``.

    EP must divide both ``num_experts`` (whole experts per rank) and ``world_size`` (a valid mesh
    factorization); we take the largest such EP, and each expert shard is then replicated over the
    remaining ``EDP = world_size // EP`` ranks. Falls back to ``(1, world_size)`` when the model is not
    MoE or no valid EP degree exists. Heuristic -- callers may override.
    """
    if num_experts <= 0 or world_size <= 1:
        return 1, world_size
    for ep in range(world_size, 0, -1):
        if world_size % ep == 0 and num_experts % ep == 0:
            return ep, world_size // ep
    return 1, world_size


# The styles a base_model_ep_plan gives a layer's routed-experts container. A plan that styles only
# the experts' grouped-GEMM parameters names the containers through their parents.
_EXPERTS_STYLES = ("moe_tp_experts", "nemotron_experts_ep")


def _routed_expert_keys(ep_plan: dict[str, str]) -> set[str]:
    """The keys of ``ep_plan``, a ``base_model_ep_plan``, that name a routed-experts container."""
    keys = {key for key, style in ep_plan.items() if style in _EXPERTS_STYLES}
    return keys or {
        key.rsplit(".", 1)[0] for key, style in ep_plan.items() if style == "grouped_gemm"
    }


def _routed_expert_modules(model: "nn.Module", ep_plan: dict[str, str]) -> "dict[str, nn.Module]":
    """``model``'s routed-expert containers, ``{name: module}``, as its ``base_model_ep_plan`` names them.

    A module is one when its name, with layer indices wildcarded as transformers does, is a container
    key or ends with one: the keys are relative to the base model, so they omit the ``model.`` or
    ``language_model.model.`` a causal LM or a VLM puts in front. Deciding a model's distribution and
    loading it expert-parallel both find the experts this way, so the two cannot disagree.
    """
    from transformers.integrations.tensor_parallel import replace_layer_number_by_wildcard

    keys = _routed_expert_keys(ep_plan)
    return {
        name: module
        for name, module in model.named_modules()
        if any(
            (generic := replace_layer_number_by_wildcard(name)) == key
            or generic.endswith("." + key)
            for key in keys
        )
    }


def apply_distributed_support(
    model_type: str,
    hf_config,
    *,
    ckpt_path: "str | None" = None,
    trust_remote_code: bool = False,
):
    """Prepare ``hf_config`` so ``model_type`` can be loaded expert-parallel, from its model spec.

    Reads the model's :class:`~modelopt.torch.models.EPSpec` and applies it: enforce the
    transformers floor this model's expert parallelism needs, then attach the
    ``base_model_ep_plan`` it requires to whichever config carries its MoE -- the root for a plain
    MoE, the nested text/LLM config for a VLM whose experts live in its language sub-model, since
    the plan's globs are written against that module tree and a plan on the wrong config matches
    nothing at all rather than failing.

    An existing plan always wins: stock transformers ships one for some models, and overwriting it
    would substitute our idea of the layout for the model's own.

    Returns the config to load, unchanged for a model type with no spec or no EP section. The
    generic all-to-all transport is not wired here -- it self-registers in
    :func:`~modelopt.torch.utils.plugins.model_load_utils.load_and_prepare_ep`, so nothing about it
    is per-model.
    """
    from modelopt.torch.models import get_spec

    spec = get_spec(model_type)
    ep = getattr(spec, "ep_spec", None) if spec is not None else None
    if ep is None:
        return hf_config

    if ep.unsupported:
        raise NotImplementedError(
            f"{model_type} cannot be quantized expert-parallel yet: {ep.unsupported} Quantize it "
            "without expert parallelism, e.g. with FSDP2 (--use_fsdp2 in examples/hf_ptq/hf_ptq.py)."
        )
    if ep.requires_transformers:
        require_transformers(ep.requires_transformers, f"{model_type} expert-parallel PTQ")

    if ep.base_model_ep_plan:
        target = hf_config
        if ep.plan_config_attr:
            target = getattr(hf_config, ep.plan_config_attr, None)
            if target is None:
                raise ValueError(
                    f"{model_type} expects its expert-parallel plan on "
                    f"config.{ep.plan_config_attr}, which this checkpoint's config does not have."
                )
        if not getattr(target, "base_model_ep_plan", None):
            target.base_model_ep_plan = dict(ep.base_model_ep_plan)

    if ep.prepare is not None:
        hf_config = ep.prepare(hf_config, ckpt_path=ckpt_path, trust_remote_code=trust_remote_code)
    return hf_config


def _spec_ep_plan(model_type: str) -> dict[str, str]:
    """The ``base_model_ep_plan`` :func:`apply_distributed_support` would attach for ``model_type``.

    Itself, or through the spec's ``prepare``. Empty for a model type with no spec, no
    expert-parallel section, or one that refuses expert parallelism. Only reads the spec: nothing
    is applied, checked or patched, and ``prepare`` does not run.
    """
    from modelopt.torch.models import get_spec

    spec = get_spec(model_type)
    ep = getattr(spec, "ep_spec", None) if spec is not None else None
    if ep is None or ep.unsupported:
        return {}
    return dict(ep.base_model_ep_plan or ep.prepared_ep_plan or {})


# ---------------------------------------------------------------------------------------------------
# DP=EP all-to-all expert transport (`torch_a2a`)
#
# The throughput-correct MoE parallelism for calibration: dense+router run data-parallel (each rank its
# own batch), experts are EP-sharded, and tokens are routed to the expert-owning rank by an all-to-all
# (dispatch) / reduce (combine). The expert matmul stays on transformers' `_grouped_linear`, so
# ModelOpt's per-expert quantizers attach and collect amax over the *global* (dispatched) token pool.
# Transport is plain `torch.distributed.all_to_all_single` (NCCL) -- works cross-node at any rank count.
# ---------------------------------------------------------------------------------------------------


# all_to_all_single addresses its flattened buffers with int32 element offsets, so a hop larger than
# this overflows them; _a2a_moe_torch chunks such transfers along the hidden dim.
_A2A_MAX_HOP_ELEMENTS = 2**31 - 1


def _a2a_moe_torch(
    group,
    hidden_states,
    topk_idx,
    topk_weights,
    num_global_experts,
    num_local_experts,
    expert_fn,
):
    """All-to-all MoE expert stage via plain `torch.distributed.all_to_all_single` (NCCL).

    Works cross-node at any rank count, with explicit token permute/unpermute.

    Flow: expand (token, expert) pairs -> route each to dest_rank = global_expert // num_local ->
    all_to_all the tokens (+ local-expert id + weight) -> local grouped_mm (quantizable) -> weight ->
    all_to_all back -> scatter-sum to original tokens.

    Tokens travel and the result returns in ``hidden_states``' dtype -- the model's, so its residual
    stream keeps one dtype -- and only the per-token combine accumulates in fp32.
    """
    world = group.size()
    x = hidden_states.contiguous()
    H = x.size(-1)  # noqa: N806
    T, top_k = topk_idx.shape  # noqa: N806
    dev = x.device

    def _one_all_to_all(output, input_, out_split, in_split):
        """One all_to_all_single hop (hidden, expert-output, AND routing weight).

        Uses the AUTOGRAD-AWARE ``torch.distributed.nn.functional.all_to_all_single`` (whose backward is
        the reverse all-to-all) when a graph is being built AND the input needs grad -- so gradients
        propagate through dispatch/combine to the routed experts (AutoQuantize gradient score estimation)
        AND, via the routing-weight hop, back to the gate/router (MoE training through the experts).
        Otherwise (forward-only calibration / inference, or a frozen input) it uses the plain, no-graph
        collective, leaving the validated production path byte-for-byte unchanged.
        """
        if torch.is_grad_enabled() and input_.requires_grad:
            # torch.distributed.nn.functional.all_to_all_single is the autograd-aware collective (backward =
            # reverse all-to-all). It is deprecated in torch>=2.13 in favor of
            # torch.distributed._functional_collectives.all_to_all_single (AsyncCollectiveTensor); migrating
            # is a follow-up -- suppress the per-call FutureWarning here to keep score-estimation logs clean.
            import warnings

            from torch.distributed.nn.functional import all_to_all_single as _ata_autograd

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, message=".*all_to_all_single.*"
                )
                return _ata_autograd(output, input_, out_split, in_split, group=group)
        dist.all_to_all_single(output, input_, out_split, in_split, group=group)
        return output

    def _all_to_all(output, input_, out_split, in_split):
        """All-to-all that chunks along the hidden dim to stay inside the int32 element limit.

        all_to_all_single, transparently CHUNKED along the hidden dim when a single transfer would
        exceed the int32 element limit. ``all_to_all_single`` addresses the flattened ``[rows, H]`` buffer
        with element offsets ``= row_offset * H``; once ``rows*H > 2**31`` that offset overflows int32 in
        the NCCL path and the collective HANGS (observed: qwen3_5_moe_text EP calib at batch8 x seq4096 ->
        Q*H = 327680*8192 = 2.68e9 > 2**31, all_to_all ran the full 3h NCCL timeout, then aborted). Split
        the LAST (hidden) dim into column slices so every hop stays < 2**31 elements; the per-rank ROW split
        sizes are identical for every slice, and cat-ing the slices is autograd-safe.
        """
        int32_max = _A2A_MAX_HOP_ELEMENTS
        if input_.dim() >= 2:
            # n_chunks (the number of collective HOPS we issue) MUST be identical on every rank, but the
            # recv size Q*H is per-rank data-dependent (routing imbalance -> different Q). Deciding n_chunks
            # from the LOCAL numel desyncs the hop count and re-deadlocks. All-reduce the MAX transfer size
            # so every rank agrees on the same split. Unconditional over the group (every rank runs this same
            # branch for a 2-D hop) so the all_reduce itself can never desync; it's a cheap scalar.
            gmax_t = torch.tensor(
                [max(output.numel(), input_.numel())], device=output.device, dtype=torch.int64
            )
            dist.all_reduce(gmax_t, op=dist.ReduceOp.MAX, group=group)
            gmax = int(gmax_t.item())
            if gmax > int32_max:
                hdim = input_.shape[-1]
                n_chunks = (gmax + int32_max - 1) // int32_max
                step = (hdim + n_chunks - 1) // n_chunks
                if torch.is_grad_enabled() and input_.requires_grad:
                    # Autograd (AutoQuantize scoring): cat is graph-safe (an in-place copy_ would break the
                    # reverse-all-to-all backward). Higher peak, but grad scoring uses far fewer tokens.
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
                # Forward-only calibration: write each hidden-column slice back into `output` IN PLACE so we
                # never hold a full second [Q,H] copy (torch.cat would). The imbalanced-rank recv buffer is
                # already tens of GiB at long seq x batch; the cat doubling is what tips calibration into OOM.
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
        return _one_all_to_all(output, input_, out_split, in_split)

    # (token, expert) pairs (drop any -1 "no selection" slots).
    token_rows = torch.arange(T, device=dev).repeat_interleave(top_k)  # [P]
    exp = topk_idx.reshape(-1).to(torch.int64)
    w = topk_weights.reshape(-1).to(x.dtype)
    valid = exp >= 0
    token_rows, exp, w = token_rows[valid], exp[valid], w[valid]
    dest = (exp // num_local_experts).to(torch.int64)  # owner rank of each pair
    local_eid = exp % num_local_experts

    order = torch.argsort(dest, stable=True)  # group pairs by dest rank
    token_rows_s, local_eid_s, w_s = token_rows[order], local_eid[order], w[order]
    send_hidden = x[token_rows_s].contiguous()  # [P, H]
    in_counts = torch.bincount(dest[order], minlength=world).to(torch.int64)  # rows -> each rank

    out_counts = torch.empty_like(in_counts)  # rows <- each rank
    # The host reads out_counts (.tolist()) immediately below to size the dispatch buffers (Q = sum(out)).
    # all_to_all_single runs on the NCCL stream; a HOST read of its output must wait for the collective, or
    # .tolist() can copy stale/uninitialized out_counts -> a garbage Q -> a size-mismatched dispatch
    # all_to_all -> NCCL out-of-bounds -> SIGSEGV (deterministic on the qwen3_5_moe_text EP path; masked on
    # heavier models only by pre-MoE timing, and masked everywhere by CUDA_LAUNCH_BLOCKING=1). Take the
    # work handle and wait() so the current stream (hence the .tolist() DtoH) orders after the exchange.
    _cnt_work = dist.all_to_all_single(out_counts, in_counts, group=group, async_op=True)
    _cnt_work.wait()
    in_list, out_list = in_counts.tolist(), out_counts.tolist()
    Q = int(sum(out_list))  # noqa: N806

    # Q (tokens RECEIVED) is what sizes every buffer below, and it is data-dependent: a balanced
    # router gives Q ~= P, but a collapsed one piles the whole cluster's tokens onto a few experts
    # and Q grows without bound. Report it HERE rather than only in the end-of-function trace --
    # the expert stage allocates O(Q*H) immediately after, so on a bad routing draw the function
    # OOMs before that trace can run and the one number that explains the failure is never printed.
    P_sent = int(send_hidden.size(0))  # noqa: N806
    if 4 * max(P_sent, 1) < Q:
        warnings.warn(
            f"[a2a] rank {dist.get_rank()} received Q={Q} tokens for {num_local_experts} local "
            f"expert(s) after sending P={P_sent} -- a {Q / max(P_sent, 1):.1f}x imbalance. The "
            f"expert stage allocates O(Q x H), so this is a likely OOM. Routing has probably "
            f"collapsed onto few experts (check the router/hidden states for NaN or blow-up).",
            stacklevel=2,
        )
    # dispatch hidden (autograd-aware when scoring) + local-expert id + weight (same split sizes)
    recv_hidden = torch.empty(Q, H, device=dev, dtype=x.dtype)
    recv_hidden = _all_to_all(recv_hidden, send_hidden, out_list, in_list)
    recv_eid = torch.empty(Q, device=dev, dtype=torch.int64)
    dist.all_to_all_single(recv_eid, local_eid_s.contiguous(), out_list, in_list, group=group)
    # Routing WEIGHT hop is gradient-carrying (its backward reaches the gate/router when training the MoE
    # through the experts), so it goes through the autograd-aware ``_all_to_all`` -- unlike the integer
    # expert-id hop above, which is a non-differentiable routing DECISION. Falls back to the plain
    # collective when ``w_s`` doesn't need grad (frozen router / forward-only), so PTQ is unchanged.
    recv_w = torch.empty(Q, device=dev, dtype=x.dtype)
    recv_w = _all_to_all(recv_w, w_s.contiguous(), out_list, in_list)

    # group received tokens by local expert -> grouped_mm (quantizable) -> weight
    rorder = torch.argsort(recv_eid, stable=True)
    counts = torch.bincount(recv_eid[rorder], minlength=num_local_experts)
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    eo = expert_fn(recv_hidden[rorder], offsets).to(x.dtype) * recv_w[rorder].unsqueeze(-1)
    eo_recv = torch.empty(Q, H, device=dev, dtype=x.dtype)
    eo_recv[rorder] = eo  # back to recv order

    # combine: all_to_all back (swap counts, autograd-aware when scoring) -> scatter-sum to original tokens
    send_back = torch.empty(send_hidden.size(0), H, device=dev, dtype=x.dtype)
    send_back = _all_to_all(send_back, eo_recv, in_list, out_list)
    out_pair = torch.empty_like(send_back)
    out_pair[order] = send_back  # back to pair order
    # Accumulate the per-token top-k scatter-sum in FP32, then round ONCE. Each token's output is a sum
    # of its top_k expert results; accumulating that in bf16 rounds at every partial add (~0.4% each), so
    # a different summation ORDER gives a different answer -- and the a2a returns partials in
    # post-all_to_all order, which differs from the native (non-EP) forward's local order. That made the
    # exported activation scales EP-dependent: on Qwen3-30B-A3B (top_k=8) 27% of input_scale values
    # differed from the single-GPU reference, by up to 28%. FP32 accumulation is order-independent to
    # ~1e-7, so every order rounds to the same bf16 value -> byte-identical scales at EP1/2/4/8
    # (validated single- and multi-node). Costs one T x H fp32 temporary; no measurable runtime change.
    _acc = torch.zeros(T, H, device=dev, dtype=torch.float32)
    _acc.index_add_(0, token_rows, out_pair.float())
    result = _acc.to(x.dtype)
    return result


def a2a_moe(
    group,
    hidden_states,
    topk_idx,
    topk_weights,
    num_global_experts,
    num_local_experts,
    expert_fn,
):
    """Route one MoE layer's expert stage by all-to-all (NCCL)."""
    return _a2a_moe_torch(
        group,
        hidden_states,
        topk_idx,
        topk_weights,
        num_global_experts,
        num_local_experts,
        expert_fn,
    )


def torch_a2a_experts_forward(self, hidden_states, top_k_index, top_k_weights, process_group=None):
    """All-to-all expert-parallel forward for PTQ calibration (DP=EP).

    Registered as the `"torch_a2a"` experts implementation on transformers' `ExpertsInterface`.

    Dense+router run data-parallel; experts are EP-sharded; tokens are routed to the expert-owning
    rank by the NCCL all-to-all above. Local experts run through the ModelOpt-quantizable
    `_grouped_linear` so per-expert weight + shared input quantizers attach over the dispatched
    (global) token pool. `process_group` (the EP group) is injected by the `torch_a2a_experts` TP style's
    `_prepare_input_fn`; the router must emit GLOBAL ids unmasked (`torch_a2a_router`).
    """
    from transformers.integrations.moe import _grouped_linear

    if process_group is None:
        raise ValueError(
            "torch_a2a experts forward requires the EP `process_group` (use the `torch_a2a_experts` "
            "TP style, which injects it via _prepare_input_fn)."
        )
    num_local = self.num_experts
    num_global = num_local * process_group.size()
    # Per-expert biases (e.g. GPT-OSS ``gate_up_proj_bias`` / ``down_proj_bias``, shape [num_local, out],
    # EP-sharded dim-0 by grouped_gemm like the weights). transformers' ``_grouped_linear`` (and ModelOpt's
    # quantized override) already accept ``bias`` and do ``out += bias`` -- transposed-safe. Bias is added
    # AFTER each expert's grouped GEMM, unquantized.
    has_bias = getattr(self, "has_bias", False)

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

    return a2a_moe(
        process_group, hidden_states, top_k_index, top_k_weights, num_global, num_local, expert_fn
    )


_TORCH_A2A_REGISTERED = False


def register_torch_a2a():
    """Register everything the torch_a2a path needs into stock transformers (idempotent).

    Done lazily from ``load_and_prepare_ep`` -- NOT at import time, so importing this module has no
    transformers side effect:

    - the ``torch_a2a`` experts FORWARD into ``ExpertsInterface`` (``ALL_EXPERTS_FUNCTIONS``);
    - two EP parallel STYLES into ``ALL_PARALLEL_STYLES``:

      * ``torch_a2a_router`` -- emit GLOBAL unmasked expert ids (skip ``RouterParallel``'s
        per-EP-rank masking) so the a2a dispatch can route each (token, expert) to its owning rank;
      * ``torch_a2a_experts`` -- inject the EP ``process_group`` into the experts forward (4th arg)
        and skip the all-reduce (the a2a combine does the reduction).

    These two styles are functionally the stock ``megamoe_router`` / ``megamoe_experts`` (which were
    added for DeepGEMM Mega-MoE in transformers 5.10.4), reimplemented here -- ~5 lines each -- so the
    a2a path OWNS its EP styles and does NOT depend on megamoe (i.e. it stays portable below 5.10.4).
    The base classes (``RouterParallel``, ``TensorParallelLayer``) are stock wherever the EP infra
    exists (transformers >= 5.6, the framework floor enforced in ``load_and_prepare_ep``), so this
    works on any such version; ``grouped_gemm`` (expert weight sharding) is stock too.
    """
    global _TORCH_A2A_REGISTERED
    if _TORCH_A2A_REGISTERED:
        return
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        RouterParallel,
        TensorParallelLayer,
    )

    ALL_EXPERTS_FUNCTIONS.register("torch_a2a", torch_a2a_experts_forward)

    class TorchA2ARouter(RouterParallel):
        """torch_a2a router style: emit GLOBAL unmasked expert ids (== stock ``megamoe_router``).

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

        # --- transformers >=5.16 API ---------------------------------------------------------
        # install_forward() there calls transform_inputs_pre_forward/transform_output_post_forward
        # and NEVER the _prepare_* pair above, whose base implementations are no-ops. Without these
        # two methods the EP process_group is silently not injected and torch_a2a_experts_forward
        # raises "requires the EP process_group". Additive: <5.16 ignores them.
        def should_use_local_tensors(self, module):
            # The a2a forward dispatches tokens itself and runs the LOCAL expert shard through
            # _grouped_linear; without this the grouped matmul mixes DTensor params with plain
            # activation tensors and aten._grouped_mm rejects the call.
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


def expert_parallel_mesh(model: nn.Module):
    """The mesh ``model``'s routed experts are sharded over, or ``None`` if its plan shards none.

    :func:`~modelopt.torch.utils.plugins.model_load_utils.load_and_prepare_ep` loads the model over
    its expert-parallel submesh, which transformers keeps on the model as ``_device_mesh``.
    Calibration takes the mesh to re-install the all-to-all forwards
    (:func:`~modelopt.torch.utils.distributed.distributed_forward_loop`).
    """
    if not distribution_plan(model).is_expert_parallel:
        return None
    return model._device_mesh
