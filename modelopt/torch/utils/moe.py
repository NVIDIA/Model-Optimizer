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

Finding a model's routed experts, choosing the EP degree they are sharded over, the DP=EP
all-to-all expert transport (``torch_a2a``), and the EP-sharded model loader.

The all-to-all transport is plain ``torch.distributed.all_to_all_single`` (NCCL) wrapped around
transformers' quantizable ``_grouped_linear``. ``register_torch_a2a()`` registers
it into transformers' ``ExpertsInterface`` (lazily, from ``load_and_prepare_ep``'s a2a path -- never at
import time, so importing this module has no transformers-``ExpertsInterface`` side effect).
"""

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.nn.functional import all_to_all_single as autograd_all_to_all_single

if TYPE_CHECKING:
    from .distributed import ModelOptSession

if TYPE_CHECKING:
    from torch import nn


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


# ---------------------------------------------------------------------------------------------------
# DP=EP all-to-all expert transport (`torch_a2a`)
#
# The throughput-correct MoE parallelism for calibration: dense+router run data-parallel (each rank its
# own batch), experts are EP-sharded, and tokens are routed to the expert-owning rank by an all-to-all
# (dispatch) / reduce (combine). The expert matmul stays on transformers' `_grouped_linear`, so
# ModelOpt's per-expert quantizers attach and collect amax over the *global* (dispatched) token pool.
# Transport is plain `torch.distributed.all_to_all_single` (NCCL) -- works cross-node at any rank count.
# ---------------------------------------------------------------------------------------------------


# all_to_all_single addresses its flattened buffers with int32 element offsets; a2a_moe sends a larger
# hop as hidden-column slices, so every hop stays within them.
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
    all_to_all the tokens (+ local-expert id) -> local grouped_mm (quantizable) -> all_to_all back ->
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

        all_to_all_single addresses the flattened ``[rows, H]`` buffers with int32 element offsets;
        slicing keeps every hop within them. Every slice has the same row splits, and concatenating the
        slices is autograd-safe.
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

    # group received tokens by local expert -> grouped_mm (quantizable); weighted back home
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
    rank by the NCCL all-to-all above. Local experts run through the ModelOpt-quantizable
    `_grouped_linear` so per-expert weight + shared input quantizers attach over the dispatched
    (global) token pool. `process_group` (the EP group) is injected by the `torch_a2a_experts` TP style's
    `_prepare_input_fn`; the router must emit GLOBAL ids unmasked (`torch_a2a_router`).

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
    # ``_grouped_linear`` (and ModelOpt's quantized override) add them after each expert's grouped GEMM,
    # unquantized.
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


def _routed_expert_modules(model: nn.Module, ep_plan: dict[str, Any]) -> dict[str, nn.Module]:
    """Find the EP-sharded routed-expert containers from ``base_model_ep_plan``; return ``{name: module}``.

    Uses the SAME plan that drove the EP sharding to decide expert-ness (not a per-class flag or FQN
    heuristic): a module is a routed-expert container iff its name -- with layer indices wildcarded exactly
    as HuggingFace does (``replace_layer_number_by_wildcard``) -- matches, as a SUFFIX (tolerating the
    ``model.`` / ``language_model.model.`` prefixes the base-relative plan keys omit), a plan key whose
    (post-swap) style is ``torch_a2a_experts``. ``grouped_gemm`` (per-parameter shard) and
    ``torch_a2a_router`` (replicated router) entries are excluded -- only the grouped-GEMM experts
    container holds the expert quantizers.
    """
    from transformers.integrations.tensor_parallel import replace_layer_number_by_wildcard

    # a2a tags the experts container ``torch_a2a_experts``.
    expert_keys = [k for k, v in ep_plan.items() if v == "torch_a2a_experts"]
    experts = {
        name: module
        for name, module in model.named_modules()
        if any(
            (generic := replace_layer_number_by_wildcard(name)) == k or generic.endswith("." + k)
            for k in expert_keys
        )
    }

    # Quantize the grouped experts through the patched ``_grouped_linear`` and keep the fused grouped-GEMM
    # weights + the ``torch_a2a`` experts forward. If ModelOpt has a per-expert-nn.Linear quant wrapper
    # registered for the experts class (Qwen3VLMoeTextExperts, Llama4TextExperts, DbrxExperts),
    # ``mtq.quantize`` would convert the module to that wrapper -- whose forward drops the a2a-injected
    # process_group ("forward() takes 4 positional args but 5 given") and discards the grouped structure.
    # Unregister those classes so the experts stay a2a-managed (matches deepseek/glm/minimax, which have NO
    # registered wrapper).
    try:
        from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry

        for cls in {type(m) for m in experts.values()}:
            if cls in QuantModuleRegistry:
                QuantModuleRegistry.unregister(cls)
    except Exception as _e:  # pragma: no cover - defensive; never block the load on this
        print(f"[ep] WARNING: could not unregister experts quant wrapper: {_e}", flush=True)
    return experts


def _seed_parallel_states(model: nn.Module, expert_names, dense_state, expert_state) -> None:
    """Seed every module's ``_parallel_state`` for calibration's amax sync.

    The routed-expert containers named in ``expert_names`` and everything under them get
    ``expert_state``; every other module gets ``dense_state``. Seeding the plain modules works because
    quantization keeps a ``_parallel_state`` that is already present when it converts a module -- the same
    way Megatron's expert MLPs seed theirs.
    """
    prefixes = tuple(name + "." for name in expert_names)
    for name, module in model.named_modules():
        module._parallel_state = (
            expert_state if prefixes and (name + ".").startswith(prefixes) else dense_state
        )


def _ckpt_quant_method(cfg):
    """Return the checkpoint's native quantization method, or ``None``.

    Return the checkpoint's native quantization method (e.g. ``"fp8"`` for block-fp8 DeepSeek-V3),
    or ``None`` if the checkpoint isn't natively quantized. Checks the top config and the VLM ``text_config``.
    """
    for c in (cfg, getattr(cfg, "text_config", None)):
        qc = getattr(c, "quantization_config", None)
        m = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        if m:
            # transformers' config objects carry a QuantizationMethod enum, whose str() is not its value.
            return str(getattr(m, "value", m))
    return None


def load_and_prepare_ep(
    ckpt_path: str,
    mo_session: "ModelOptSession",
    ep_degree: int,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    freeze: bool = True,
    hf_config=None,
    require_lm_head: bool = True,
) -> nn.Module:
    """Load + shard a MoE causal LM for expert-parallel PTQ via the DP=EP all-to-all transport.

    Builds a ``(dp, ep)`` device mesh and loads experts EP-sharded over the ``ep`` submesh
    (``enable_expert_parallel=True`` -- each rank owns ``num_experts / ep`` experts as grouped-GEMM
    shards), while ATTENTION and dense params stay REPLICATED (so EP has no KV-head-divisibility cap).
    The model's ``base_model_ep_plan`` is swapped to the ``torch_a2a`` styles: the router emits GLOBAL
    unmasked expert ids and the experts forward routes each token to its expert-owning rank by a NCCL
    all-to-all (dispatch) -> quantizable ``_grouped_linear`` -> all-to-all (combine), rather than every
    rank computing all tokens followed by an EP all-reduce. Every rank is data-parallel (sees DIFFERENT
    tokens), so calibration MUST shard the data per-rank.

    The ``dp`` dimension (present when ``ep_degree < world_size``) is additional data parallelism across
    EP groups: non-expert params stay REPLICATED per rank and each EP group calibrates on different
    data; experts stay EP-sharded (resident). No FSDP and no concurrent ``mesh_dp``/``mesh_ep``
    collectives -> no deadlock by construction. Costs more per-rank memory (non-experts + experts
    replicated across dp) but is the reliable path at scale.

    EP CANNOT be applied to a meta-init model (transformers' ``GroupedGemmParallel.shard_tensor``
    slices materialized tensors), so this loads real weights via ``from_pretrained`` over the ``ep``
    submesh. Requires the transformers MoE EP framework (ExpertsInterface / grouped-GEMM EP styles /
    enable_expert_parallel), first present in transformers >= 5.6. (Per-model plugins enforce a higher
    floor where their stock support landed later, e.g. qwen3_moe's stock base_model_ep_plan in >= 5.7.)

    Assumes the caller has already initialized (and device-bound) the process group -- distributed
    bring-up is the launcher's job, not this loader's (hf_ptq.main() does it via modelopt_session).
    """
    # Hard floor: the transformers MoE expert-parallel framework. Below 5.6 there is no ExpertsInterface
    # / grouped-GEMM EP styles, and enable_expert_parallel silently shards nothing (no error) -> fail loud.
    require_transformers(
        "5.6", "ModelOpt DP=EP expert-parallel PTQ (transformers MoE EP framework)"
    )
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    from .distributed import ParallelState

    if mo_session.world_size % ep_degree != 0:
        raise ValueError(
            f"world_size {mo_session.world_size} not divisible by ep_degree {ep_degree}"
        )
    dp = mo_session.world_size // ep_degree
    if hf_config is None:
        config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if attn_implementation is not None:
            config_kwargs["attn_implementation"] = attn_implementation
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
    dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16

    # This loader assumes the caller has ALREADY initialized (and device-bound) the process group --
    # distributed bring-up belongs to the launcher, not this library function. The example launcher does
    # it in hf_ptq.main() via modelopt_session() -> modelopt.torch.utils.distributed.setup() before
    # the load. A live, device-bound PG is required here so the mesh subgroups built below create their
    # NCCL comms eagerly (the synchronized warmup all-reduce per subgroup further down) instead of lazily
    # on first collective (which deadlocks at DP>1 because ranks reach the lazy init in different phases).
    if not mo_session.initialized:
        raise RuntimeError(
            "load_and_prepare_ep requires an initialized ModelOptSession (with a live, device-bound "
            "process group); enter modelopt.torch.utils.distributed.modelopt_session() (or pass an "
            "initialized session) before calling it."
        )
    # Two INDEPENDENT meshes (NeMo-AutoModel style), named by the model part they govern -- two
    # factorizations of the same world:
    #   sparse_moe_mesh = (edp, ep): experts are SHARDED over the ``ep`` axis and REPLICATED over ``edp``;
    #       the ``ep`` submesh also drives from_pretrained's expert sharding + the a2a transport.
    #   dense_mesh = (dp,): dense/non-expert params are data-parallel-REPLICATED over the WHOLE world.
    # This subsumes the old a2a-vs-partial split: a2a is simply dp == 1 (``edp`` size 1), and dense amax
    # always reduces over the whole world (fixing the DP>1 case, where the old dp submesh missed the ep dim).
    sparse_moe_mesh = init_device_mesh("cuda", (dp, ep_degree), mesh_dim_names=("edp", "ep"))
    dense_mesh = init_device_mesh("cuda", (mo_session.world_size,), mesh_dim_names=("dp",))
    ep_submesh = sparse_moe_mesh["ep"]

    # NO TENSOR PARALLELISM IS REQUESTED HERE, despite the name. transformers implements expert
    # parallelism THROUGH the TP application path: with enable_expert_parallel=True the model's
    # ``tp_plan`` PROPERTY returns ``_ep_plan`` (distributed/mixin.py), so what actually gets applied
    # is the expert plan. ``tp_plan="auto"`` is only the SWITCH that makes ``tp_size > 1`` in
    # DistributedConfig.__post_init__ and therefore reaches ``apply_tensor_parallelism`` at all --
    # drop it and __post_init__ early-returns with tp_size=None, which then TypeErrors on
    # ``tp_size > 1`` in prepare_distribute_model. Dense params stay REPLICATED (= our DP) because
    # ``swapped_plan`` below keeps only expert entries; attention/embeds match no plan key.
    #
    # WHERE it goes is version-dependent: transformers >=5.16 raises if tp_plan is passed alongside
    # distributed_config ("Set tensor-parallel options on DistributedConfig when using it"); older
    # releases have no tp_plan field on DistributedConfig at all. Passing tp_plan ALONE is not an
    # option either -- the >=5.16 back-compat path rebuilds the config as
    # DistributedConfig(tp_plan=..., tp_size=...) and DROPS enable_expert_parallel, silently losing
    # EP. So decide from the dataclass fields, which tell the two apart on any release.
    import dataclasses as _dc

    _dc_has_tp_plan = any(f.name == "tp_plan" for f in _dc.fields(DistributedConfig))
    from_pretrained_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_mesh": ep_submesh,
        "distributed_config": (
            DistributedConfig(enable_expert_parallel=True, tp_plan="auto")
            if _dc_has_tp_plan
            else DistributedConfig(enable_expert_parallel=True)
        ),
        "trust_remote_code": trust_remote_code,
    }
    if not _dc_has_tp_plan:
        from_pretrained_kwargs["tp_plan"] = "auto"
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    if _ckpt_quant_method(hf_config) == "mxfp4":
        # Native mxfp4 checkpoints (e.g. gpt-oss: fused `gate_up_proj_blocks`/`_scales` + biases) must be
        # DEQUANTIZED to bf16 on load so ModelOpt can insert nvfp4/fp8 quantizers on the fused bf16 experts
        # (the a2a grouped_gemm expects bf16). transformers dequantizes blocks->bf16 during from_pretrained.
        # NB: the single-process loader needed device_map="sequential" to dodge a CUDA illegal-access in the
        # mxfp4 dequant under multi-GPU device_map="auto"; the EP path uses device_mesh (no device_map), so
        # watch for an EP-shard-awareness issue in the dequant convert-op.
        from transformers import Mxfp4Config

        from_pretrained_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        if mo_session.rank == 0:
            print(
                "[ep] native mxfp4 checkpoint (e.g. gpt-oss) -> Mxfp4Config(dequantize=True) -> bf16",
                flush=True,
            )
    ep_cfg = _ep_config(hf_config)  # for VLMs the plan lives on .text_config
    plan = dict(getattr(ep_cfg, "base_model_ep_plan", None) or {})
    if not plan:
        raise ValueError("expert-parallel PTQ requires the model to declare base_model_ep_plan")
    # DP=EP all-to-all: swap the ep_plan to our torch_a2a styles (torch_a2a_router emits GLOBAL ids
    # unmasked; torch_a2a_experts injects the EP process_group + skips the all-reduce) and select the
    # `torch_a2a` experts forward (NCCL all-to-all dispatch/combine around the quantizable
    # _grouped_linear). Tokens are routed to the expert-owning rank by all-to-all instead of every rank
    # computing all tokens + all-reduce. Requires per-rank (data-parallel) calib data. nemotron_h's
    # logits-only gate has no ep_router entry, so its nemotron_experts_ep also maps to torch_a2a_experts.
    register_torch_a2a()  # idempotent; register the experts forward + the two styles, right before use
    swap = {
        "ep_router": "torch_a2a_router",
        "moe_tp_experts": "torch_a2a_experts",
        "nemotron_experts_ep": "torch_a2a_experts",
    }
    # Keep expert-related entries (grouped_gemm expert params + the router/experts styles); DROP the TP
    # styles (a2a builds no TP group for them -> would shard attention across the wrong mesh).
    swapped_plan = {k: swap.get(v, v) for k, v in plan.items() if v in swap or v == "grouped_gemm"}
    # Ensure the routed-experts CONTAINER is tagged torch_a2a_experts (injects the EP process_group +
    # marks the routed expert). Some plans tag only the expert PARAMS grouped_gemm, not the container;
    # derive it as the parent of the grouped_gemm expert-param keys and add it if missing.
    if not any(v == "torch_a2a_experts" for v in swapped_plan.values()):
        for parent in {k.rsplit(".", 1)[0] for k, v in plan.items() if v == "grouped_gemm"}:
            swapped_plan[parent] = "torch_a2a_experts"
    from_pretrained_kwargs["experts_implementation"] = "torch_a2a"
    ep_cfg.base_model_ep_plan = swapped_plan
    from_pretrained_kwargs["config"] = hf_config
    if mo_session.rank == 0:
        print(f"[ep] ep_plan={swapped_plan} experts_implementation=torch_a2a", flush=True)
    # Pick the loader by model type, matching examples' get_model: VLM (nested text_config / a
    # *ForConditionalGeneration arch) -> AutoModel so the WHOLE checkpoint loads (vision tower + projector
    # + nested LM, which carries the lm_head); a plain causal LM -> AutoModelForCausalLM so the model
    # INCLUDES the lm_head. NB: for a plain causal LM, AutoModel returns the BASE model (e.g.
    # Qwen3MoeModel / NemotronHModel) WITHOUT lm_head -> the head is never calibrated/exported and the
    # exported checkpoint generates garbage (untied random head). The per-model plugin (models/) has
    # already installed any on-the-fly weight decompression as a transformers monkeypatch.
    _archs = getattr(hf_config, "architectures", None) or []
    _is_vlm = (
        getattr(hf_config, "text_config", None) is not None
        or getattr(hf_config, "vision_config", None) is not None
        or any("conditionalgeneration" in str(a).lower() for a in _archs)
    )
    loader = AutoModel if _is_vlm else AutoModelForCausalLM
    # A VLM's AutoModel returns the BACKBONE (vision + projector + nested LM) WITHOUT the lm_head. We now
    # DEFAULT require_lm_head=True (for ALL models, VLMs included) so the loaded model carries the lm_head
    # and .generate(), and the pre/post-PTQ generation preview works everywhere. For M3,
    # AutoModel -> MiniMaxM3VLModel (no lm_head; decoder at .language_model); the with-head load returns the
    # full image-text-to-text model instead (*ForConditionalGeneration): [model, lm_head], decoder at
    # model.language_model -- both resolvable. The LM submodel is extracted for quantize
    # (extract_and_prepare_language_model_from_vl); the head/vision are excluded from quant but kept for gen.
    if require_lm_head and _is_vlm:
        # A remote-code VLM may register the FULL *ForConditionalGeneration (WITH lm_head) under
        # AutoModel/AutoModelForCausalLM in its auto_map and provide NO AutoModelForImageTextToText entry
        # (e.g. Kimi-K3: auto_map AutoModelForCausalLM -> KimiK3ForConditionalGeneration). For those,
        # AutoModelForImageTextToText.from_pretrained raises "Unrecognized configuration class ...", so use
        # the remote AutoModelForCausalLM -- it already yields the with-head model + a .language_model
        # decoder. Native VLMs (M3: MiniMaxM3VLConfig; qwen3_5_moe: Qwen3_5MoeConfig) have no auto_map ->
        # the AutoModelForImageTextToText path below applies.
        _auto_map = getattr(hf_config, "auto_map", None) or {}
        _causal_full = (
            "conditionalgeneration" in str(_auto_map.get("AutoModelForCausalLM", "")).lower()
        )
        if _auto_map and _causal_full and "AutoModelForImageTextToText" not in _auto_map:
            loader = AutoModelForCausalLM
        else:
            try:
                from transformers import AutoModelForImageTextToText

                # Only commit to the with-head image-text-to-text class if it actually recognizes this
                # config; otherwise KEEP the headless AutoModel backbone rather than crashing at
                # from_pretrained with "Unrecognized configuration class". PTQ still runs; the generation
                # preview is skipped downstream (pre_quantize guards on hasattr(model, "generate")).
                if type(hf_config) in AutoModelForImageTextToText._model_mapping:
                    loader = AutoModelForImageTextToText
                elif mo_session.rank == 0:
                    print(
                        f"[ep] require_lm_head: {type(hf_config).__name__} is not registered under "
                        f"AutoModelForImageTextToText; keeping headless {loader.__name__} "
                        f"(no lm_head -> generation preview will be skipped).",
                        flush=True,
                    )
            except ImportError:
                pass
    if mo_session.rank == 0:
        print(
            f"[ep] loader={loader.__name__} (is_vlm={_is_vlm}, "
            f"require_lm_head={require_lm_head}, archs={_archs})",
            flush=True,
        )
    model = loader.from_pretrained(ckpt_path, **from_pretrained_kwargs)
    model.eval()
    # Force the experts forward to dispatch to `torch_a2a`: from_pretrained's `experts_implementation`
    # kwarg resolves non-fp8 impls back to "grouped_mm", but @use_experts_implementation reads
    # `config._experts_implementation` at FORWARD time, so set it on every (sub)config now. (Without
    # this the torch_a2a_experts style passes a 5th process_group arg to grouped_mm_experts_forward ->
    # TypeError.)
    for _m in [model, *model.modules()]:
        _cfg = getattr(_m, "config", None)
        if _cfg is not None and hasattr(_cfg, "_experts_implementation"):
            _cfg._experts_implementation = "torch_a2a"

    # DP=EP all-to-all: no FSDP. Non-experts stay replicated per rank (from_pretrained already placed
    # them on this rank's GPU); experts stay EP-sharded on the ep submesh. The model carries
    # model._device_mesh = ep_submesh (sparse_moe_mesh["ep"]), which the export uses to detect EP and the
    # per-expert distributed writer reads for the ep_rank offset + dp-replica dedup.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if freeze:
        model.requires_grad_(False)

    # The routed-expert containers, from the SAME base_model_ep_plan that drove the sharding. A MoE model
    # loaded WITHOUT EP (dense DP) never reaches this loader, so its experts stay dense.
    expert_modules = _routed_expert_modules(model, swapped_plan)

    # Each module's ParallelState for calibration's amax sync: dense modules reduce over the whole world;
    # routed experts reduce over their expert-data-parallel replicas, and share one input amax across the
    # layer's experts over the expert-parallel group. A size-1 axis is left unused (-1): no collective.
    def _axis(mesh):
        return mesh.get_group() if mesh.size() > 1 else -1

    _seed_parallel_states(
        model,
        expert_modules,
        dense_state=ParallelState(data_parallel_group=_axis(dense_mesh["dp"])),
        expert_state=ParallelState(
            data_parallel_group=_axis(sparse_moe_mesh["edp"]),
            expert_model_parallel_group=_axis(sparse_moe_mesh["ep"]),
        ),
    )
    if mo_session.rank == 0:
        print(
            f"[ep] seeded the expert-parallel state of {len(expert_modules)} routed-expert "
            "module(s) from base_model_ep_plan",
            flush=True,
        )
    # Re-planting the a2a forwards after quantizer insertion (``_reinstall_a2a_forwards``) needs this mesh.
    mo_session.ep_mesh = ep_submesh

    if mo_session.distributed:
        # Eagerly create each amax collective's NCCL comm in a globally-consistent order (barrier around
        # each), so the sub-communicators are built in lockstep rather than lazily mid-calibration --
        # which deadlocks when submesh groups are phase-skewed at DP>1. NO forward warmup (variable
        # per-group time would desync ranks before comm creation). Skip size-1 axes (no collective). The
        # EP group is already live from from_pretrained's expert sharding; re-warming it is a harmless
        # redundant all-reduce. Same order on every rank -> lockstep.
        warm_device = next(model.parameters()).device
        for _sub in (dense_mesh["dp"], sparse_moe_mesh["ep"], sparse_moe_mesh["edp"]):
            if _sub.size() <= 1:
                continue
            torch.distributed.barrier()
            torch.distributed.all_reduce(torch.ones(1, device=warm_device), group=_sub.get_group())
            torch.cuda.synchronize()
        torch.distributed.barrier()
    return model
