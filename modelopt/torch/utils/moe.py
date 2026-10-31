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
from packaging.version import parse
from torch.distributed.nn.functional import all_to_all_single as autograd_all_to_all_single

from .distributed import distribution_plan

# The transformers release the expert-parallel load needs, for every model: 5.6 builds a model only
# with the experts implementations it ships, never torch_a2a.
_EP_MIN_TRANSFORMERS = "5.7"


def require_transformers(min_version: str, feature: str) -> None:
    """Fail loudly if the installed transformers is older than ``min_version`` (the floor for ``feature``).

    Callers enforce the version where the needed APIs first exist, so a too-old transformers errors
    clearly instead of silently mis-behaving (e.g. transformers 5.5 silently shards nothing for EP).
    A pre-release of the floor release (``5.7.0.dev0`` for ``5.7``) is refused too: it can predate
    the feature the floor stands for.
    """
    # Imported here: this module imports without transformers.
    import transformers

    have = transformers.__version__
    if parse(have) < parse(min_version):
        raise RuntimeError(
            f"{feature} requires transformers >= {min_version}, but {have} is installed. "
            f"Upgrade with: pip install -U 'transformers>={min_version}'."
        )


# The nested configs a VLM's expert-parallel plan and expert count are read from. A spec attaching
# its plan anywhere else would be read as having none, and the MoE planned as a dense model.
_SUB_CONFIG_ATTRS = ("text_config", "llm_config")


def config_levels(hf_config) -> list:
    """``hf_config`` and its nested language-model configs, those it has, which may carry its settings."""
    nested = (getattr(hf_config, attr, None) for attr in _SUB_CONFIG_ATTRS)
    return [c for c in (hf_config, *nested) if c is not None]


def _ep_config(hf_config):
    """Return the config that carries the EP plan / expert count.

    For a plain MoE causal LM this is the root config. For a VLM whose nested text submodel
    is the MoE (e.g. ``KimiK25.text_config`` -> deepseek_v3, or ``NemotronH_Omni.llm_config``
    -> nemotron_h), this is that nested config.
    """
    if getattr(hf_config, "base_model_ep_plan", None):
        return hf_config
    for _attr in _SUB_CONFIG_ATTRS:
        sub = getattr(hf_config, _attr, None)
        if sub is not None and getattr(sub, "base_model_ep_plan", None):
            return sub
    # Fall through: no plan anywhere; let callers decide.
    return hf_config


def _num_experts_from_config(hf_config) -> int:
    """Return the MoE expert count from an HF config (0 if the model is not MoE).

    Recurses into ``.text_config`` / ``.llm_config`` so VLMs whose nested text submodel is the MoE
    (Kimi-K2.7, Llama-4 multimodal, NemotronH-Omni, ...) report the LM's expert count.
    """
    for cfg in (hf_config, *(getattr(hf_config, attr, None) for attr in _SUB_CONFIG_ATTRS)):
        if cfg is None:
            continue
        for attr in ("num_experts", "num_local_experts", "n_routed_experts", "num_routed_experts"):
            value = getattr(cfg, attr, None)
            if value:
                return int(value)
    return 0


def _choose_ep_degree(world_size: int, num_experts: int) -> int:
    """The largest EP degree dividing both ``world_size`` and ``num_experts``; ``1`` if none does.

    EP must divide ``num_experts`` (whole experts per rank) and ``world_size`` (a valid mesh
    factorization). Each expert shard is then replicated over the remaining ranks
    (``DistributionPlan.edp``). The largest degree holds the fewest experts per rank, which leaves
    the most memory for activations, i.e. for calibration batches; any degree above 1 pays the same
    two all-to-alls per MoE layer. Heuristic -- callers may override.
    """
    if num_experts <= 0:
        return 1
    return next(
        (ep for ep in range(world_size, 0, -1) if world_size % ep == 0 and num_experts % ep == 0),
        1,
    )


# The styles a base_model_ep_plan gives a layer's routed-experts container. A plan that styles the
# experts' grouped-GEMM parameters names the containers through their parents too, so a plan that
# styles some containers one way and others the other names them all.
_EXPERTS_STYLES = ("moe_tp_experts", "nemotron_experts_ep")


def _routed_expert_keys(ep_plan: dict[str, str]) -> set[str]:
    """The keys of ``ep_plan``, a ``base_model_ep_plan``, that name a routed-experts container."""
    return {key for key, style in ep_plan.items() if style in _EXPERTS_STYLES} | {
        key.rsplit(".", 1)[0] for key, style in ep_plan.items() if style == "grouped_gemm"
    }


def _routed_expert_modules(model: "nn.Module", ep_plan: dict[str, str]) -> "dict[str, nn.Module]":
    """``model``'s routed-expert containers, ``{name: module}``, as its ``base_model_ep_plan`` names them.

    A module is one when its name, with layer indices wildcarded as transformers does, is a container
    key or ends with one: the keys are relative to the base model, so they omit the ``model.`` or
    ``language_model.model.`` a causal LM or a VLM puts in front. Deciding a model's distribution and
    loading it expert-parallel both find the experts this way, so the two cannot disagree.
    """
    # Local: transformers is the optional [hf] extra, and this module must import without it.
    from transformers.integrations.tensor_parallel import replace_layer_number_by_wildcard

    keys = _routed_expert_keys(ep_plan)
    found = {}
    for name, module in model.named_modules():
        generic = replace_layer_number_by_wildcard(name)
        if any(generic == key or generic.endswith("." + key) for key in keys):
            found[name] = module
    return found


def _ep_spec(model_type: str):
    """``model_type``'s :class:`~modelopt.torch.models.EPSpec`, or ``None``."""
    # Imported here: importing modelopt.torch.models registers every model spec, which importing
    # this module should not do.
    from modelopt.torch.models import get_spec

    return getattr(get_spec(model_type), "ep_spec", None)


def _ep_floor(model_type: str) -> str:
    """The transformers release ``model_type``'s expert parallelism needs.

    The expert-parallel load's own, or its spec's when that is later. A model whose spec states no
    release, or that has no spec, still needs the load's: auto must pick no EP degree below it,
    rather than reach the load and fail there.
    """
    ep = _ep_spec(model_type)
    stated = ep.requires_transformers if ep is not None else None
    return max(_EP_MIN_TRANSFORMERS, stated or _EP_MIN_TRANSFORMERS, key=parse)


def _fsdp2_way_out(fsdp: str = "MODELOPT_FSDP_SIZE=world") -> str:
    """FSDP2 as a refusal names it, the way around expert parallelism ``fsdp`` asks for."""
    # The FSDP2 load builds a causal LM and reads its tensors as stored, so a quantized or
    # multimodal checkpoint (GPT-OSS's MXFP4, Llama-4's) needs a single process, as does one with
    # an MTP head the load cannot build (--use_fsdp2 refuses Nemotron-H's).
    return (
        f"(--use_fsdp2 in examples/hf_ptq/hf_ptq.py, or {fsdp}) if the FSDP2 load supports the "
        "checkpoint, which takes an unquantized causal LM with no MTP head it cannot build (such "
        "as Nemotron-H's)"
    )


def check_distributed_support(model_type: str) -> None:
    """Refuse ``model_type``'s expert parallelism if its spec does, or below its transformers floor.

    The floor is the expert-parallel load's own, or its spec's release when that is later; a model
    with no spec or no EP section still needs the load's.
    :func:`apply_distributed_support` checks this before it prepares a config; a config already
    prepared with its plan needs only the check. An EP degree left to ``auto`` picks none for a
    model this refuses (:func:`_spec_refuses_ep`), so only an EP degree asked for explicitly, or an
    expert-parallel plan handed in, meets these errors; both name the way around expert parallelism.
    """
    ep = _ep_spec(model_type)
    without_ep = (
        f"without expert parallelism: in a single process, or FSDP2-sharded {_fsdp2_way_out()}"
    )
    if ep is not None and ep.unsupported:
        raise NotImplementedError(
            f"{model_type} cannot be quantized expert-parallel yet: {ep.unsupported} Quantize it "
            f"{without_ep}."
        )
    try:
        require_transformers(_ep_floor(model_type), f"{model_type} expert-parallel PTQ")
    except RuntimeError as e:
        raise RuntimeError(f"{e} Or quantize it {without_ep}.") from None


def _spec_refuses_ep(model_type: str) -> bool:
    """Whether :func:`check_distributed_support` refuses ``model_type``'s expert parallelism here."""
    try:
        check_distributed_support(model_type)
    except RuntimeError:  # NotImplementedError is one too
        return True
    return False


def apply_distributed_support(
    model_type: str,
    hf_config,
    *,
    ckpt_path: "str | None" = None,
    trust_remote_code: bool = False,
    prepare: bool = True,
):
    """Prepare ``hf_config`` so ``model_type`` can be loaded expert-parallel, from its model spec.

    Reads the model's :class:`~modelopt.torch.models.EPSpec` and applies it: enforce the
    transformers floor this model's expert parallelism needs, then attach the
    ``base_model_ep_plan`` it requires to whichever config carries its MoE -- the root for a plain
    MoE, the nested text/LLM config for a VLM whose experts live in its language sub-model, since
    the plan's globs are written against that module tree and a plan on the wrong config matches
    nothing at all rather than failing.

    An existing plan always wins: stock transformers ships one for some models, and overwriting it
    would substitute our idea of the layout for the model's own. It wins wherever the load reads
    one (:func:`_ep_config`), not only on the config the spec would take: on the root, or on
    another nested config.

    With ``prepare=False``, the spec's ``prepare`` does not run: the plan it would attach
    (``prepared_ep_plan``) goes on the config instead, and only what makes the checkpoint's own code
    import and build runs (``make_buildable``). That is what a plan decision reads, and it changes
    nothing the model computes or loads, as a load that is not expert-parallel must find the process.

    Returns the config to load, unchanged for a model type with no spec or no EP section. The plan
    is written into the config it is given, so a caller that keeps its own config passes a copy, as
    every caller here does. The generic all-to-all transport is not wired here -- it self-registers
    in :func:`~modelopt.torch.utils.plugins.model_load_utils.load_and_prepare_ep`, so nothing about
    it is per-model.
    """
    check_distributed_support(model_type)
    ep = _ep_spec(model_type)
    if ep is None:
        return hf_config

    # The spellings of the config the plan goes on; the first this config has takes it. Checked for
    # every spec, before anything is attached or prepared, whatever supplies the plan, but for a
    # config carrying a plan of its own, which wins wherever it sits and needs no nested config.
    attrs = ep.plan_config_attr or ()
    if isinstance(attrs, str):
        attrs = (attrs,)
    unread = [attr for attr in attrs if attr not in _SUB_CONFIG_ATTRS]
    if unread:
        raise ValueError(
            f"{model_type}'s spec attaches its expert-parallel plan to "
            f"config.{' or config.'.join(unread)}, where the expert-parallel load does not read "
            f"one; it reads config.{' and config.'.join(_SUB_CONFIG_ATTRS)}."
        )
    existing = getattr(_ep_config(hf_config), "base_model_ep_plan", None)
    target = hf_config
    if attrs and not existing:
        target = next(
            (sub for attr in attrs if (sub := getattr(hf_config, attr, None)) is not None), None
        )
        if target is None:
            raise ValueError(
                f"{model_type} expects its expert-parallel plan on "
                f"config.{' or config.'.join(attrs)}, which this config does not have, nor a plan "
                "of its own."
            )
    plan = ep.base_model_ep_plan or (None if prepare else ep.prepared_ep_plan)
    if plan and not existing:
        target.base_model_ep_plan = dict(plan)

    if prepare and ep.prepare is not None:
        hf_config = ep.prepare(hf_config, ckpt_path=ckpt_path, trust_remote_code=trust_remote_code)
    elif not prepare and ep.make_buildable is not None:
        ep.make_buildable(hf_config, ckpt_path=ckpt_path, trust_remote_code=trust_remote_code)
    return hf_config


def _spec_ep_plan(model_type: str) -> dict[str, str]:
    """The ``base_model_ep_plan`` :func:`apply_distributed_support` would attach for ``model_type``.

    Itself, or through the spec's ``prepare``. Empty for a model type with no spec, no
    expert-parallel section, or one whose spec refuses expert parallelism outright
    (``unsupported``). A model below its spec's transformers floor keeps its plan: its experts are
    routed all the same, so a run holding them whole on every rank is refused rather than
    replicated. Only reads the spec: nothing is applied, checked or patched, and ``prepare`` does
    not run.
    """
    ep = _ep_spec(model_type)
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
    exists (transformers >= 5.6); ``grouped_gemm`` (expert weight sharding) is stock too. A model
    builds with ``torch_a2a`` only from 5.7, the floor ``load_and_prepare_ep`` enforces: 5.6 accepts
    only the experts implementations it ships.
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


def expert_parallel_mesh(model: nn.Module):
    """The mesh ``model``'s routed experts are sharded over, or ``None`` if its plan shards none.

    :func:`~modelopt.torch.utils.plugins.model_load_utils.load_and_prepare_ep` loads the model over
    its expert-parallel submesh, and keeps that submesh on the model as ``_device_mesh``.
    Calibration takes the mesh to re-install the all-to-all forwards
    (:func:`~modelopt.torch.utils.distributed.distributed_forward_loop`).
    """
    if not distribution_plan(model).is_expert_parallel:
        return None
    mesh = getattr(model, "_device_mesh", None)
    if mesh is None:
        raise RuntimeError(
            f"{type(model).__name__} is planned expert-parallel but carries no device mesh. Pass "
            "the model load_and_prepare_ep returned, not a submodule of it: the loader keeps the "
            "mesh it shards the experts over on that model alone."
        )
    return mesh
