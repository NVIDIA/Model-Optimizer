# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EP-native MoE activation scorers.

Only the ranking methods that survived diagnosis are implemented here:

* exact expert removal with native rerouting and weight renormalization;
* coupled iterative per-expert channel removal;
* the ordinary iterative FFN scorer for the shared expert; and
* the old activation-aware latent-MoE ASVD sufficient statistics.

Router frequency and independent per-expert channel proxies are intentionally
absent.  They do not measure the configured removal operation.
"""

from __future__ import annotations

import os
import weakref
from typing import Any

import torch
import torch.distributed as _tdist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from modelopt.torch.prune.importance_hooks.base_hooks import get_pruning_schedule

from ..reduction import full_weight, gather_scored_axis, reduce_token_sum
from .base import ScoringHook
from .ffn import FFNIterativeScorer

__all__ = [
    "MoEExpertRemovalDiffScorer",
    "MoEGroupedExpertChannelScorer",
    "MoESharedExpertChannelScorer",
    "MoELatentCalibrationScorer",
]


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _group_size(group) -> int:
    if group is None or not _tdist.is_initialized():
        return 1
    return _tdist.get_world_size(group)


def _reduce_group_sum(tensor: torch.Tensor, group):
    if _group_size(group) > 1:
        _tdist.all_reduce(tensor, op=_tdist.ReduceOp.SUM, group=group)
    return tensor


def _all_gather_varlen_dim0(tensor: torch.Tensor, group):
    """All-gather a possibly variable-length tensor along its first dimension."""
    if _group_size(group) == 1:
        return tensor
    local_len = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    gathered_len = [torch.zeros_like(local_len) for _ in range(_group_size(group))]
    _tdist.all_gather(gathered_len, local_len, group=group)
    lengths = [int(item.item()) for item in gathered_len]
    max_len = max(lengths)
    if tensor.shape[0] < max_len:
        pad_shape = (max_len - tensor.shape[0],) + tuple(tensor.shape[1:])
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)], dim=0
        )
    gathered = [torch.empty_like(tensor) for _ in lengths]
    _tdist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat([item[:length] for item, length in zip(gathered, lengths)], dim=0)


def _same_process_group(left, right) -> bool:
    if left is None or right is None or not _tdist.is_initialized():
        return left is right
    if left is right:
        return True
    try:
        return _tdist.get_process_group_ranks(left) == _tdist.get_process_group_ranks(right)
    except Exception:
        return False


def _reduce_tokens_not_already_gathered_over_ep(tensor: torch.Tensor, groups):
    """Reduce remaining token shards without summing EP-gathered tokens twice."""
    if groups.ep_inputs_replicated:
        return reduce_token_sum(tensor, groups.token_group)
    if groups.ep_shard_group is not None:
        return reduce_token_sum(tensor, groups.ep_shard_group)
    if _same_process_group(groups.token_group, groups.ep_group):
        return tensor
    return reduce_token_sum(tensor, groups.token_group)


def _gather_ep_inputs(tensor: torch.Tensor, groups):
    """Gather distinct EP inputs, or preserve one copy of replicated inputs."""
    if groups.ep_inputs_replicated:
        return tensor
    return _all_gather_varlen_dim0(tensor, groups.ep_group)


def _expert_owner_range(module, groups, num_experts: int) -> tuple[int, int]:
    del module
    ep_size = groups.ep_size or 1
    ep_rank = groups.ep_rank
    if num_experts % ep_size:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    local = num_experts // ep_size
    return ep_rank * local, (ep_rank + 1) * local


def _activation(module, gate_and_up_out: torch.Tensor, route_weight: torch.Tensor) -> torch.Tensor:
    if hasattr(module, "expert_activation_grouped"):
        return module.expert_activation_grouped(gate_and_up_out, route_weight)
    if hasattr(module, "expert_activation"):
        return module.expert_activation(gate_and_up_out, route_weight)
    return F.relu(gate_and_up_out).square() * route_weight


_LAST_GROUPED_EXPERT_CACHE: dict[str, Any] | None = None


def _grouped_cache_matches(cache: dict[str, Any] | None, module, source: torch.Tensor) -> bool:
    """Return whether ``cache`` belongs to this exact live forward input."""
    return bool(
        cache is not None
        and cache.get("module") is module
        and callable(cache.get("input_ref"))
        and cache["input_ref"]() is source
    )


def _centered_subsample(valid_ids: torch.Tensor, cap: int) -> torch.Tensor:
    if valid_ids.numel() <= cap:
        return valid_ids
    positions = torch.floor(
        (torch.arange(cap, device=valid_ids.device, dtype=torch.float64) + 0.5)
        * float(valid_ids.numel())
        / float(cap)
    ).long()
    return valid_ids[positions]


def _grouped_route_cache(
    module,
    args,
    groups,
    *,
    canonical_token_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the shared, deterministic grouped-expert route sketch."""
    global _LAST_GROUPED_EXPERT_CACHE

    source = _local(args[0])
    if _grouped_cache_matches(_LAST_GROUPED_EXPERT_CACHE, module, source):
        return _LAST_GROUPED_EXPERT_CACHE

    x = _gather_ep_inputs(source, groups)
    if x is source:
        x = source.clone()
    local_token_mask = _local(args[1]).reshape(-1).to(torch.bool)
    if canonical_token_mask is not None:
        canonical_token_mask = canonical_token_mask.to(
            device=local_token_mask.device,
            dtype=torch.bool,
        )
        if canonical_token_mask.shape != local_token_mask.shape:
            raise ValueError(
                "canonical MoE token mask does not match native routing mask: "
                f"{tuple(canonical_token_mask.shape)} != {tuple(local_token_mask.shape)}"
            )
        local_token_mask = local_token_mask & canonical_token_mask
    token_mask = _gather_ep_inputs(local_token_mask, groups)
    weights = _gather_ep_inputs(_local(args[2]).float(), groups)
    indices = _gather_ep_inputs(_local(args[3]).to(torch.long), groups)

    token_cap = max(1, int(os.environ.get("MOE_SCORING_TOKENS_PER_BATCH", "64")))
    valid_ids = _centered_subsample(torch.where(token_mask)[0], token_cap)
    selected_mask = torch.zeros_like(token_mask)
    selected_mask[valid_ids] = True
    token_mask = token_mask & selected_mask

    gate_up = _local(module.gate_and_up_projs).to(x.dtype)
    down = _local(module.down_projs)
    local_experts = int(gate_up.shape[0])
    global_experts = int(getattr(module, "n_routed_experts", local_experts))
    local_start, local_end = _expert_owner_range(module, groups, global_experts)

    num_tokens, topk = indices.shape
    flat_indices = indices.masked_fill(~token_mask[:, None], -1).reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_token_ids = (
        torch.arange(num_tokens, device=indices.device)[:, None].expand(-1, topk).reshape(-1)
    )
    local_mask = (flat_indices >= local_start) & (flat_indices < local_end)
    local_ids = flat_indices[local_mask] - local_start
    token_ids = flat_token_ids[local_mask]
    route_weights = flat_weights[local_mask]
    order = local_ids.argsort(stable=True)
    local_ids = local_ids[order]
    token_ids = token_ids[order]
    route_weights = route_weights[order]
    counts = torch.bincount(local_ids, minlength=local_experts)
    offsets = counts.cumsum(dim=0).to(torch.int32)

    if token_ids.numel():
        gate_out = torch._grouped_mm(x[token_ids], gate_up, offs=offsets)
        gate_bias = getattr(module, "gate_up_proj_bias", None)
        if gate_bias is not None:
            gate_out = gate_out + _local(gate_bias).to(gate_out.dtype)[local_ids]
        unweighted = _activation(module, gate_out, torch.ones_like(route_weights[:, None]))
    else:
        inter = int(down.shape[-2])
        unweighted = x.new_empty((0, inter))

    _LAST_GROUPED_EXPERT_CACHE = {
        "module": module,
        "input_ref": weakref.ref(source),
        "x": x,
        "token_mask": token_mask,
        "weights": weights,
        "indices": indices,
        "token_ids": token_ids,
        "local_ids": local_ids,
        "route_weights": route_weights,
        "counts": counts,
        "offsets": offsets,
        "unweighted": unweighted,
        "gate_up": gate_up,
        "down": down.to(unweighted.dtype),
        "gate_bias": (
            _local(module.gate_up_proj_bias).to(unweighted.dtype)
            if getattr(module, "gate_up_proj_bias", None) is not None
            else None
        ),
        "down_bias": (
            _local(module.down_proj_bias).to(unweighted.dtype)
            if getattr(module, "down_proj_bias", None) is not None
            else None
        ),
        "local_start": local_start,
        "local_end": local_end,
        "num_tokens": int(token_mask.sum().item()),
        "num_all_tokens": int(num_tokens),
        "ep_group": groups.ep_group,
    }
    return _LAST_GROUPED_EXPERT_CACHE


def _grouped_down(cache: dict[str, Any], activation: torch.Tensor, *, include_bias: bool) -> torch.Tensor:
    if activation.shape[0] == 0:
        return activation.new_empty((0, cache["down"].shape[-1]))
    output = torch._grouped_mm(activation, cache["down"], offs=cache["offsets"])
    if include_bias and cache["down_bias"] is not None:
        output = output + cache["down_bias"][cache["local_ids"]]
    return output


def _gate_score_state(gate, x: torch.Tensor) -> dict[str, Any]:
    """Compute the exact native gate score tensors before top-k selection."""
    gate_precision = getattr(gate, "gate_precision", None)
    x_compute = x.to(gate_precision) if gate_precision is not None else x
    weight = _full(gate.weight).to(device=x.device, dtype=gate_precision or x.dtype)
    bias = getattr(gate, "bias", None)
    bias = (
        _full(bias).to(device=x.device, dtype=gate_precision or x.dtype)
        if bias is not None
        else None
    )
    raw = F.linear(x_compute, weight, bias)
    score_func = str(getattr(gate, "score_func", "sigmoid"))
    e_bias = getattr(gate, "e_score_correction_bias", None)
    e_bias = _full(e_bias).to(raw.device) if e_bias is not None else None

    if score_func == "softmax":
        if bool(getattr(gate, "softmax_before_topk", False)):
            original = raw.softmax(dim=-1, dtype=gate_precision or torch.float32)
            choice = original
            weight_mode = "gather"
        else:
            original = raw
            choice = raw
            weight_mode = "selected_softmax"
    elif score_func == "softmax_with_bias":
        original = raw.softmax(dim=-1, dtype=gate_precision or torch.float32)
        choice = original + e_bias if e_bias is not None else original
        weight_mode = "gather"
    elif score_func == "sqrtsoftplus":
        original = torch.sqrt(F.softplus(raw.float()))
        choice = original + e_bias if e_bias is not None else original
        weight_mode = "gather"
    else:
        original = raw.sigmoid()
        choice = original + e_bias if e_bias is not None else original
        weight_mode = "gather"

    return {
        "original": original,
        "choice": choice,
        "score_func": score_func,
        "weight_mode": weight_mode,
        "has_correction_bias": e_bias is not None,
    }


def _route_from_score_state(gate, state: dict[str, Any], disabled: torch.Tensor):
    """Apply native group choice, top-k, normalization, and scale after disabling experts."""
    choice = state["choice"].clone()
    choice.scatter_(1, disabled[:, None], float("-inf"))
    score_func = state["score_func"]
    n_groups = int(getattr(gate, "n_groups", 1) or 1)
    topk_groups = int(getattr(gate, "topk_groups", n_groups) or n_groups)

    if n_groups > 1 and score_func not in ("softmax", "sqrtsoftplus"):
        grouped = choice.view(choice.shape[0], n_groups, -1)
        if score_func in ("softmax_with_bias", "sigmoid_with_bias") or state[
            "has_correction_bias"
        ]:
            group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1)[0].sum(dim=-1)
        else:
            group_scores = grouped.amax(dim=-1)
        group_idx = group_scores.topk(min(topk_groups, n_groups), dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_idx, True)
        if score_func == "sigmoid_with_bias":
            choice = grouped.masked_fill(~group_mask.unsqueeze(-1), float("-inf")).flatten(1)
        else:
            # Matches native softmax_with_bias/plain-sigmoid group masking.
            choice = (grouped * group_mask.unsqueeze(-1)).flatten(1)
        choice.scatter_(1, disabled[:, None], float("-inf"))

    topk = int(getattr(gate, "topk", getattr(gate, "n_activated_experts", 1)))
    indices = torch.topk(choice, k=topk, dim=-1, sorted=False).indices
    if state["weight_mode"] == "selected_softmax":
        weights = state["original"].gather(1, indices).softmax(
            dim=1, dtype=getattr(gate, "gate_precision", None) or torch.float32
        )
    else:
        weights = state["original"].gather(1, indices)
    if bool(getattr(gate, "norm_topk_prob", False)) and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * float(getattr(gate, "route_scale", 1.0))
    return weights.float(), indices


def _evaluate_union_pairs(
    module, cache: dict[str, Any], token_ids: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate each unique expert-token pair once on its owning EP rank."""
    keys = torch.unique(token_ids.to(torch.long) * num_experts + expert_ids.to(torch.long))
    union_tokens = torch.div(keys, num_experts, rounding_mode="floor")
    union_experts = keys.remainder(num_experts)
    owned = (union_experts >= cache["local_start"]) & (union_experts < cache["local_end"])
    local_tokens = union_tokens[owned]
    local_global_experts = union_experts[owned]
    local_ids = local_global_experts - cache["local_start"]
    order = local_ids.argsort(stable=True)
    local_ids = local_ids[order]
    local_tokens = local_tokens[order]
    local_global_experts = local_global_experts[order]

    if local_ids.numel():
        counts = torch.bincount(local_ids, minlength=cache["local_end"] - cache["local_start"])
        offsets = counts.cumsum(dim=0).to(torch.int32)
        gate_out = torch._grouped_mm(cache["x"][local_tokens], cache["gate_up"], offs=offsets)
        if cache["gate_bias"] is not None:
            gate_out = gate_out + cache["gate_bias"][local_ids]
        act = _activation(module, gate_out, torch.ones_like(local_ids[:, None], dtype=gate_out.dtype))
        output = torch._grouped_mm(act, cache["down"], offs=offsets)
        if cache["down_bias"] is not None:
            output = output + cache["down_bias"][local_ids]
    else:
        output = cache["x"].new_empty((0, cache["down"].shape[-1]))

    local_keys = local_tokens * num_experts + local_global_experts
    all_keys = _all_gather_varlen_dim0(local_keys, cache["ep_group"])
    all_outputs = _all_gather_varlen_dim0(output, cache["ep_group"])
    order = all_keys.argsort()
    return all_keys[order], all_outputs[order]


def _reconstruct_routes(
    pair_keys: torch.Tensor,
    pair_outputs: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    requested = token_ids[:, None] * num_experts + indices
    positions = torch.searchsorted(pair_keys, requested)
    if not torch.equal(pair_keys[positions], requested):
        raise RuntimeError("expert-token union is missing a routed pair")
    return (pair_outputs[positions] * weights[..., None].to(pair_outputs.dtype)).sum(dim=1)


class MoEExpertRemovalDiffScorer(ScoringHook):
    """Exact single-expert removal with native rerouting under EP."""

    target_type = "moe"
    method = "removed_expert_diff"
    checkpoint_tensor_names = ("_mse", "_cosine", "_displaced", "_denom")

    def __init__(self, module, groups, *, top_k: int, num_experts: int, block_idx=None, name=None):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        self.top_k = int(top_k)
        self.num_experts = int(num_experts)
        self.local_start, self.local_end = _expert_owner_range(
            module.experts, groups, self.num_experts
        )
        device = next(module.parameters()).device
        local = self.local_end - self.local_start
        self._mse = torch.zeros(local, dtype=torch.float64, device=device)
        self._cosine = torch.zeros(local, dtype=torch.float64, device=device)
        self._displaced = torch.zeros(local, dtype=torch.float64, device=device)
        self._denom = torch.zeros(local, dtype=torch.float64, device=device)
        self._gate_input: torch.Tensor | None = None

    def register(self):
        if self._handles:
            return self._handles[-1]
        self._register_handle(self.module.gate.register_forward_hook(self._cache_gate_input))
        return self._register_handle(self.module.experts.register_forward_hook(self._dispatch))

    def _cache_gate_input(self, module, args, output):
        if self.enabled and not ScoringHook._nested_disable_depth:
            self._gate_input = _local(args[0]).detach()

    def __call__(self, experts, args, output):
        if self._gate_input is None:
            raise RuntimeError("expert-removal scorer did not observe the matching gate input")
        canonical_token_mask = self._valid_token_mask(
            _local(args[0]),
            trailing_dims=1,
            stream="experts",
        )
        cache = _grouped_route_cache(
            experts,
            args,
            self.groups,
            canonical_token_mask=canonical_token_mask,
        )
        selected_tokens = torch.where(cache["token_mask"])[0]
        if selected_tokens.numel() == 0:
            return
        gate_x = _gather_ep_inputs(self._gate_input, self.groups)
        original_indices = cache["indices"][selected_tokens]
        original_weights = cache["weights"][selected_tokens].float()
        candidate_tokens = selected_tokens.repeat_interleave(self.top_k)
        removed = original_indices.reshape(-1)

        state = _gate_score_state(self.module.gate, gate_x[candidate_tokens])
        candidate_weights, candidate_indices = _route_from_score_state(
            self.module.gate, state, removed
        )

        pair_tokens = torch.cat(
            [
                selected_tokens[:, None].expand_as(original_indices).reshape(-1),
                candidate_tokens[:, None].expand_as(candidate_indices).reshape(-1),
            ]
        )
        pair_experts = torch.cat([original_indices.reshape(-1), candidate_indices.reshape(-1)])
        pair_keys, pair_outputs = _evaluate_union_pairs(
            experts, cache, pair_tokens, pair_experts, self.num_experts
        )
        original = _reconstruct_routes(
            pair_keys,
            pair_outputs,
            selected_tokens,
            original_weights,
            original_indices,
            self.num_experts,
        )
        candidate = _reconstruct_routes(
            pair_keys,
            pair_outputs,
            candidate_tokens,
            candidate_weights,
            candidate_indices,
            self.num_experts,
        )
        original_repeated = original.repeat_interleave(self.top_k, dim=0)

        fc2 = getattr(self.module, "fc2_latent_proj", None)
        if fc2 is not None:
            projection = full_weight(fc2.weight).to(original.dtype)
            original_metric = F.linear(original_repeated, projection)
            candidate_metric = F.linear(candidate, projection)
        else:
            original_metric = original_repeated
            candidate_metric = candidate
        mse = (candidate_metric - original_metric).float().square().mean(dim=-1).double()
        cosine = (
            1.0 - F.cosine_similarity(candidate_metric.float(), original_metric.float(), dim=-1)
        ).double()

        owned = (removed >= self.local_start) & (removed < self.local_end)
        local_removed = removed[owned] - self.local_start
        self._mse.index_add_(0, local_removed, mse[owned])
        self._cosine.index_add_(0, local_removed, cosine[owned])
        self._displaced.index_add_(
            0, local_removed, torch.ones_like(local_removed, dtype=self._displaced.dtype)
        )
        self._denom += float(selected_tokens.numel())
        self._gate_input = None

    def finalize(self) -> dict:
        mse = _reduce_tokens_not_already_gathered_over_ep(self._mse.clone(), self.groups)
        cosine = _reduce_tokens_not_already_gathered_over_ep(self._cosine.clone(), self.groups)
        displaced = _reduce_tokens_not_already_gathered_over_ep(
            self._displaced.clone(), self.groups
        )
        denom = _reduce_tokens_not_already_gathered_over_ep(
            self._denom.clone(), self.groups
        ).clamp_min(1)
        mse = gather_scored_axis(mse / denom, self.groups.ep_group, dim=0)
        cosine = gather_scored_axis(cosine / denom, self.groups.ep_group, dim=0)
        displaced = gather_scored_axis(displaced, self.groups.ep_group, dim=0)
        return {
            "score": mse.float(),
            "expert_ranks_mse": torch.argsort(torch.argsort(mse)).long(),
            "expert_ranks_cosine": torch.argsort(torch.argsort(cosine)).long(),
            "mse_diffs": mse.float(),
            "cosine_diffs": cosine.float(),
            "num_tokens_displaced": displaced.float(),
            "removal_mode": "exact_native_reroute",
        }


class MoEGroupedExpertChannelScorer(ScoringHook):
    """Coupled iterative channel ranking for all EP-owned experts at once."""

    target_type = "moe"
    method = "moe_channel"

    def __init__(
        self,
        module,
        groups,
        *,
        num_experts: int,
        validation_full_iters: int,
        block_idx=None,
        name=None,
    ):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        self.num_experts = int(num_experts)
        self.local_start, self.local_end = _expert_owner_range(module, groups, self.num_experts)
        self.local_experts = self.local_end - self.local_start
        self.intermediate = int(module.down_projs.shape[-2])
        self.pruning_iters = int(validation_full_iters)
        self.schedule = get_pruning_schedule(self.intermediate, self.pruning_iters)
        device = next(module.parameters()).device
        self.curr_iter = 0
        self._pruned = torch.zeros(
            self.local_experts, self.intermediate, dtype=torch.bool, device=device
        )
        self._agg = torch.zeros(
            self.local_experts, self.intermediate, dtype=torch.float32, device=device
        )
        self._last_score = torch.full_like(self._agg, torch.inf)
        self._prune_debt = torch.zeros(self.local_experts, dtype=torch.long, device=device)
        self._pending: torch.Tensor | None = None
        self._pending_count: torch.Tensor | None = None
        self._orders: list[list[int]] = [[] for _ in range(self.local_experts)]

    def __call__(self, module, args, output):
        canonical_token_mask = self._valid_token_mask(
            _local(args[0]),
            trailing_dims=1,
            stream="experts",
        )
        cache = _grouped_route_cache(
            module,
            args,
            self.groups,
            canonical_token_mask=canonical_token_mask,
        )
        weighted = cache["unweighted"] * cache["route_weights"][:, None].to(
            cache["unweighted"].dtype
        )
        full_routes = _grouped_down(cache, weighted, include_bias=False).float()
        keep = (~self._pruned)[cache["local_ids"]]
        current_routes = _grouped_down(
            cache, weighted * keep.to(weighted.dtype), include_bias=False
        ).float()
        residual = torch.zeros(
            cache["num_all_tokens"], full_routes.shape[-1], dtype=torch.float32, device=full_routes.device
        )
        residual.index_add_(0, cache["token_ids"], full_routes - current_routes)
        _reduce_group_sum(residual, self.groups.ep_group)

        partial = torch.zeros_like(self._agg, dtype=torch.float64)
        if cache["token_ids"].numel():
            route_residual = residual[cache["token_ids"]]
            down = cache["down"].float()
            b = torch._grouped_mm(
                route_residual, down.transpose(1, 2).contiguous(), offs=cache["offsets"]
            )
            a = weighted.float()
            c = down.square().sum(dim=-1)
            base = route_residual.square().sum(dim=-1, keepdim=True)
            contribution = (base + 2.0 * a * b + a.square() * c[cache["local_ids"]])
            contribution = contribution.clamp_min(0).add_(1e-8).sqrt().double()
            partial.index_add_(0, cache["local_ids"], contribution)
        self._pending = partial if self._pending is None else self._pending + partial
        route_count = cache["counts"].to(dtype=torch.float64)
        self._pending_count = (
            route_count if self._pending_count is None else self._pending_count + route_count
        )

    def step_iteration(self) -> None:
        if self._pending is None:
            raise RuntimeError("MoE channel iteration observed no routed tokens")
        pending = _reduce_tokens_not_already_gathered_over_ep(self._pending, self.groups)
        if self._pending_count is None:
            raise RuntimeError("MoE channel iteration is missing per-expert route counts")
        count = _reduce_tokens_not_already_gathered_over_ep(
            self._pending_count, self.groups
        )
        observed = count > 0
        score = (pending / count.clamp_min(1).unsqueeze(-1)).float()
        score[self._pruned] = torch.inf
        self._last_score[observed] = score[observed]
        self._agg[observed] += score[observed]
        n_to_prune = self.schedule[self.curr_iter]
        if n_to_prune:
            remaining = (~self._pruned).sum(dim=-1)
            self._prune_debt = torch.minimum(
                self._prune_debt + int(n_to_prune), remaining
            )
            for local_e in range(self.local_experts):
                if not bool(observed[local_e]):
                    continue
                catch_up = int(self._prune_debt[local_e].item())
                if catch_up == 0:
                    continue
                selected = [
                    int(value)
                    for value in torch.topk(
                        self._agg[local_e], catch_up, largest=False
                    ).indices.tolist()
                ]
                if any(self._pruned[local_e, value] for value in selected):
                    raise RuntimeError("coupled MoE scorer selected an already-pruned channel")
                self._orders[local_e].extend(selected)
                self._pruned[local_e, selected] = True
                self._prune_debt[local_e] = 0
                self._agg[local_e].zero_()
        self.curr_iter += 1
        self._pending = None
        self._pending_count = None

    def finalize(self) -> dict:
        if self.curr_iter != self.pruning_iters:
            raise RuntimeError(
                f"MoE channel scoring ran {self.curr_iter}/{self.pruning_iters} iterations"
            )
        local_scores = torch.empty_like(self._agg, dtype=torch.long)
        for local_e, order in enumerate(self._orders):
            # A sparse routed-token sketch may leave pruning debt when an expert is not
            # selected in the final calibration batches.  Complete that expert's order from
            # its last *observed* conditioned contribution instead of treating the missing
            # observation as zero damage.  In normal Super calibration every expert is
            # observed many times; the index fallback only covers a truly never-routed expert.
            if len(order) < self.intermediate:
                remaining = torch.where(~self._pruned[local_e])[0]
                tail_score = self._last_score[local_e, remaining]
                if not bool(torch.isfinite(tail_score).any()):
                    tail = remaining
                else:
                    tail = remaining[torch.argsort(tail_score, descending=False, stable=True)]
                order.extend(int(value) for value in tail.tolist())
            if len(order) != self.intermediate:
                raise RuntimeError(
                    f"expert {self.local_start + local_e} ranked {len(order)}/{self.intermediate} channels"
                )
            ascending = torch.tensor(order, dtype=torch.long, device=local_scores.device)
            local_scores[local_e, ascending] = torch.arange(
                self.intermediate, dtype=torch.long, device=local_scores.device
            )
        full_scores = gather_scored_axis(local_scores, self.groups.ep_group, dim=0)
        expert_stats = {
            int(expert): {
                "score": full_scores[expert],
                "channels_importance_ascending": torch.argsort(full_scores[expert]),
            }
            for expert in range(self.num_experts)
        }
        return {
            "score": full_scores,
            "expert_stats_dict": expert_stats,
            "ranking_mode": "coupled_iterative_global_residual",
        }

    def checkpoint_state(self) -> dict:
        return {
            "curr_iter": self.curr_iter,
            "pruned": self._pruned.detach().cpu(),
            "agg": self._agg.detach().cpu(),
            "last_score": self._last_score.detach().cpu(),
            "prune_debt": self._prune_debt.detach().cpu(),
            "orders": [list(order) for order in self._orders],
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.curr_iter = int(state["curr_iter"])
        self._pruned.copy_(state["pruned"].to(self._pruned.device))
        self._agg.copy_(state["agg"].to(self._agg.device))
        if state.get("last_score") is not None:
            self._last_score.copy_(state["last_score"].to(self._last_score.device))
        if state.get("prune_debt") is not None:
            self._prune_debt.copy_(state["prune_debt"].to(self._prune_debt.device))
        self._orders = [[int(value) for value in order] for order in state["orders"]]
        self._pending = None
        self._pending_count = None


class MoESharedExpertChannelScorer(FFNIterativeScorer):
    """The standard Qwen iterative FFN scorer applied to the shared expert."""

    target_type = "moe"
    method = "moe_shared_channel"


class MoELatentCalibrationScorer(ScoringHook):
    """Exact sufficient statistics for the old activation-aware latent-MoE ASVD."""

    target_type = "moe"
    method = "moe_latent"
    checkpoint_tensor_names = (
        "_latent_cov_in_sum",
        "_latent_cov_in_n",
        "_expert_weights_sum",
        "_latent_cov_out_sum",
        "_latent_cov_out_weight",
    )

    def __init__(self, module, groups, *, num_experts: int, block_idx=None, name=None):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        if getattr(module, "fc1_latent_proj", None) is None or getattr(
            module, "fc2_latent_proj", None
        ) is None:
            raise RuntimeError("moe_latent requires fc1_latent_proj and fc2_latent_proj")
        self.num_experts = int(num_experts)
        self.local_start, self.local_end = _expert_owner_range(
            module.experts, groups, self.num_experts
        )
        latent = int(module.experts.down_projs.shape[-1])
        local = self.local_end - self.local_start
        device = next(module.parameters()).device
        self._latent_cov_in_sum = torch.zeros(latent, latent, dtype=torch.float64, device=device)
        self._latent_cov_in_n = torch.zeros((), dtype=torch.float64, device=device)
        self._expert_weights_sum = torch.zeros(local, dtype=torch.float64, device=device)
        self._latent_cov_out_sum = torch.zeros(latent, latent, dtype=torch.float64, device=device)
        self._latent_cov_out_weight = torch.zeros((), dtype=torch.float64, device=device)
        self._tokens_per_batch = max(
            1, int(os.environ.get("MOE_LATENT_COV_TOKENS_PER_BATCH", "1"))
        )

    def register(self):
        if self._handles:
            return self._handles[0]
        return self._register_handle(self.module.experts.register_forward_hook(self._dispatch))

    def __call__(self, experts, args, output):
        canonical_token_mask = self._valid_token_mask(
            _local(args[0]),
            trailing_dims=1,
            stream="experts",
        )
        cache = _grouped_route_cache(
            experts,
            args,
            self.groups,
            canonical_token_mask=canonical_token_mask,
        )
        selected = _centered_subsample(torch.where(cache["token_mask"])[0], self._tokens_per_batch)
        if selected.numel() == 0:
            return

        # cache["x"] is exactly z = W_in x.  It was EP-gathered, so one EP rank
        # accumulates it and the finalize all-reduce broadcasts the exact sum.
        if self.groups.ep_rank == 0:
            z = cache["x"][selected].double()
            self._latent_cov_in_sum += z.T @ z
            self._latent_cov_in_n += z.shape[0]

        keep = torch.isin(cache["token_ids"], selected)
        if not bool(keep.any()):
            return
        local_ids = cache["local_ids"][keep]
        route_weights = cache["route_weights"][keep].double().clamp_min(0)
        self._expert_weights_sum.index_add_(0, local_ids, route_weights)

        counts = torch.bincount(local_ids, minlength=self.local_end - self.local_start)
        sampled_cache = {
            **cache,
            "local_ids": local_ids,
            "token_ids": cache["token_ids"][keep],
            "counts": counts,
            "offsets": counts.cumsum(dim=0).to(torch.int32),
        }
        # Bias is excluded to match the old equations exactly.  Super has no
        # expert bias, but keeping this explicit prevents future silent drift.
        y = _grouped_down(
            sampled_cache, cache["unweighted"][keep], include_bias=False
        ).double()
        weighted_y = y * route_weights.sqrt()[:, None]
        self._latent_cov_out_sum += weighted_y.T @ weighted_y
        self._latent_cov_out_weight += route_weights.sum()

    def finalize(self) -> dict[str, Any]:
        cov_in = _reduce_tokens_not_already_gathered_over_ep(
            self._latent_cov_in_sum.clone(), self.groups
        )
        cov_in_n = _reduce_tokens_not_already_gathered_over_ep(
            self._latent_cov_in_n.clone(), self.groups
        )
        _reduce_group_sum(cov_in, self.groups.ep_group)
        _reduce_group_sum(cov_in_n, self.groups.ep_group)

        expert_weights = _reduce_tokens_not_already_gathered_over_ep(
            self._expert_weights_sum.clone(), self.groups
        )
        expert_weights = gather_scored_axis(expert_weights, self.groups.ep_group, dim=0)

        cov_out = _reduce_tokens_not_already_gathered_over_ep(
            self._latent_cov_out_sum.clone(), self.groups
        )
        cov_out_weight = _reduce_tokens_not_already_gathered_over_ep(
            self._latent_cov_out_weight.clone(), self.groups
        )
        _reduce_group_sum(cov_out, self.groups.ep_group)
        _reduce_group_sum(cov_out_weight, self.groups.ep_group)

        return {
            "format_version": 5,
            "score": expert_weights.float(),
            "latent_cov_in": cov_in / cov_in_n.clamp_min(1.0),
            "latent_cov_in_n": cov_in_n,
            "expert_weights_sum": expert_weights,
            "latent_cov_out": cov_out / cov_out_weight.clamp_min(1e-20),
            "latent_cov_out_weight": cov_out_weight,
            "input_covariance": "exact_latent_sufficient_statistic",
            "output_covariance": "exact_route_weighted_sufficient_statistic",
            "covariance_tokens_per_batch": self._tokens_per_batch,
        }
