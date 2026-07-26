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

"""FFN intermediate-channel scoring hooks for AutoModel.

Port of ``modelopt.torch.prune.importance_hooks.base_hooks.IndependentChannelContributionHook``.
The hook attaches to the FFN ``down_proj`` and captures its input — the FFN
intermediate activation ``[*tokens, intermediate]``. The scored axis is the
intermediate channel dim (TP-sharded under tensor parallel); tokens are summed
locally then SUM-reduced across the data-partition group.

``score = weight_norm * mean|activation|`` per intermediate channel, where
``weight_norm`` is the per-input-channel L2 norm of the ``down_proj`` weight.

Unlike the legacy hook (which accumulates a sum of per-microbatch means), this
accumulates a token sum and a token count and finalizes to a true global mean.
That is partition-invariant (identical across any number of nodes) and yields the
same channel ranking; for a single forward pass the values match the legacy hook
exactly.
"""

import os

import torch
import torch.distributed as _tdist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard

from modelopt.torch.prune.importance_hooks.base_hooks import clear_gpu_memory, get_pruning_schedule

# Set ACTIVATION_SCORE_DEBUG=N to print per-rank activation stats for the first N iterations.
_DBG_ITERS = int(os.environ.get("ACTIVATION_SCORE_DEBUG", "0"))

from ..reduction import (
    finalize_additive,
    full_weight,
    reduce_token_sum,
    to_local_with_feature_group,
)
from .base import ScoringHook

__all__ = ["FFNIndependentScorer", "FFNIterativeScorer"]


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a (possibly DTensor) tensor to its full replicated form."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _assert_token_axis_local(tensor: torch.Tensor, feature_dim: int = -1) -> None:
    """Guard the iterative scorer's correctness assumption (see class docstring).

    The iterative hook needs the full feature axis, so it uses ``full_tensor()`` which gathers
    **every** sharded mesh dim. That is only token-safe while the token (batch/seq) axis is a plain
    local chunk rather than a DTensor placement. If a future parallel layout shards the token axis
    as ``Shard(token_dim)``, ``full_tensor()`` would gather those tokens and the subsequent ``dp_cp``
    SUM-reduce in :meth:`step_iteration` would double-count them. Fail loudly instead of silently
    corrupting the importances; the fix in that case is a feature-only gather.
    """
    if not isinstance(tensor, DTensor):
        return
    fdim = feature_dim % tensor.ndim
    for placement in tensor.placements:
        if isinstance(placement, Shard) and placement.dim != fdim:
            raise RuntimeError(
                "FFN iterative scorer: activation is DTensor-sharded on a non-feature (token) axis "
                f"({placement}); full_tensor() would double-count tokens under the dp_cp reduce. "
                "Switch this layout to a feature-only gather."
            )


class FFNIndependentScorer(ScoringHook):
    """Independent (weight-norm x mean-activation) FFN intermediate-channel scorer."""

    target_type = "ffn"
    method = "independent"

    def __init__(self, module, groups, *, block_idx=None, name=None):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        self._sum_abs: torch.Tensor | None = None  # [local_intermediate], float64
        self._count: int = 0
        self._feature_group = None  # TP group sharding the intermediate axis (or None)

    def __call__(self, module, args, output):
        activation = args[0]
        local, feature_group = to_local_with_feature_group(activation, feature_dim=-1)
        self._feature_group = feature_group
        # Flatten all leading (token) dims; accumulate per-channel sum of |activation|.
        flat = self._flatten_valid_tokens(local, trailing_dims=1).abs().to(torch.float64)
        if flat.shape[0] == 0:
            return
        partial = flat.sum(dim=0)
        self._sum_abs = partial if self._sum_abs is None else self._sum_abs + partial
        self._count += flat.shape[0]

    def finalize(self) -> dict:
        assert self._sum_abs is not None, "No activations captured before finalize()."
        # Token SUM-reduce + scored-axis GATHER -> full per-intermediate-channel sum.
        full_sum = finalize_additive(
            self._sum_abs, feature_group=self._feature_group, groups=self.groups
        )
        # Total token count over the data-partition group (replicated across TP).
        count = torch.tensor(float(self._count), dtype=torch.float64, device=self._sum_abs.device)
        reduce_token_sum(count, self.groups.token_group)
        agg = (full_sum / count).float()

        weight = full_weight(self.module.weight).float()
        weight_norm = torch.linalg.vector_norm(weight, dim=0)  # per input (intermediate) channel
        score = weight_norm * agg
        return {
            "score": score,
            "weight_norm": weight_norm,
            "agg_channel_activations": agg,
        }


class FFNIterativeScorer(ScoringHook):
    """Iterative greedy FFN intermediate-channel scorer.

    Port of ``modelopt.torch.prune.importance_hooks.base_hooks.IterativeChannelContributionHook``.
    Each forward call is one pruning iteration: it measures every channel's contribution
    (conditioned on the channels pruned so far), accumulates it, and once the per-iteration
    schedule says so, removes the worst channels. After ``validation_full_iters`` calls all
    channels are ranked.

    The greedy state (``pruned_channels``, ``agg``) must stay identical on every rank, so the
    per-iteration reduction is split: ``__call__`` (which may fire several times per iteration
    under pipeline micro-batching) accumulates this rank's per-channel contribution locally,
    while :meth:`step_iteration` (driven once per batch by the recipe, in lock-step across all
    ranks) SUM-reduces it across the token group (``dp_cp``), applies the greedy topk, and
    advances. With a single data rank this reproduces the legacy hook exactly.

    Parallelism correctness (scored axis = intermediate channel ``I``; token axis = batch x seq):
      * TP: ``down_proj`` is row-parallel, so its input is ``Shard(I)`` and its output is
        ``Partial`` over TP. ``_full`` (``DTensor.full_tensor``) all-gathers ``I`` and all-reduces
        the Partial output -> every TP rank holds the identical full activation/output, so the
        per-channel ``token_sum`` is identical on all TP ranks. TP is NOT in ``dp_cp``, so
        ``step_iteration`` does not reduce over it -> no double count; the topk decision is
        identical on every TP rank.
      * DP / FSDP: data ranks hold disjoint (or replicated) token shards; ``token_sum`` and
        ``count`` are SUM-reduced over ``dp_cp`` so ``token_sum/count`` is the true global mean
        regardless of node count (replicated data gives ``k*sum / k*count`` = same mean). FSDP
        shards the weight; ``full_weight`` re-gathers it (a no-op if already unsharded). Covered
        by ``test_ffn_iterative_dp_equivalence``.
      * CP: NeMo's ring / FLA context parallel keeps the per-rank sequence chunk as a *local*
        tensor (not a DTensor seq placement) at the MLP, so ``_full`` only gathers the feature
        axis; each CP rank's local-sequence ``token_sum`` is then summed across CP via the
        ``dp_cp`` reduce (CP is part of the flattened ``dp_cp`` group). If a future CP path
        instead sharded the activation as ``Shard(seq)`` across CP, ``_full`` would gather the
        sequence and the ``dp_cp`` reduce would double-count it -- switch to a feature-only
        gather (``to_local_with_feature_group`` + ``gather_scored_axis``) in that case.
      * SP: shards the sequence across TP only *outside* the MLP; at ``down_proj`` the sequence
        is full and only ``I`` is TP-sharded (the output may be ``Shard(seq)`` across TP, which
        ``_full`` gathers -- safe because TP is not in ``dp_cp``).
      * Packed sequences: the scorer is token-flat (sums ``|contribution|`` over all token
        positions), so packing is transparent -- packed/cu_seqlens rows contribute exactly like
        unpacked tokens, matching the legacy hook.
    """

    target_type = "ffn"
    method = "iterative"

    def __init__(
        self,
        module,
        groups,
        *,
        validation_full_iters: int,
        calibration_method: str | None = None,
        clear_gpu_memory: bool = False,
        token_sample_cap: int | None = None,
        block_idx=None,
        name=None,
    ):
        super().__init__(module, groups, block_idx=block_idx, name=name)
        self.pruning_iters = validation_full_iters
        self.calibration_method = calibration_method
        self._clear = clear_gpu_memory
        self.token_sample_cap = None if token_sample_cap is None else int(token_sample_cap)
        self.epsilon = 1e-8
        self.curr_iter = 0
        self.pruned_channels: list[int] = []
        self._weight: torch.Tensor | None = None  # full [hidden, intermediate]
        self._num_channels: int | None = None
        self._agg: torch.Tensor | None = None  # [intermediate], full
        self._schedule: list[int] | None = None
        # Per-iteration accumulation (this rank, summed over its tokens/micro-batches).
        self._pending_sum: torch.Tensor | None = None  # [intermediate]
        self._pending_count: int = 0
        self._resume_agg: torch.Tensor | None = None

    def _ensure_init(self) -> None:
        if self._weight is not None:
            return
        weight = full_weight(self.module.weight)  # [hidden, intermediate]
        self._weight = weight
        self._num_channels = weight.shape[1]
        self._agg = torch.zeros(self._num_channels, dtype=torch.float32, device=weight.device)
        self._schedule = get_pruning_schedule(self._num_channels, self.pruning_iters)
        if self._resume_agg is not None:
            self._agg.copy_(self._resume_agg.to(device=weight.device, dtype=self._agg.dtype))
            self._resume_agg = None

    def __call__(self, module, args, output):
        self._ensure_init()
        _assert_token_axis_local(args[0], feature_dim=-1)

        if _DBG_ITERS > 0 and self.curr_iter < _DBG_ITERS:
            raw = args[0]
            gr = _tdist.get_rank() if _tdist.is_initialized() else 0
            is_dt = isinstance(raw, DTensor)
            pls = str(raw.placements) if is_dt else "plain"
            lshape = tuple(raw.to_local().shape) if is_dt else tuple(raw.shape)
            _pre_full = _full(raw)
            print(
                f"[DBG:call rank={gr} pp={self.groups.pp_rank} tp={self.groups.tp_rank} "
                f"tok={self.groups.token_rank}] {self.name}  iter={self.curr_iter}  "
                f"input: dtensor={is_dt} placements={pls} local={lshape} full={tuple(_pre_full.shape)}  "
                f"|act|.mean={_pre_full.abs().float().mean().item():.5f}",
                flush=True,
            )

        # Compute the contribution in float32. The per-channel weight-norm term c = sum(w^2) and the
        # final sqrt are otherwise bf16, whose ~0.2% rounding at c~O(0.1-1) randomizes the order of
        # the densely-packed near-tied channels and differs across backends' kernels -> the greedy
        # then diverges (~0.94-0.98). The independent scorer already uses a float32 weight norm,
        # which is why it matched. Recompute the reference (unpruned) output via F.linear in float32
        # (instead of the captured bf16 forward output) so it stays consistent with output_curr.
        activations = _full(args[0]).float()  # [...tokens, I] full (TP-gathered), float32
        activations = self._flatten_valid_tokens(activations, trailing_dims=1)
        if activations.shape[0] == 0:
            return
        if self.token_sample_cap is not None and activations.shape[0] > self.token_sample_cap:
            positions = torch.floor(
                (torch.arange(self.token_sample_cap, device=activations.device, dtype=torch.float64) + 0.5)
                * float(activations.shape[0])
                / float(self.token_sample_cap)
            ).long()
            activations = activations[positions]
        weight = self._weight.float()
        output_tensor = F.linear(activations, weight)  # [B, T, E] full (unpruned) reference output

        curr_activations = activations.clone()
        curr_activations[..., self.pruned_channels] = 0
        output_curr = F.linear(curr_activations, weight)

        if self.calibration_method is None:
            scaling = torch.ones_like(output_tensor[..., 0])
        elif self.calibration_method == "scale_by_magnitude":
            output_norms = torch.linalg.vector_norm(output_tensor, dim=-1)
            output_curr_norms = torch.linalg.vector_norm(output_curr, dim=-1)
            scaling = output_curr_norms / (output_norms + self.epsilon)
        else:
            raise NotImplementedError(self.calibration_method)
        del curr_activations
        clear_gpu_memory(clear=self._clear)

        s = scaling.unsqueeze(-1) * output_tensor - output_curr  # [B, T, E]
        s_squared_per_token = torch.sum(s**2, dim=-1)  # [tokens]
        b = s @ weight  # [tokens, I]
        c = torch.sum(weight**2, dim=0)  # [I]
        del s, output_curr
        clear_gpu_memory(clear=self._clear)

        contribution_squared = (
            s_squared_per_token.unsqueeze(-1) + 2 * activations * b + (activations**2) * c
        )
        del s_squared_per_token, b, c, activations
        clear_gpu_memory(clear=self._clear)

        contribution = torch.sqrt(contribution_squared + self.epsilon)  # [tokens, I]
        n_tokens = contribution.shape[0]
        # Match the legacy hook's local reduction exactly, then convert back to a token sum so
        # distributed token-group reductions remain weighted by the number of local tokens.
        mean_cont_per_channel = contribution.mean(dim=0, dtype=torch.float64)
        token_sum = mean_cont_per_channel * n_tokens  # [I], float64
        del contribution, contribution_squared, mean_cont_per_channel
        clear_gpu_memory(clear=self._clear)

        # Accumulate this rank's contribution for the current iteration (sum over tokens /
        # pipeline micro-batches); the cross-rank reduce + topk happen in step_iteration().
        self._pending_sum = (
            token_sum if self._pending_sum is None else self._pending_sum + token_sum
        )
        self._pending_count += n_tokens

        if _DBG_ITERS > 0 and self.curr_iter < _DBG_ITERS:
            gr = _tdist.get_rank() if _tdist.is_initialized() else 0
            ts = token_sum.float()
            print(
                f"[DBG:call rank={gr} pp={self.groups.pp_rank} tp={self.groups.tp_rank} "
                f"tok={self.groups.token_rank}] {self.name}  iter={self.curr_iter}  "
                f"token_sum(this micro-batch): n_tokens={n_tokens} norm={ts.norm().item():.5f} "
                f"mean={ts.mean().item():.5f} std={ts.std().item():.5f}  "
                f"top3_ch={ts.topk(3).indices.tolist()}",
                flush=True,
            )

    def step_iteration(self) -> None:
        """Reduce this iteration's contribution across the token group, then greedily prune."""
        assert self._pending_sum is not None, "step_iteration() called before any forward."
        token_sum = self._pending_sum  # float64 (accumulated in __call__)
        count = torch.tensor(
            float(self._pending_count), dtype=torch.float64, device=token_sum.device
        )

        if _DBG_ITERS > 0 and self.curr_iter < _DBG_ITERS:
            gr = _tdist.get_rank() if _tdist.is_initialized() else 0
            print(
                f"[DBG:step rank={gr} pp={self.groups.pp_rank} tp={self.groups.tp_rank} "
                f"tok={self.groups.token_rank}] {self.name}  iter={self.curr_iter}  "
                f"BEFORE reduce: pending_sum norm={token_sum.norm().item():.5f} "
                f"count={count.item():.0f}",
                flush=True,
            )

        reduce_token_sum(token_sum, self.groups.token_group)
        reduce_token_sum(count, self.groups.token_group)
        # Keep float64 through the division to preserve precision; convert to float32 before
        # the greedy state so that the topk ordering is identical to the skit path which also
        # converts float64->float32 at this point.
        mean_cont_per_channel = (token_sum / count).float()  # global mean, float32

        if _DBG_ITERS > 0 and self.curr_iter < _DBG_ITERS:
            gr = _tdist.get_rank() if _tdist.is_initialized() else 0
            finite = mean_cont_per_channel[mean_cont_per_channel.isfinite()]
            n_to_prune = self._schedule[self.curr_iter]
            print(
                f"[DBG:step rank={gr} pp={self.groups.pp_rank} tp={self.groups.tp_rank} "
                f"tok={self.groups.token_rank}] {self.name}  iter={self.curr_iter}  "
                f"AFTER reduce: count={count.item():.0f} "
                f"mean_cont mean={finite.mean().item():.5f} std={finite.std().item():.5f} "
                f"norm={finite.norm().item():.5f}  n_to_prune={n_to_prune}  "
                f"bottom3_ch={mean_cont_per_channel.topk(3, largest=False).indices.tolist()}",
                flush=True,
            )

        mean_cont_per_channel[self.pruned_channels] = torch.inf

        self._agg += mean_cont_per_channel
        n_to_prune = self._schedule[self.curr_iter]
        if n_to_prune > 0:
            _, worst_indices = torch.topk(self._agg, n_to_prune, largest=False)
            worst_list = worst_indices.tolist()
            assert not set(self.pruned_channels).intersection(worst_list)
            self.pruned_channels.extend(worst_list)
            self._agg.zero_()
        self.curr_iter += 1
        self._pending_sum = None
        self._pending_count = 0

    def finalize(self) -> dict:
        assert self.curr_iter == self.pruning_iters, (
            f"iterative scoring ran {self.curr_iter}/{self.pruning_iters} iterations; the number "
            "of calibration batches (eval_samples // micro_batch_size) must equal "
            "validation_full_iters."
        )
        assert self._num_channels == len(self.pruned_channels)
        importance_ascending = torch.tensor(self.pruned_channels, dtype=torch.long)
        score = torch.empty(self._num_channels, dtype=torch.long)
        score[importance_ascending] = torch.arange(self._num_channels, dtype=torch.long)
        return {"score": score, "channels_importance_ascending": importance_ascending}

    def checkpoint_state(self) -> dict:
        return {
            "curr_iter": self.curr_iter,
            "pruned_channels": list(self.pruned_channels),
            "agg": None if self._agg is None else self._agg.detach().cpu(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.curr_iter = int(state["curr_iter"])
        self.pruned_channels = [int(value) for value in state["pruned_channels"]]
        # ``_ensure_init`` normally derives this on the next forward. An exact
        # checkpoint at pruning_iters has no next forward, so recover the global
        # structural width from DTensor metadata without materializing weights.
        self._num_channels = int(self.module.weight.shape[1])
        self._resume_agg = state.get("agg")
        self._pending_sum = None
        self._pending_count = 0
