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
# mypy: ignore-errors

"""MoE expert-removal and ranked-choice importance hooks (uses Puzzletron BlockConfig)."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import transformers
from packaging.version import Version
from torch import nn

from .base_hooks import ForwardHook

if TYPE_CHECKING:
    # Okay since this is only used for type hints else we should not import puzzletron here
    # as its dependencies may not be installed
    from modelopt.torch.puzzletron.block_config import BlockConfig

__all__ = [
    "NemotronHRemoveExpertsIndependentHook",
    "Qwen3VLRemoveExpertsIndependentHook",
    "RankedChoiceVotingHook",
    "RankedChoiceVotingHookNemotronH",
    "RemoveExpertsIndependentHook",
]


class RemoveExpertsIndependentHook(ForwardHook, ABC):
    """Base hook for measuring expert importance in Mixture-of-Experts models.

    This hook measures how much removing each expert affects the model output
    by comparing outputs with and without each expert.
    """

    def __init__(self, moe: nn.Module, activation_hooks_kwargs: dict):
        """Initialize the hook.

        Args:
            moe: The MoE module to analyze
            activation_hooks_kwargs: Configuration dict containing block_config
        """
        self.moe = moe
        block_config: BlockConfig = activation_hooks_kwargs["block_config"]
        self.num_local_experts = block_config.ffn.moe.num_local_experts
        self.num_experts_per_tok = block_config.ffn.moe.num_experts_per_tok
        # tensor of zeros of size num experts
        self.diffs = ["mse", "cosine"]
        some_param = next(self.moe.parameters())
        self.diffs = {
            k: torch.zeros(
                size=(self.num_local_experts,), dtype=torch.float32, device=some_param.device
            )
            for k in self.diffs
        }
        self.call_count = 0

    @abstractmethod
    def get_router_logits_and_routed_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract router logits and expert outputs for measuring expert importance.

        This method is called twice per forward pass:
        1. First call (router_logits=None): Compute original routing and expert outputs
        2. Second call (router_logits provided): Re-run with modified logits (expert disabled)

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
            router_logits: Optional pre-computed router logits. If None, compute from hidden_states.

        Returns:
            tuple of (router_logits, routed_experts):
                - router_logits: Shape (num_tokens, num_local_experts)
                - routed_experts: Shape (num_tokens, hidden_dim)
        """
        raise NotImplementedError

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook that measures expert importance."""
        hidden_states = args[0]
        router_logits, original_routed_out = self.get_router_logits_and_routed_experts(
            hidden_states
        )

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        original_routed_out = original_routed_out.view(-1, original_routed_out.shape[-1])

        _, router_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        self.call_count += 1

        for i_expert in range(self.num_local_experts):
            expert_mask = router_indices == i_expert
            is_token_routed_to_this_expert = expert_mask.any(dim=-1)

            num_tokens_displaced = is_token_routed_to_this_expert.sum()
            if num_tokens_displaced == 0:
                continue
            num_total_tokens = is_token_routed_to_this_expert.numel()

            relevant_hidden_states = hidden_states[is_token_routed_to_this_expert, :]

            router_logits_without_i = router_logits.clone()
            router_logits_without_i[..., i_expert] = -float("inf")  # disable expert i
            router_logits_without_i = router_logits_without_i[is_token_routed_to_this_expert, :]
            _, routed_out_without_i = self.get_router_logits_and_routed_experts(
                relevant_hidden_states, router_logits_without_i
            )

            relevant_tokens_original_out = original_routed_out[is_token_routed_to_this_expert, :]
            self.diffs["mse"][i_expert] += (
                nn.functional.mse_loss(
                    relevant_tokens_original_out, routed_out_without_i, reduction="mean"
                )
                * num_tokens_displaced
                / num_total_tokens
            )
            self.diffs["cosine"][i_expert] += (
                -nn.functional.cosine_similarity(
                    relevant_tokens_original_out, routed_out_without_i, dim=-1
                ).mean()
                * num_tokens_displaced
                / num_total_tokens
            )

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert accumulated statistics to dict format."""
        expert_ranks_mse = torch.argsort(self.diffs["mse"])
        expert_ranks_cosine = torch.argsort(self.diffs["cosine"])
        return {
            "expert_ranks_mse": expert_ranks_mse.cpu(),
            "expert_ranks_cosine": expert_ranks_cosine.cpu(),
            "cosine_diffs": (self.diffs["cosine"] / self.call_count).cpu(),
            "mse_diffs": (self.diffs["mse"] / self.call_count).cpu(),
        }

    def accumulate(self) -> torch.Tensor:
        """Return accumulated expert importance scores."""
        return self.diffs["mse"]

    def state_dict(self) -> dict:
        """Return the internal state for checkpointing."""
        return {
            "diffs_mse": self.diffs["mse"].cpu(),
            "diffs_cosine": self.diffs["cosine"].cpu(),
            "call_count": self.call_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.diffs["mse"] = state_dict["diffs_mse"].to(self.diffs["mse"].device)
        self.diffs["cosine"] = state_dict["diffs_cosine"].to(self.diffs["cosine"].device)
        self.call_count = state_dict["call_count"]


class NemotronHRemoveExpertsIndependentHook(RemoveExpertsIndependentHook):
    """Expert removal importance hook for NemotronH models."""

    def get_router_logits_and_routed_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract router logits and expert outputs for NemotronH MoE.

        Based on NemotronHMOE forward, uses minimum ops to get router_logits and routed_experts.
        """
        orig_shape = hidden_states.shape
        # NemotronHMOE.gate forward, copied to extract router_logits
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if router_logits is None:
            router_logits = nn.functional.linear(
                hidden_states.type(torch.float32), self.moe.gate.weight.type(torch.float32)
            )
            router_logits = router_logits.sigmoid()
            router_logits = router_logits + self.moe.gate.e_score_correction_bias.unsqueeze(0)

        topk_indices = self._get_topk_indices_without_correction_bias(router_logits)
        topk_weights = router_logits.gather(1, topk_indices)
        if self.moe.gate.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.moe.gate.routed_scaling_factor
        # Routed experts forward
        hidden_states = self.moe.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return router_logits, hidden_states

    @torch.no_grad()
    def _get_topk_indices_without_correction_bias(self, scores: torch.Tensor) -> torch.Tensor:
        """Get topk indices without correction bias.

        Same as NemotronHMOE.gate.get_topk_indices but without adding e_score_correction_bias.
        """
        group_scores = (
            scores.view(
                -1, self.moe.gate.n_group, self.moe.gate.n_routed_experts // self.moe.gate.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.moe.gate.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                -1, self.moe.gate.n_group, self.moe.gate.n_routed_experts // self.moe.gate.n_group
            )
            .reshape(-1, self.moe.gate.n_routed_experts)
        )
        scores_for_choice = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.moe.gate.top_k, dim=-1, sorted=False)[1]
        return topk_indices


class RankedChoiceVotingHook(ForwardHook):
    """Hook for ranking experts using ranked choice voting algorithm.

    This hook tracks router decisions and uses ranked choice voting to determine
    which experts are least important (can be pruned first).
    """

    def __init__(self, router: nn.Module, activation_hooks_kwargs: dict):
        """Initialize the hook.

        Args:
            router: The router module (typically nn.Linear)
            activation_hooks_kwargs: Configuration dict containing block_config
        """
        self.router_argsort: list[torch.Tensor] = []
        block_config: BlockConfig = activation_hooks_kwargs["block_config"]
        self.top_k = block_config.ffn.moe.num_experts_per_tok

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook that records router decisions.

        Args:
            module: The router module
            args: Tuple with one tensor entry (B, T, I)
            output: Router logits of shape (B, T, E)
        """
        router_logits = output[0] if isinstance(output, tuple) else output
        num_experts = router_logits.shape[-1]
        router_argsort = torch.argsort(router_logits, dim=-1, descending=True)
        router_argsort = router_argsort.view(-1, num_experts).to(torch.int16).cpu()
        self.router_argsort.append(router_argsort)

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert accumulated statistics to dict format using ranked choice voting."""
        router_argsort = torch.concat(self.router_argsort, dim=0)
        num_tokens, num_experts = router_argsort.shape

        expert_ranks = torch.full((num_experts,), -1)
        expert_counts_at_pruning_time = {}

        expert_kept_per_iteration: list[list[int]] = []
        expert_counts_per_iteration: list[dict[int, int]] = []

        for rank in range(num_experts):
            ids, counts = router_argsort[:, : self.top_k].unique(return_counts=True)
            ids = ids.tolist()
            counts = counts.tolist()
            expert_counts = dict(zip(ids, counts))

            expert_kept_per_iteration.append(ids)
            expert_counts_per_iteration.append(expert_counts)

            least_popular_expert, min_count = min(expert_counts.items(), key=lambda tup: tup[1])

            expert_ranks[least_popular_expert] = rank
            expert_counts_at_pruning_time[least_popular_expert] = min_count
            router_argsort = router_argsort[router_argsort != least_popular_expert].view(
                num_tokens, -1
            )

        zero_shot_expert_counts = torch.zeros((num_experts,), dtype=torch.long)
        for expert_id, expert_counts_val in expert_counts_per_iteration[0].items():
            zero_shot_expert_counts[expert_id] = expert_counts_val

        # Compute zero-shot expert ranks (double argsort converts counts to rank positions)
        zero_shot_expert_ranks = torch.argsort(torch.argsort(zero_shot_expert_counts))

        return {
            "expert_ranks": expert_ranks,
            "zero_shot_expert_ranks": zero_shot_expert_ranks,
            "expert_counts_at_pruning_time": expert_counts_at_pruning_time,
            "expert_counts_per_iteration": expert_counts_per_iteration,
            "top_k": self.top_k,
        }

    def accumulate(self) -> torch.Tensor:
        """Return accumulated expert ranks."""
        if not self.router_argsort:
            return torch.tensor([])
        router_argsort = torch.concat(self.router_argsort, dim=0)
        return router_argsort[:, 0].float()

    def state_dict(self) -> dict:
        """Return the internal state for checkpointing."""
        return {
            "router_argsort": [tensor.cpu().clone() for tensor in self.router_argsort],
            "top_k": self.top_k,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.router_argsort = [tensor.cpu() for tensor in state_dict["router_argsort"]]
        self.top_k = state_dict["top_k"]

    def get_progress_info(self) -> dict:
        """Get progress information."""
        return {
            "num_batches_processed": len(self.router_argsort),
            "total_tokens_processed": sum(tensor.shape[0] for tensor in self.router_argsort)
            if self.router_argsort
            else 0,
        }


class RankedChoiceVotingHookNemotronH(RankedChoiceVotingHook):
    """Ranked choice voting hook for NemotronH models.

    In NemotronH, router_logits is an internal temporary state that never leaves
    the forward() function. We reconstruct router_logits from the input hidden_states.
    """

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook that reconstructs router logits from hidden states."""
        hidden_states = args[0]
        hidden_states = hidden_states.view(-1, module.config.hidden_size)
        router_logits = nn.functional.linear(
            hidden_states.type(torch.float32), module.weight.type(torch.float32)
        )
        super().__call__(module, args, router_logits)


class Qwen3VLRemoveExpertsIndependentHook(RemoveExpertsIndependentHook):
    """Expert removal importance hook for Qwen3-VL models."""

    def get_router_logits_and_routed_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract router logits and expert outputs for Qwen3-VL MoE.

        Based on Qwen3VLMoeSparseMoe forward pass.
        """
        orig_shape = hidden_states.shape
        # Use hidden_states.shape[-1] instead of self.moe.hidden_size for transformers v5 compatibility
        hidden_size = (
            self.moe.hidden_size if hasattr(self.moe, "hidden_size") else hidden_states.shape[-1]
        )

        # Flatten to (num_tokens, hidden_size) for processing
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)

        if router_logits is None:
            router_logits = self.moe.gate(hidden_states_flat)
            # In transformers vf the gate returns (logits, aux_loss) tuple
            if isinstance(router_logits, tuple):
                router_logits = router_logits[0]

        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        if Version(transformers.__version__) >= Version("5.0"):
            # transformers 5.x: grouped_mm_experts_forward expects
            # (hidden_states_flat 2D, top_k_index, top_k_weights)
            routed_out = self.moe.experts(hidden_states_flat, router_indices, routing_weights)
        else:
            # transformers 4.x: loop-based experts expects
            # (hidden_states_3d 3D, routing_weights_full, router_indices)
            batch_size = orig_shape[0] if hidden_states.ndim == 3 else 1
            hidden_states_3d = hidden_states_flat.reshape(batch_size, -1, hidden_size)
            router_weights = torch.zeros(
                router_logits.shape, dtype=routing_weights.dtype, device=router_logits.device
            ).scatter_(1, router_indices, routing_weights)
            routed_out = self.moe.experts(hidden_states_3d, router_weights, router_indices)

        # Return in same shape as input
        routed_out = routed_out.reshape(*orig_shape)

        return router_logits, routed_out
