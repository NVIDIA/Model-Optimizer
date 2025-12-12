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

"""Provides hooks for capturing the inputs and the outputs of pytorch modules that are used for
activation scoring for pruning.
"""

import gc
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

import modelopt.torch.utils.distributed as dist

# BlockConfig used at runtime, not just type hints (lines 680, 790)
from modelopt.torch._compress.decilm.deci_lm_hf_code.block_config import BlockConfig  # noqa: TC001
from modelopt.torch._compress.decilm.deci_lm_hf_code.configuration_decilm import (
    DeciLMConfig,  # noqa: TC001
)
from modelopt.torch._compress.decilm.deci_lm_hf_code.modeling_decilm import DeciLMRMSNorm
from modelopt.torch._compress.tools.logger import aprint
from modelopt.torch._compress.tools.robust_json import json_dump


def clear_gpu_memory(clear: bool) -> None:
    if clear:
        gc.collect()
        torch.cuda.empty_cache()


class ActivationsHook(ABC):
    @abstractmethod
    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        A hook to be registered in pytorch modules: torch.nn.Module.register_forward_hook()

        Args:
            module (nn.Module):
            args (tuple[torch.Tensor]): Input of the pytorch module
            output (torch.Tensor): Output of the pytorch module
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, torch.Tensor]: ...

    def save_state(self) -> dict:
        """
        Save the internal state of the hook for checkpointing.

        Returns:
            dict: State dictionary that can be used to restore the hook's state
        """
        # Default implementation - hooks should override this if they have state to save
        return {}

    def load_state(self, state_dict: dict) -> None:
        """
        Load the internal state of the hook from a checkpoint.

        Args:
            state_dict: State dictionary previously returned by save_state()
        """
        # Default implementation - hooks should override this if they have state to load

    def get_progress_info(self) -> dict:
        """
        Get progress information for this hook (e.g., current iteration, samples processed).

        Returns:
            dict: Progress information
        """
        # Default implementation - hooks can override to provide progress info
        return {}

    @classmethod
    def dump_activations_logs(
        cls: type["ActivationsHook"],
        activation_hooks: dict[str, "ActivationsHook"],
        activations_log_dir: Path | str,
        args: DictConfig,
    ) -> None:
        """
        Default implementation for dumping final activation scores logs to disk.
        This is called only at the end of scoring to save final results.
        """

        activations_log_dir = Path(activations_log_dir)
        activations_log_dir.mkdir(exist_ok=True, parents=True)
        rank = dist.rank()
        activations_log_path = activations_log_dir / f"rank_{rank}.pth"
        activations_log = {
            module_name: hook.to_dict() for module_name, hook in activation_hooks.items()
        }
        torch.save(activations_log, activations_log_path)

        if rank == 0:
            args.activation_hooks_kwargs.pop("model")
            json_dump(OmegaConf.to_container(args, resolve=True), activations_log_dir / "args.json")
        dist.barrier()

        aprint(f"Dumped final activations log to {activations_log_path}")

    @classmethod
    def save_hook_states(
        cls: type["ActivationsHook"],
        activation_hooks: dict[str, "ActivationsHook"],
        activations_log_dir: Path | str,
    ) -> None:
        """
        Save hook states for checkpointing (separate from final results).
        This can be called periodically during scoring.
        Note: Synchronization should be handled at a higher level to avoid deadlocks.
        """
        activations_log_dir = Path(activations_log_dir)
        activations_log_dir.mkdir(exist_ok=True, parents=True)
        rank = dist.rank()

        hook_states_path = activations_log_dir / f"hook_states_rank_{rank}.pth"
        hook_states = {
            module_name: hook.save_state() for module_name, hook in activation_hooks.items()
        }
        torch.save(hook_states, hook_states_path)

        return hook_states_path


class IndependentChannelContributionHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        weight_matrix = linear_layer.weight.float()
        self.weight_norm = torch.linalg.vector_norm(weight_matrix, dim=0)
        num_channels = linear_layer.in_features
        self.agg_channel_activations = torch.zeros(
            size=(num_channels,), dtype=torch.float32, device=weight_matrix.device
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        activations = args[0]
        mean_abs_channel_activations = (
            activations.abs().float().mean(dim=list(range(activations.ndim - 1)))
        )
        self.agg_channel_activations[:] += mean_abs_channel_activations  # shape [I]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": (self.weight_norm * self.agg_channel_activations).cpu(),
            "weight_norm": self.weight_norm.cpu(),
            "agg_channel_activations": self.agg_channel_activations.cpu(),
        }

    def save_state(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "agg_channel_activations": self.agg_channel_activations.cpu().clone(),
            "weight_norm": self.weight_norm.cpu().clone(),
        }

    def load_state(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.agg_channel_activations = state_dict["agg_channel_activations"].to(
            self.agg_channel_activations.device
        )
        # weight_norm should be the same as it's derived from the model weights
        # but we can verify it matches
        expected_weight_norm = state_dict["weight_norm"].to(self.weight_norm.device)
        if not torch.allclose(self.weight_norm, expected_weight_norm, rtol=1e-5):
            print(
                "Warning: weight_norm mismatch during state loading - model weights may have changed"
            )


def get_pruning_schedule(num_channels, pruning_iters):
    """
    Spending decreases monotonically when num_channels >= pruning_iters.
    Intervals between spends increase monotonically when pruning_iters > num_channels.
    The budget is fully utilized, and there's spending in the last iteration.
    num_channels = 10, pruning_iters = 4 ==> [3, 3, 2, 2]
    num_channels = 4, pruning_iters = 10 ==> [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    """
    if num_channels >= pruning_iters:
        # Case when budget is greater than or equal to iterations
        q = num_channels // pruning_iters  # Base spend per iteration
        r = num_channels % pruning_iters  # Remainder to distribute

        schedule = []
        for i in range(pruning_iters):
            if i < r:
                # Assign higher spend to earlier iterations
                schedule.append(q + 1)
            else:
                schedule.append(q)
    else:
        # Case when iterations are greater than budget
        schedule = [0] * pruning_iters
        for i in range(1, num_channels + 1):
            # Distribute spends at positions where intervals increase monotonically
            pos = ((i * pruning_iters) // num_channels) - 1
            schedule[pos] = 1
    return schedule


class IterativeChannelContributionHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        """TODO: Add docstring.

        Args:
            linear_layer: The linear projection layer
            activation_hooks_kwargs: The activation hooks kwargs
        """
        self.weight_matrix = linear_layer.weight
        self.num_channels = linear_layer.in_features
        self.pruning_iters = activation_hooks_kwargs["validation_full_iters"]
        self.clear_gpu_memory = activation_hooks_kwargs.get("clear_gpu_memory", False)
        self.curr_iter = 0
        self.pruning_schedule = get_pruning_schedule(
            num_channels=self.num_channels, pruning_iters=self.pruning_iters
        )

        self.agg_cont_per_channel = torch.zeros(
            size=(self.num_channels,),
            dtype=torch.float32,
            device=self.weight_matrix.device,
        )
        self.pruned_channels = []
        self.calibration_method = activation_hooks_kwargs.get("calibration_method")
        self.epsilon = 1e-8

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        activations = args[0]
        n_channels_to_prune = self.pruning_schedule[self.curr_iter]

        curr_activations = activations.clone()  # Shape B,T,I
        curr_activations[..., self.pruned_channels] = 0
        output_curr = F.linear(input=curr_activations, weight=self.weight_matrix)  # Shape B,T,E

        if self.calibration_method is None:
            scaling_factor_per_token = torch.ones_like(output[..., 0])  # Shape B,T
        elif self.calibration_method == "scale_by_magnitude":
            output_norms = torch.linalg.vector_norm(output, dim=-1)  # Shape B,T
            output_curr_norms = torch.linalg.vector_norm(output_curr, dim=-1)  # Shape B,T
            scaling_factor_per_token = output_curr_norms / (output_norms + self.epsilon)
            del output_curr_norms, output_norms
        else:
            raise NotImplementedError
        del curr_activations
        clear_gpu_memory(clear=self.clear_gpu_memory)

        s = scaling_factor_per_token.unsqueeze(-1) * output - output_curr  # Shape: (B, T, E)
        s_squared_per_token = torch.sum(s**2, dim=-1)  # Shape: (B, T)
        b = s @ self.weight_matrix  # Shape: (B, T, I)
        c = torch.sum(self.weight_matrix**2, dim=0)  # Shape: (I)
        del s, output_curr
        clear_gpu_memory(clear=self.clear_gpu_memory)

        contribution_squared = (
            s_squared_per_token.unsqueeze(2) + 2 * activations * b + (activations**2) * c
        )  # Shape: (B, T, I)
        del s_squared_per_token, b, c, activations
        clear_gpu_memory(clear=self.clear_gpu_memory)

        contribution = torch.sqrt(contribution_squared + self.epsilon)  # Shape: (B, T, I)
        mean_cont_per_channel = torch.mean(contribution, dim=(0, 1))  # Shape: (I)
        mean_cont_per_channel[self.pruned_channels] = torch.inf
        del contribution, contribution_squared
        clear_gpu_memory(clear=self.clear_gpu_memory)

        if n_channels_to_prune == 0:
            self.agg_cont_per_channel += mean_cont_per_channel
        else:
            _, worst_indices = torch.topk(mean_cont_per_channel, n_channels_to_prune, largest=False)
            worst_indices_list = worst_indices.tolist()
            assert not set(self.pruned_channels).intersection(set(worst_indices_list))
            self.pruned_channels.extend(worst_indices_list)
            self.agg_cont_per_channel.zero_()
        self.curr_iter += 1

    def to_dict(self) -> dict[str, torch.Tensor]:
        assert self.num_channels == len(self.pruned_channels)
        channels_importance_ascending = torch.tensor(self.pruned_channels, dtype=torch.long)
        score = torch.empty(self.num_channels, dtype=torch.long)
        score[channels_importance_ascending] = torch.arange(self.num_channels, dtype=torch.long)

        return {
            "score": score.cpu(),
            "channels_importance_ascending": channels_importance_ascending.cpu(),
        }

    def save_state(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "curr_iter": self.curr_iter,
            "pruned_channels": self.pruned_channels.copy(),
            "agg_cont_per_channel": self.agg_cont_per_channel.cpu().clone(),
            "num_channels": self.num_channels,
            "pruning_iters": self.pruning_iters,
            "pruning_schedule": self.pruning_schedule.copy(),
            "calibration_method": self.calibration_method,
            "epsilon": self.epsilon,
        }

    def load_state(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.curr_iter = state_dict["curr_iter"]
        self.pruned_channels = state_dict["pruned_channels"].copy()
        self.agg_cont_per_channel = state_dict["agg_cont_per_channel"].to(self.weight_matrix.device)
        # Verify other parameters match
        assert self.num_channels == state_dict["num_channels"], "Channel count mismatch"
        assert self.pruning_iters == state_dict["pruning_iters"], "Iteration count mismatch"
        assert self.pruning_schedule == state_dict["pruning_schedule"], "Pruning schedule mismatch"

    def get_progress_info(self) -> dict:
        """Get progress information."""
        progress = self.curr_iter / self.pruning_iters if self.pruning_iters > 0 else 0.0
        return {
            "curr_iter": self.curr_iter,
            "total_iters": self.pruning_iters,
            "progress": progress,
            "pruned_channels_count": len(self.pruned_channels),
            "total_channels": self.num_channels,
        }


class IndependentKvHeadContributionHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        """TODO: Add docstring.

        Args:
            linear_layer: The linear projection layer
            activation_hooks_kwargs: The activation hooks kwargs
        """
        model_config: DeciLMConfig = activation_hooks_kwargs["model"].config
        block_config: BlockConfig = activation_hooks_kwargs["block_config"]

        self.optimize_for = activation_hooks_kwargs.get("optimize_for", "memory")
        assert self.optimize_for in ["latency", "memory"]

        self.hidden_size = model_config.hidden_size
        self.n_heads_in_group = block_config.attention.n_heads_in_group
        self.num_q_heads = model_config.num_attention_heads
        self.num_kv_heads = self.num_q_heads // self.n_heads_in_group
        self.head_dim = getattr(model_config, "head_dim", self.hidden_size // self.num_q_heads)

        self.agg_kv_head_contributions = torch.zeros(
            size=(self.num_kv_heads,),
            dtype=torch.float32,
            device=linear_layer.weight.device,
        )

        # Reshape weight matrix to group by KV heads
        self.weight_grouped = linear_layer.weight.view(
            self.hidden_size, self.num_kv_heads, self.head_dim * self.n_heads_in_group
        ).permute((1, 0, 2))
        # weight_grouped.shape: (kv_heads, hidden_dim, head_dim * n_heads_in_group)

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module: The linear projection layer
        :param args: tuple containing attention output tensor (B, T, num_q_heads * head_dim)
        :param output: The projected output (B, T, hidden_dim)
        """
        attn_out = args[0]  # Shape: (B, T, num_q_heads * head_dim)
        batch_size, seq_len, _ = attn_out.shape

        # Reshape attention output to group by KV heads
        attn_out_grouped = attn_out.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim * self.n_heads_in_group,
        ).unsqueeze(-2)
        # attn_out_grouped.shape: (B, T, kv_heads, 1, head_dim * n_heads_in_group)

        if self.optimize_for == "latency":
            # Compute contribution per KV head group
            # First compute the projection for each KV head group
            layer_out_grouped = attn_out_grouped @ self.weight_grouped.transpose(-1, -2)
            layer_out_grouped = layer_out_grouped.squeeze(-2)
            # layer_out_grouped.shape: (B, T, kv_heads, hidden_dim)

        else:
            layer_out_grouped = []
            for i in range(self.num_kv_heads):
                _layer_out = attn_out_grouped[:, :, i] @ self.weight_grouped[i].transpose(-1, -2)
                layer_out_grouped.append(_layer_out)
            layer_out_grouped = torch.cat(layer_out_grouped, dim=2)

        # Compute L2 norm of each group's contribution
        contrib_per_kv_head = torch.linalg.vector_norm(layer_out_grouped, dim=-1)
        # contrib_per_kv_head.shape: (B, T, kv_heads)

        contrib_per_kv_head = contrib_per_kv_head.mean(dim=(0, 1))
        # contrib_per_kv_head.shape: (kv_heads,)

        # Accumulate contributions
        self.agg_kv_head_contributions += contrib_per_kv_head

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": self.agg_kv_head_contributions.cpu(),
        }


class LayerNormContributionHook(ActivationsHook):
    def __init__(self, layernorm_layer: DeciLMRMSNorm, activation_hooks_kwargs: dict):
        """Aggregates mean absolute activation values per channel for a layer normalization layer.

        Args:
            layernorm_layer: The layer normalization layer
            activation_hooks_kwargs: The activation hooks kwargs (not used)
        """
        self.agg_embedding_activations = torch.zeros(
            size=(layernorm_layer.weight.shape[0],),
            dtype=torch.float32,
            device=layernorm_layer.weight.device,
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.agg_embedding_activations += (
            output.abs().float().mean(dim=list(range(output.ndim - 1)))
        )

    @classmethod
    def dump_activations_logs(
        cls: type["LayerNormContributionHook"],
        activation_hooks: dict[str, "ActivationsHook"],
        activations_log_dir: Path | str,
        args: DictConfig,
    ) -> None:
        """
        At the end of the default implementation of dumping activation scores to disc,
        save aggregated channel importance results.
        """

        super().dump_activations_logs(activation_hooks, activations_log_dir, args)

        rank = dist.rank()
        if rank == 0:
            LayerNormContributionHook._save_channel_importance_results(
                activation_hooks, activations_log_dir, args
            )

        dist.barrier()

    @staticmethod
    def _save_channel_importance_results(
        activation_hooks: dict[str, ActivationsHook],
        activations_log_dir: Path,
        args: DictConfig,
    ) -> None:
        """
        Save channel importance results from activation hooks.
        """

        # Find all activation files (for multi-rank scenarios)
        activations_log_dir = Path(activations_log_dir)
        activation_files = list(activations_log_dir.glob("rank_*.pth"))
        if not activation_files:
            aprint(f"Warning: No activation files found in {activations_log_dir}")
            return

        # Load and aggregate activation data from all ranks
        all_scores = []
        for activation_file in activation_files:
            aprint(f"Loading activations from {activation_file}")
            activation_data = torch.load(activation_file, map_location="cpu")

            # Extract scores from the activation data
            for module_name, hook_data in activation_data.items():
                if "score" in hook_data:
                    scores = hook_data["score"]
                    all_scores.append(scores)
                    aprint(f"Loaded {len(scores)} channel scores from {module_name}")

        if not all_scores:
            aprint("Warning: No valid activation data found")
            return

        # Average scores across all ranks and modules
        avg_scores = torch.stack(all_scores).mean(dim=0)
        aprint(f"Averaged {len(all_scores)} score sets into {len(avg_scores)} channels")

        # Create channel importance ranking (descending order)
        ranked_channels = torch.argsort(avg_scores, descending=True).tolist()

        # Create output data structure
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        output_data = {
            "model_path": getattr(args, "model_name_or_path", "unknown"),
            "dataset_path": getattr(args, "dataset_path", "unknown"),
            "experiment_id": getattr(args, "experiment_id", f"experiment_{timestamp}"),
            "eval_samples": getattr(args, "eval_samples", 0),
            "micro_batch_size": getattr(args, "micro_batch_size", 0),
            "timestamp": timestamp,
            "total_channels": len(ranked_channels),
            "channel_importance_ranking": ranked_channels,
            "channel_scores": avg_scores.tolist(),
            "score_statistics": {
                "min": float(avg_scores.min()),
                "max": float(avg_scores.max()),
                "mean": float(avg_scores.mean()),
                "std": float(avg_scores.std()),
            },
        }

        # Save the output
        output_path = activations_log_dir / "channel_importance_results.json"
        aprint(f"Saving channel importance data to {output_path}")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        # Print summary statistics
        aprint("=== Channel Importance Summary ===")
        aprint(f"Total channels: {len(ranked_channels)}")
        aprint(f"Top 10 most important channels: {ranked_channels[:10]}")
        aprint(f"Bottom 10 least important channels: {ranked_channels[-10:]}")
        aprint(f"Score range: {avg_scores.min():.4f} to {avg_scores.max():.4f}")
        aprint(f"Score mean: {avg_scores.mean():.4f}")
        aprint(f"Score std: {avg_scores.std():.4f}")

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": self.agg_embedding_activations.cpu(),
            "channels_importance_ascending": self.agg_embedding_activations.sort()[1].cpu(),
        }
