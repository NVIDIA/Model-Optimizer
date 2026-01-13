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
"""Forward hooks for activation-based importance estimation."""

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
from modelopt.torch.puzzletron.tools.logger import aprint
from modelopt.torch.puzzletron.tools.robust_json import json_dump

__all__ = [
    "ForwardHook",
    "IndependentChannelContributionHook",
    "IndependentKvHeadContributionHook",
    "IterativeChannelContributionHook",
    "L2NormHook",
    "LayerNormContributionHook",
]


def clear_gpu_memory(clear: bool) -> None:
    """Clear GPU memory cache if requested.

    Args:
        clear: If True, runs garbage collection and empties CUDA cache.
    """
    if clear:
        gc.collect()
        torch.cuda.empty_cache()


class ForwardHook(ABC):
    """Base class for PyTorch forward hooks.

    This follows the PyTorch forward hook API where the second
    parameter is 'args' (a tuple of positional arguments passed to forward()).

    Usage:
        hook = MyHook()
        module.register_forward_hook(hook)
    """

    @abstractmethod
    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook that is called after the module's forward pass.

        Args:
            module: The module this hook is registered on
            args: Tuple of positional arguments passed to module.forward()
            output: The output from module.forward()

        Returns:
            None (does not modify the output)
        """
        ...

    @abstractmethod
    def accumulate(self) -> torch.Tensor:
        """Return accumulated importance scores.

        This method should be called after all forward passes to retrieve
        the final importance scores for each channel/feature.

        Returns:
            Tensor of importance scores, one per channel/feature.

        Raises:
            AssertionError: If no activations have been collected yet.
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Return the internal state for checkpointing.

        Returns:
            dict: State dictionary containing checkpoint data.
                  Can contain tensors, ints, lists, etc.
        """
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint.

        Args:
            state_dict: State dictionary previously returned by state_dict()
        """
        ...

    def get_progress_info(self) -> dict:
        """Get progress information for this hook.

        Returns:
            dict: Progress information (e.g., current iteration, samples processed).
                  Default implementation returns empty dict.
        """
        return {}

    @abstractmethod
    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert hook results to dictionary format for saving.

        Returns:
            dict: Dictionary containing result tensors (e.g., "score", "channels_importance_ascending").
        """
        ...

    @classmethod
    def dump_activations_logs(
        cls: type["ForwardHook"],
        activation_hooks: dict[str, "ForwardHook"],
        activations_log_dir: Path | str,
        args: DictConfig,
    ) -> None:
        """Default implementation for dumping final activation scores logs to disk.

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
        cls: type["ForwardHook"],
        activation_hooks: dict[str, "ForwardHook"],
        activations_log_dir: Path | str,
    ) -> None:
        """Save hook states for checkpointing (separate from final results).

        This can be called periodically during scoring.
        Note: Synchronization should be handled at a higher level to avoid deadlocks.
        """
        activations_log_dir = Path(activations_log_dir)
        activations_log_dir.mkdir(exist_ok=True, parents=True)
        rank = dist.rank()

        hook_states_path = activations_log_dir / f"hook_states_rank_{rank}.pth"
        hook_states = {
            module_name: hook.state_dict() for module_name, hook in activation_hooks.items()
        }
        torch.save(hook_states, hook_states_path)


class L2NormHook(ForwardHook):
    """Hook for accumulating activation statistics for importance estimation.

    Activations are computed as mean over seq_len and then squared and summed over batch_size.
    In the accumulate() method we take the square root of the sum to get the L2 norm.

    This is the base version without tensor parallelism support.
    For megatron with TP > 1, use MegatronL2NormHook instead.

    Args:
        max_size: Optional maximum expected size to validate against (skips if mismatch).
                Useful for skipping non-max subnets during profiling.
    """

    def __init__(self, max_size: int | None = None):
        """Initialize the L2NormHook."""
        self.max_size = max_size
        self._activations: torch.Tensor | None = None

    def _get_input_tensor(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Get input tensor from args. Override in subclass for TP gathering."""
        return args[0].detach()

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Accumulate activation statistics from the forward pass.

        Args:
            module: The module this hook is registered on.
            args: Tuple of input tensors. args[0] expected shape: [seq_len, batch_size, hidden_size]
                  (Megatron sequence-first format).
            output: Output tensor from the module's forward pass.
        """
        input_tensor = self._get_input_tensor(args)

        if input_tensor.dim() == 2:
            # For sparse experts, there is no batch dimension.
            input_tensor = input_tensor[:, None, :]

        # Dont aggregate activations from non-max subnets (e.g. from profiling)
        if self.max_size is not None and input_tensor.shape[-1] != self.max_size:
            return

        input_tensor = input_tensor.to(torch.float32)  # use full precision to avoid overflow
        activations = input_tensor.abs().mean(dim=0)  # [batch_size, hidden_size]
        activations = activations.pow(2).sum(dim=0)  # [hidden_size]

        if self._activations is None:
            self._activations = activations
        else:
            self._activations += activations

    def accumulate(self) -> torch.Tensor:
        """Return the accumulated L2 norm of activations.

        Returns:
            Tensor of accumulated scores, one per channel

        Raises:
            AssertionError: If no activations have been collected yet
        """
        assert self._activations is not None, "No activations collected for importance estimation."
        # Convert squared sum to L2 norm
        return self._activations.pow(0.5)

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dict format for saving."""
        return {"score": self.accumulate().cpu()}

    def state_dict(self) -> dict:
        """Return the state dictionary containing activations."""
        return {"activations": self._activations}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load activations from checkpoint."""
        self._activations = state_dict["activations"]


class IndependentChannelContributionHook(ForwardHook):
    """Hook for channel importance estimation using weight norms and activation magnitudes.

    Computes channel importance as the product of:
    - L2 norm of each column in the weight matrix (how much each input channel affects output)
    - Mean absolute activation for each channel (how strongly each channel is activated)

    Args:
        linear_layer: The linear projection layer to analyze. Must have a `weight` attribute
            and either `in_features` (nn.Linear) or `input_size` (Megatron RowParallelLinear).
        max_size: Optional maximum expected size to validate against (skips if mismatch).
                Useful for skipping non-max subnets during profiling.
    """

    def __init__(
        self,
        linear_layer: nn.Module,
        max_size: int | None = None,
    ):
        """Initialize the independent channel contribution hook."""
        self.max_size = max_size

        weight_matrix = linear_layer.weight.float()
        self.weight_norm = torch.linalg.vector_norm(weight_matrix, dim=0)

        # Check if it's a RowParallelLinear (Megatron-Core) or nn.Linear (PyTorch)
        if hasattr(linear_layer, "input_size"):
            self.num_channels = linear_layer.input_size  # Megatron-Core
        else:
            self.num_channels = linear_layer.in_features  # PyTorch

        self.agg_channel_activations = torch.zeros(
            size=(self.num_channels,),
            dtype=torch.float32,
            device=weight_matrix.device,
        )

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor | tuple
    ) -> None:
        """Accumulate mean absolute activations per channel.

        Args:
            module: The module this hook is registered on.
            args: Tuple with single input tensor. args[0] expected shape: [batch_size, seq_len, input_channels]
                  (PyTorch batch-first format).
            output: Output tensor of shape [batch_size, seq_len, output_channels], or tuple (output_tensor, bias)
                    for parallel layers.
        """
        activations = args[0]

        # Don't aggregate activations from non-max subnets (e.g. from profiling)
        if self.max_size is not None and activations.shape[-1] != self.max_size:
            return

        mean_abs_channel_activations = (
            activations.abs().float().mean(dim=list(range(activations.ndim - 1)))
        )
        self.agg_channel_activations[:] += mean_abs_channel_activations  # shape [input_channels]

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert results to dict with channel importance scores.

        Returns:
            Dict with "score" (weight_norm * activations), "weight_norm", and
            "agg_channel_activations".
        """
        return {
            "score": (self.weight_norm * self.agg_channel_activations).cpu(),
            "weight_norm": self.weight_norm.cpu(),
            "agg_channel_activations": self.agg_channel_activations.cpu(),
        }

    def accumulate(self) -> torch.Tensor:
        """Return importance scores as a tensor.

        Returns:
            Tensor of importance scores (weight_norm * activations), one per channel.
        """
        return self.to_dict()["score"]

    def state_dict(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "agg_channel_activations": self.agg_channel_activations.cpu().clone(),
            "weight_norm": self.weight_norm.cpu().clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.agg_channel_activations = state_dict["agg_channel_activations"].to(
            self.agg_channel_activations.device
        )
        # weight_norm should be the same as it's derived from the model weights
        # but we can verify it matches
        expected_weight_norm = state_dict["weight_norm"].to(self.weight_norm.device)
        if not torch.allclose(self.weight_norm, expected_weight_norm, rtol=1e-5):
            raise AssertionError(
                "weight_norm mismatch during state loading - model weights may have changed"
            )


def get_pruning_schedule(num_channels, pruning_iters):
    """Spending decreases monotonically when num_channels >= pruning_iters.

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


class IterativeChannelContributionHook(ForwardHook):
    """Hook for iterative channel pruning based on contribution analysis.

    Progressively identifies and removes the least important input channels of a linear layer
    by measuring channel contribution as the L2 norm of output change when removed.

    Args:
        linear_layer: The linear projection layer to analyze. Must have a `weight` attribute
            and either `in_features` (nn.Linear) or `input_size` (Megatron RowParallelLinear).
        activation_hooks_kwargs: Configuration dict with:
            - validation_full_iters (int): Number of pruning iterations.
            - clear_gpu_memory (bool, optional): Clear GPU memory during computation.
            - calibration_method (str, optional): "scale_by_magnitude" or None.
        max_size: Optional maximum expected size to validate against (skips if mismatch).
                Useful for skipping non-max subnets during profiling.
    """

    def __init__(
        self,
        linear_layer: nn.Module,
        activation_hooks_kwargs: dict,
        max_size: int | None = None,
    ):
        """Initialize the iterative channel contribution hook."""
        self.weight_matrix = linear_layer.weight

        # Check if it's a RowParallelLinear (Megatron-Core) or nn.Linear (PyTorch)
        # TODO: Consider better design to handle RowParallelLinear and nn.Linear
        if hasattr(linear_layer, "input_size"):
            self.num_channels = linear_layer.input_size  # Megatron-Core
        else:
            self.num_channels = linear_layer.in_features  # PyTorch

        self.max_size = max_size
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

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor | tuple
    ) -> None:
        """Compute channel contributions and prune channels according to schedule.

        Args:
            module: The module this hook is registered on.
            args: Tuple with single input tensor. args[0] expected shape: [batch_size, seq_len, input_channels]
                  (PyTorch batch-first format).
            output: Output tensor of shape [batch_size, seq_len, output_channels], or tuple (output_tensor, bias)
                    for parallel layers.
        """
        # Handle case where output is a tuple (e.g., from ColumnParallelLinear/RowParallelLinear)
        # TODO: Consider better design to handle RowParallelLinear and nn.Linear
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        activations = args[0]

        # Don't aggregate activations from non-max subnets (e.g. from profiling)
        if self.max_size is not None and activations.shape[-1] != self.max_size:
            return

        n_channels_to_prune = self.pruning_schedule[self.curr_iter]

        curr_activations = activations.clone()  # Shape B,T,I
        curr_activations[..., self.pruned_channels] = 0
        output_curr = F.linear(input=curr_activations, weight=self.weight_matrix)  # Shape B,T,E

        if self.calibration_method is None:
            scaling_factor_per_token = torch.ones_like(output_tensor[..., 0])  # Shape B,T
        elif self.calibration_method == "scale_by_magnitude":
            output_norms = torch.linalg.vector_norm(output_tensor, dim=-1)  # Shape B,T
            output_curr_norms = torch.linalg.vector_norm(output_curr, dim=-1)  # Shape B,T
            scaling_factor_per_token = output_curr_norms / (output_norms + self.epsilon)
            del output_curr_norms, output_norms
        else:
            raise NotImplementedError
        del curr_activations
        clear_gpu_memory(clear=self.clear_gpu_memory)

        s = scaling_factor_per_token.unsqueeze(-1) * output_tensor - output_curr  # Shape: (B, T, E)
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

        self.agg_cont_per_channel += mean_cont_per_channel
        if n_channels_to_prune > 0:
            _, worst_indices = torch.topk(
                self.agg_cont_per_channel, n_channels_to_prune, largest=False
            )
            worst_indices_list = worst_indices.tolist()
            assert not set(self.pruned_channels).intersection(set(worst_indices_list))
            self.pruned_channels.extend(worst_indices_list)
            self.agg_cont_per_channel.zero_()
        self.curr_iter += 1

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert pruning results to dict with channel importance rankings.

        Returns:
            Dict with "score" (importance rank per channel) and
            "channels_importance_ascending" (channel indices in ascending importance).
        """
        assert self.num_channels == len(self.pruned_channels)
        channels_importance_ascending = torch.tensor(self.pruned_channels, dtype=torch.long)
        score = torch.empty(self.num_channels, dtype=torch.long)
        score[channels_importance_ascending] = torch.arange(self.num_channels, dtype=torch.long)

        return {
            "score": score.cpu(),
            "channels_importance_ascending": channels_importance_ascending.cpu(),
        }

    def accumulate(self) -> torch.Tensor:
        """Return importance scores as a tensor.

        Returns:
            Tensor of importance scores, one per channel. Lower scores indicate less important channels.
        """
        return self.to_dict()["score"]

    def state_dict(self) -> dict:
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

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.curr_iter = state_dict["curr_iter"]
        self.pruned_channels = state_dict["pruned_channels"].copy()
        self.agg_cont_per_channel = state_dict["agg_cont_per_channel"].to(self.weight_matrix.device)
        # Verify other parameters match
        assert self.num_channels == state_dict["num_channels"], "Channel count mismatch"
        assert self.pruning_iters == state_dict["pruning_iters"], "Iteration count mismatch"
        assert self.pruning_schedule == state_dict["pruning_schedule"], "Pruning schedule mismatch"

    def get_progress_info(self) -> dict:
        """Get progress information for this hook.

        Returns:
            dict: Progress information including iteration count and pruned channels.
        """
        progress = self.curr_iter / self.pruning_iters if self.pruning_iters > 0 else 0.0
        return {
            "curr_iter": self.curr_iter,
            "total_iters": self.pruning_iters,
            "progress": progress,
            "pruned_channels_count": len(self.pruned_channels),
            "total_channels": self.num_channels,
        }


class IndependentKvHeadContributionHook(ForwardHook):
    """Hook for estimating KV head importance based on contribution analysis.

    Measures the contribution of each KV head group to the output projection
    by computing L2 norms of per-head outputs.

    Args:
        linear_layer: The output projection layer (o_proj).
        activation_hooks_kwargs: Configuration dict with:
            - model: The model instance (to get config).
            - block_config: Block configuration with attention settings.
            - optimize_for (str, optional): "latency" or "memory". Defaults to "memory".
    """

    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        """Initialize the KV head contribution hook."""
        model_config = activation_hooks_kwargs["model"].config
        block_config = activation_hooks_kwargs["block_config"]

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
        """Compute KV head contributions from the forward pass."""
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

    def accumulate(self) -> torch.Tensor:
        """Return accumulated KV head importance scores.

        Returns:
            Tensor of importance scores, one per KV head.
        """
        return self.agg_kv_head_contributions

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dict format for saving.

        Returns:
            Dict with "score" tensor containing KV head importance scores.
        """
        return {
            "score": self.agg_kv_head_contributions.cpu(),
        }

    def state_dict(self) -> dict:
        """Return the internal state for checkpointing."""
        raise NotImplementedError("Saving state dict is not supported for this hook.")

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        raise NotImplementedError("Loading state dict is not supported for this hook.")


class LayerNormContributionHook(ForwardHook):
    """Hook for estimating channel importance based on layer normalization activations.

    Aggregates mean absolute activation values per channel for a layer normalization layer.

    Args:
        layernorm_layer: The layer normalization layer.
        activation_hooks_kwargs: The activation hooks kwargs (not used).
    """

    def __init__(self, layernorm_layer: nn.Module, activation_hooks_kwargs: dict):
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
        """Accumulate activation statistics from the forward pass."""
        self.agg_embedding_activations += (
            output.abs().float().mean(dim=list(range(output.ndim - 1)))
        )

    def accumulate(self) -> torch.Tensor:
        """Return accumulated channel importance scores."""
        return self.agg_embedding_activations

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dict format for saving."""
        return {
            "score": self.agg_embedding_activations.cpu(),
            "channels_importance_ascending": self.agg_embedding_activations.sort()[1].cpu(),
        }

    def state_dict(self) -> dict:
        """Return the internal state for checkpointing."""
        raise NotImplementedError("Saving state dict is not supported for this hook.")

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        raise NotImplementedError("Loading state dict is not supported for this hook.")

    @classmethod
    def dump_activations_logs(
        cls: type["LayerNormContributionHook"],
        activation_hooks: dict[str, "ForwardHook"],
        activations_log_dir: Path | str,
        args: DictConfig,
    ) -> None:
        """At the end of the default implementation of dumping activation scores to disc.

        Save aggregated channel importance results.
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
        activation_hooks: dict[str, "ForwardHook"],
        activations_log_dir: Path | str,
        args: DictConfig,
    ) -> None:
        """Save channel importance results from activation hooks."""
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
