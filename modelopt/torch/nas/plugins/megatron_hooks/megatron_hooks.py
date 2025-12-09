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
"""Forward hooks for activation-based importance estimation (megatron NAS plugin)."""

import gc
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.layers import RowParallelLinear
from torch import nn


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


class MegatronL2NormHook(ForwardHook):
    """Hook for accumulating activation statistics for importance estimation.

    Activations are computed as mean over seq_len and then squared and summed over batch_size.
    In the accumulate() method we take the square root of the sum to get the L2 norm.

    Args:
        max_size: Optional maximum expected size to validate against (skips if mismatch).
                Useful for skipping non-max subnets during profiling.
    """

    def __init__(self, max_size: int | None = None):
        """Initialize the L2NormHook."""
        self.max_size = max_size
        self._activations: torch.Tensor | None = None

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
        # Gather input [seq_len, batch_size, hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        input_tensor = gather_from_tensor_model_parallel_region(args[0]).detach()

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

    def state_dict(self) -> dict:
        """Return the state dictionary containing activations."""
        return {"activations": self._activations}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load activations from checkpoint."""
        self._activations = state_dict["activations"]


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
        linear_layer: The linear projection layer to analyze. Can be either nn.Linear or
            RowParallelLinear from megatron.core.tensor_parallel.layers.
        activation_hooks_kwargs: Configuration dict with:
            - validation_full_iters (int): Number of pruning iterations.
            - clear_gpu_memory (bool, optional): Clear GPU memory during computation.
            - calibration_method (str, optional): "scale_by_magnitude" or None.
        max_size: Optional maximum expected size to validate against (skips if mismatch).
                Useful for skipping non-max subnets during profiling.
    """

    def __init__(
        self,
        linear_layer: nn.Linear | RowParallelLinear,
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
