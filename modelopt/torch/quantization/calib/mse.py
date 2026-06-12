# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Calibrator that returns the MSE amax of all collected tensors."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from .. import utils as quant_utils
from .calibrator import _Calibrator

__all__ = ["MseCalibrator", "NVFP4MSECalibrator"]


class MseCalibrator(_Calibrator):
    """Per-tensor and per-channel MSE amax search that minimizes error between x and quantized x."""

    def __init__(
        self,
        amax: torch.Tensor,
        axis: int | tuple | list | None = None,
        step_size: float = 0.1,
        start_multiplier: float = 0.25,
        stop_multiplier: float = 4.0,
        quant_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize MSE calibrator.

        Args:
            amax: Initial amax value (required).
            axis: Quantization axis. None means per-tensor quantization.
            step_size: Step size for amax search. The number of steps is computed as
                        ceil((stop_multiplier - start_multiplier) / step_size) + 1.
            start_multiplier: Starting multiplier for amax search.
            stop_multiplier: Ending multiplier for amax search.
            quant_func: Function that quantizes input tensor given an amax value.
                        Should have signature: quant_func(x, amax) -> quantized_x.
            error_func: Function to compute error between x and xq.
                        Default is F.mse_loss(x, xq, reduction='none').
        """
        super().__init__(num_bits=None, axis=axis, unsigned=None)
        self._initial_amax = amax
        self._step_size = step_size
        self._start_multiplier = start_multiplier
        self._stop_multiplier = stop_multiplier
        self._num_steps = math.ceil((stop_multiplier - start_multiplier) / step_size) + 1

        self._quant_func = quant_func
        self._error_func = error_func
        self._losses_sum: list[torch.Tensor | None] | None = None
        self._candidates: torch.Tensor | None = None
        self._amax: torch.Tensor | None = None

    def _generate_candidates(self, device: torch.device) -> torch.Tensor:
        """Generate candidate multipliers. Override in subclasses for different candidate sets."""
        return torch.linspace(
            self._start_multiplier, self._stop_multiplier, steps=self._num_steps, device=device
        )

    def _compute_candidate_amax(self, candidates: torch.Tensor) -> torch.Tensor:
        """Compute amax from candidates. Override in subclasses for different amax computation."""
        if candidates.ndim != 0:  # Called during final compute amax
            candidates = candidates.view_as(self._initial_amax)
        return self._initial_amax * candidates

    @torch.no_grad()
    def collect(self, x: torch.Tensor):
        """Collect input tensor statistics and compute losses for MSE calibration.

        Args:
            x: Input tensor.
        """
        if self._quant_func is None:
            raise RuntimeError("Quantization function not set.")

        x = x.detach().to(dtype=torch.float32)
        device = x.device

        candidates = self._generate_candidates(device)
        if self._candidates is None:
            self._candidates = candidates
            self._num_steps = len(candidates)
            self._losses_sum = [None] * self._num_steps

        assert self._losses_sum is not None
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)

        for step, candidate in enumerate(candidates):
            candidate_amax = self._compute_candidate_amax(candidate)
            xq = self._quant_func(x, candidate_amax)

            if self._error_func is not None:
                error = self._error_func(x, xq)
            else:
                error = F.mse_loss(x, xq, reduction="none")

            loss = quant_utils.reduce_sum(error, axis=reduce_axis, keepdims=False)

            if self._losses_sum[step] is None:
                self._losses_sum[step] = loss.clone()
            else:
                self._losses_sum[step] += loss

    def reset(self):
        """Reset the stored losses and amax value."""
        self._losses_sum = None
        self._candidates = None
        self._amax = None
        if self._initial_amax is not None:
            del self._initial_amax
            self._initial_amax = None

    @torch.no_grad()
    def compute_amax(self, verbose: bool = False):
        """Return the amax value that minimizes quantization error.

        Args:
            verbose: If True, print the ratio of best_amax to initial_amax.
        """
        if self._losses_sum is None or not any(loss is not None for loss in self._losses_sum):
            return None

        first_loss = next((loss for loss in self._losses_sum if loss is not None), None)
        if first_loss is None:
            return None

        # Stack losses: [num_steps] or [num_steps, num_channels]
        losses = []
        for step in range(self._num_steps):
            if self._losses_sum[step] is not None:
                losses.append(self._losses_sum[step])
            elif first_loss.ndim == 0:
                losses.append(torch.tensor(float("inf"), device=first_loss.device))
            else:
                losses.append(torch.full_like(first_loss, float("inf")))

        losses = torch.stack(losses)
        best_indices = torch.argmin(losses, dim=0)
        assert self._candidates is not None
        best_candidates = self._candidates[best_indices]
        self._amax = self._compute_candidate_amax(best_candidates)

        if verbose:
            ratio = self._amax / self._initial_amax
            if ratio.ndim == 0:
                print(f"MSE Calibrator: best_amax/initial_amax ratio = {ratio.item():.4f}")
            else:
                print(
                    f"MSE Calibrator: best_amax/initial_amax ratio - "
                    f"mean: {ratio.mean().item():.4f}, "
                    f"min: {ratio.min().item():.4f}, "
                    f"max: {ratio.max().item():.4f}"
                )

        return self._amax


class NVFP4MSECalibrator(MseCalibrator):
    """Per-block FP8 scale sweep calibrator for NVFP4 static quantization."""

    def __init__(
        self,
        amax: torch.Tensor,  # per_block_amax shape [num_blocks]
        global_amax: torch.Tensor,  # scalar
        axis: int | tuple | list | None = None,
        quant_func: Callable | None = None,
        error_func: Callable | None = None,
    ):
        """Initialize NVFP4 MSE calibrator with per-block and global amax."""
        super().__init__(amax=amax, axis=axis, quant_func=quant_func, error_func=error_func)
        self._global_amax = global_amax

    def _compute_candidate_amax(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 0:  # Called during final compute amax
            candidates = candidates.view_as(self._initial_amax)
        return torch.ones_like(self._initial_amax) * self._global_amax * candidates

    def _generate_candidates(self, device: torch.device) -> torch.Tensor:
        """Generate 126 valid FP8 E4M3 scale candidates."""
        uint8_values = torch.arange(0, 128, dtype=torch.uint8, device=device)
        fp8_values = uint8_values.view(torch.float8_e4m3fn).float()
        valid_mask = torch.isfinite(fp8_values) & (fp8_values > 0)
        fp8_values = fp8_values[valid_mask]
        return fp8_values / 448.0
