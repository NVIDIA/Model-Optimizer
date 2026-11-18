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

"""Calibrator that returns the bias of all collected tensors."""

import torch

from .calibrator import _Calibrator

__all__ = ["BiasCalibrator"]


def compute_maxmin(
    inputs: torch.Tensor, axis: int | tuple[int, ...] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the max and min values of input tensor.

    Args:
        inputs: Input tensor
        axis: Axis or tuple of axes to keep. Other dims are reduced.
            None: reduce all dimensions (per-tensor)
            (-1,): reduce all except last dim (per-channel)
            (-1,-3): reduce all except last and third-to-last dims (per-head per-channel)

    Returns:
        Tuple of (max_values, min_values)
    """
    if axis is None:
        max_ = torch.max(inputs)
        min_ = torch.min(inputs)
    else:
        reduce_axis = ()
        for i in range(len(inputs.shape)):
            if i in axis or (i - inputs.dim()) in axis:
                reduce_axis += (i,)

        max_ = torch.amax(inputs, dim=reduce_axis, keepdim=True)
        min_ = torch.amin(inputs, dim=reduce_axis, keepdim=True)

    return max_, min_


def compute_maxmin_bias(inputs: torch.Tensor, axis: int | tuple[int, ...] | None) -> torch.Tensor:
    """Compute the max_min mean bias of input tensor."""
    max_, min_ = compute_maxmin(inputs, axis)
    return (max_ + min_) / 2


def compute_mean_bias(inputs: torch.Tensor, axis: int | tuple[int, ...] | None) -> torch.Tensor:
    """Compute the mean bias of input tensor."""
    if axis is None:
        reduce_axis = None
        bias_ = torch.mean(inputs)
    else:
        reduce_axis = ()
        for i in range(len(inputs.shape)):
            if i in axis or (i - inputs.dim()) in axis:
                reduce_axis += (i,)

        bias_ = torch.mean(inputs, dim=reduce_axis, keepdim=True)

    return bias_


def compute_bias(
    inputs: torch.Tensor, axis: int | tuple[int, ...] | None, method: str = "mean"
) -> torch.Tensor:
    """Compute the bias of input tensor. Supports mean and max_min methods."""
    if method == "mean":
        return compute_mean_bias(inputs, axis)
    else:
        return compute_maxmin_bias(inputs, axis)


def subtract_bias(inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Subtract bias from input tensor."""
    return inputs - bias


def add_bias(inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add bias to input tensor."""
    shape = inputs.shape
    # if bias_layout == "per_block" and block_size is not None:
    # inputs = inputs.reshape(-1, block_size)
    return (inputs + bias).view(shape)


class BiasCalibrator(_Calibrator):
    """Bias calibrator, tracks the bias of all tensors collected."""

    def __init__(self, method: str = "mean", axis: int | tuple[int, ...] | None = None):
        """Initialize."""
        super().__init__(axis=axis)
        self._calib_bias = None
        self._calib_max = None
        self._calib_min = None
        self._cnt = 0
        self._method = method

    def collect(self, x: torch.Tensor):
        """Compute bias of input tensor along axis."""
        # For a 4D tensor with shape [batch, heads, seq_len, hidden_dim]:
        #   - None: reduce all dimensions (per-tensor bias)
        #   - (-1,) or (3,): keep last dimension only (per-channel bias)
        #   - (-1, -3) or (1, 3): keep last and third-to-last dimensions (per-head per-channel bias)
        #   This computes separate bias per attention head and channel, which is recommended
        #
        # Examples:
        #   tensor.shape = (8, 12, 512, 64)  # [batch, heads, seq_len, hidden]
        #   axis=None      -> single bias value for entire tensor
        #   axis=(-1,)     -> bias shape: (1, 1, 1, 64)
        #   axis=(-1, -3)  -> bias shape: (1, 12, 1, 64)

        if self._method == "mean":
            bias_ = compute_bias(x, self._axis, self._method)
            # Update running average of bias values:
            #   calib_bias = (calib_bias * count + new_bias) / (count + 1)
            #
            # Note: Alternative approach using min/max values:
            #   calib_bias = (max + min) / 2
            # However, empirical results demonstrates running average provides better accuracy for most use cases
            if self._calib_bias is None:
                self._calib_bias = bias_
            else:
                dtype = bias_.dtype
                # Convert bias to float for numerical stability
                self._calib_bias = (self._calib_bias.float() * self._cnt + bias_.float()) / (
                    self._cnt + 1
                )
                self._calib_bias = self._calib_bias.to(dtype)

            self._cnt += 1
        elif self._method == "max_min":
            max_, min_ = compute_maxmin(x, self._axis)
            # NOTE: Subtracting bias here in calibration leads to accuracy decrease
            self._calib_max = (
                torch.max(self._calib_max, max_) if self._calib_max is not None else max_
            )
            self._calib_min = (
                torch.min(self._calib_min, min_) if self._calib_min is not None else min_
            )
            self._calib_bias = (self._calib_max + self._calib_min) / 2
        else:
            raise ValueError(f"Unsupported method: {self._method}")

    def compute_bias(self):
        """Return the bias of all tensors collected."""
        return self._calib_bias

    def compute_dynamic_bias(self, inputs):
        """Compute dynamic bias based on current inputs."""
        if self._method == "mean":
            # mean = (max + min) / 2
            return compute_bias(inputs, self._axis, method="mean")
        elif self._method == "max_min":
            # mean = average(all tokens)
            return compute_bias(inputs, self._axis, method="max_min")
        else:
            raise ValueError(f"Unknown bias method: {self._method}")

    def reset(self):
        """Reset the bias calibrator."""
        self._calib_bias = None
        self._calib_max = None
        self._calib_min = None
        self._cnt = 0
