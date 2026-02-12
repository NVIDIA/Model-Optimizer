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

"""Some supportive functions."""

import warnings

import torch
from torch.autograd import Function

from .. import utils


class ClipFunction(Function):
    """An universal tensor clip function.

    Pytorch's clamp() only supports scalar range and doesn't support broadcast. This implementation uses min/max which
    is more general. The gradient is defined according to IBM's PACT paper https://arxiv.org/abs/1805.06085, which is
    also the behavior of Tensorflow's clip_by_value()
    """

    @staticmethod
    def forward(ctx, input, clip_value_min, clip_value_max):
        """Forward pass for the clip function."""
        output = torch.min(input, clip_value_max)
        output = torch.max(output, clip_value_min)
        ctx.save_for_backward(input, clip_value_min, clip_value_max)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for the clip function."""
        input, clip_value_min, clip_value_max = ctx.saved_tensors
        min_mask = (input > clip_value_min).to(grad_output.dtype)
        max_mask = (input < clip_value_max).to(grad_output.dtype)
        grad_input = grad_output * min_mask * max_mask

        if clip_value_min.requires_grad or clip_value_max.requires_grad:
            warnings.warn("Learning enabled for clip min/max. This is an experimental feature.")
        if clip_value_min.numel() != 1 or clip_value_max.numel() != 1:
            raise ValueError(
                f"Learnable min/max can only be scalar, got size {clip_value_min.size()} and {clip_value_max.size()}."
            )

        # Ensure the dtypes of min/max grads matches the input dtype
        # This might be necessary if running w/ AMP which will cast to fp32 before `sum()`
        grad_clip_value_min = (
            (grad_output * (1.0 - min_mask)).sum().to(clip_value_min.dtype)
            if clip_value_min.requires_grad
            else None
        )
        grad_clip_value_max = (
            (grad_output * (1.0 - max_mask)).sum().to(clip_value_min.dtype)
            if clip_value_max.requires_grad
            else None
        )

        return grad_input, grad_clip_value_min, grad_clip_value_max


clip = ClipFunction.apply


class FastHadamardTransform(Function):
    """The fast Hadamard transform.

    This only works for inputs.shape[-1] == power of 2.
    """

    @staticmethod
    def forward(ctx, inputs):
        """Hadamard forward."""
        assert utils.is_pow2(inputs.shape[-1]), (
            "Fast hadamard only works for inputs.shape[-1] == power of 2."
        )
        return fast_hadamard_transform.hadamard_transform(inputs)  # type: ignore[name-defined]

    @staticmethod
    def backward(ctx, grad_outputs):
        """Hadamard backward."""
        return fast_hadamard_transform.hadamard_transform(grad_outputs)  # type: ignore[name-defined]


def normalized_hadamard_transform(inputs, rotate_fp32=False):
    """Normalized fast hadamard transform."""
    global fast_hadamard_transform
    try:
        import fast_hadamard_transform
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Package `fast_hadamard_transform` not found, please install it using "
            "`pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git`"
        )

    dtype = inputs.dtype
    if rotate_fp32:
        inputs = inputs.float()
    outputs = FastHadamardTransform.apply(inputs) / torch.sqrt(
        torch.tensor(inputs.shape[-1], dtype=torch.float32)
    )
    return outputs.to(dtype) if rotate_fp32 else outputs
