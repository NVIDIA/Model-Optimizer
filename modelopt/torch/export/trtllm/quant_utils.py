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

"""Quantization helpers used only by the TensorRT-LLM checkpoint export path."""

import torch

from modelopt.torch.quantization.qtensor import NVFP4QTensor

from ..quant_format import QUANTIZATION_NONE, QUANTIZATION_NVFP4_AWQ, QUANTIZATION_NVFP4_SVDQUANT


def get_scaling_factor_from_weight(weight, group_size) -> torch.tensor:
    """Calculate the weight scaling factor for a given group size."""
    [n, k] = weight.shape

    if group_size != 0:
        # int4_awq
        if k % group_size != 0:
            raise NotImplementedError(
                "Weight shape is not divisible for block size for block quantization."
            )
        weight = weight.reshape(n, k // group_size, group_size)
        maxbound = 7.0
    else:
        # int8_sq
        maxbound = 127.0
    amax = weight.abs().max(dim=-1)[0].float()

    weights_scaling_factor = amax / maxbound

    # Let's filter the zeros in the scaling factor if the weights are zero
    # to avoid the divided-by-zero error..
    weights_scaling_factor[weights_scaling_factor == 0] = 1.0

    return weights_scaling_factor


def resmooth_and_get_scale(
    merged_weights: torch.Tensor,
    pre_quant_scales: list[torch.Tensor],
    ranks: int,
    group_size: int,
    new_pre_quant_scale: torch.Tensor | None = None,
    quantization: str | None = QUANTIZATION_NONE,
):
    """Resmooths weights from a single or multiple ranks and get scaling factors and amax.

    Args:
        merged_weights: Merged weights from ranks.
        pre_quant_scales: List of pre-quantization scales for each rank.
        ranks: Number of ranks.
        group_size: Group size of the quantization block.
        new_pre_quant_scale (optional): If not provided, weights will be resmoothed using
            the average of pre_quant_scales.

    Returns:
        weights: Resmoothed weights.
        weight_scaling_factors: Resmoothed scaling factors.
        avg_pre_quant_scale: Calculated average of the quantization scale.
    """
    if new_pre_quant_scale is None:
        new_pre_quant_scale = torch.stack(pre_quant_scales).mean(dim=0)

    assert len(pre_quant_scales) > 0 and new_pre_quant_scale.numel() == merged_weights.shape[1], (
        "Shape of pre_quant_scales and weights do not match."
    )
    weights = torch.chunk(merged_weights, ranks, dim=0)

    scales = []
    new_weights = []
    for i, p_scaling_factor in enumerate(pre_quant_scales):
        # De smooth & Re smooth
        weight = (
            weights[i]
            * p_scaling_factor.type(weights[i].dtype)
            / new_pre_quant_scale.type(weights[i].dtype)
        )
        new_weights.append(weight)
        # If NVFP4_AWQ then we view the scales as uint8 to allow for cat later
        if quantization in [QUANTIZATION_NVFP4_AWQ, QUANTIZATION_NVFP4_SVDQUANT]:
            scale, _ = NVFP4QTensor.get_weights_scaling_factor(weight, group_size).view(torch.uint8)
        else:
            scale = get_scaling_factor_from_weight(weight, group_size)
        scales.append(scale)

    resmoothed_scales = torch.cat(scales, dim=0)

    return (
        torch.cat(new_weights, dim=0),
        resmoothed_scales.view(torch.float8_e4m3fn)
        if quantization in [QUANTIZATION_NVFP4_AWQ, QUANTIZATION_NVFP4_SVDQUANT]
        else resmoothed_scales,  # if NVFP4_AWQ we view the scales back as float8_e4m3fn after cat
        new_pre_quant_scale,
    )
