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

"""Quantized convolution."""

import torch.nn as nn

from ... import tensor_quant
from .quant_module import QuantLinearConvBase, QuantModuleRegistry, _LegacyQuantLinearConvBaseMixin

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "QuantConv1d",
    "QuantConv2d",
    "QuantConv3d",
    "QuantConvTranspose1d",
    "QuantConvTranspose2d",
    "QuantConvTranspose3d",
]


@QuantModuleRegistry.register({nn.Conv1d: "nn.Conv1d"})
class _QuantConv1d(QuantLinearConvBase):
    """Quantized 1D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL


class QuantConv1d(_LegacyQuantLinearConvBaseMixin, nn.Conv1d):
    """Quantized 1D convolution."""

    default_quant_desc_weight = _QuantConv1d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.Conv2d: "nn.Conv2d"})
class _QuantConv2d(QuantLinearConvBase):
    """Quantized 2D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL


class QuantConv2d(_LegacyQuantLinearConvBaseMixin, nn.Conv2d):
    """Quantized 2D convolution."""

    default_quant_desc_weight = _QuantConv2d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.Conv3d: "nn.Conv3d"})
class _QuantConv3d(QuantLinearConvBase):
    """Quantized 3D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL


class QuantConv3d(_LegacyQuantLinearConvBaseMixin, nn.Conv3d):
    """Quantized 3D convolution."""

    default_quant_desc_weight = _QuantConv3d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.ConvTranspose1d: "nn.ConvTranspose1d"})
class _QuantConvTranspose1d(QuantLinearConvBase):
    """Quantized 1D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL


class QuantConvTranspose1d(_LegacyQuantLinearConvBaseMixin, nn.ConvTranspose1d):
    """Quantized 1D transposed convolution."""

    default_quant_desc_weight = _QuantConvTranspose1d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.ConvTranspose2d: "nn.ConvTranspose2d"})
class _QuantConvTranspose2d(QuantLinearConvBase):
    """Quantized 2D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL


class QuantConvTranspose2d(_LegacyQuantLinearConvBaseMixin, nn.ConvTranspose2d):
    """Quantized 2D transposed convolution."""

    default_quant_desc_weight = _QuantConvTranspose2d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.ConvTranspose3d: "nn.ConvTranspose3d"})
class _QuantConvTranspose3d(QuantLinearConvBase):
    """Quantized 3D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL


class QuantConvTranspose3d(_LegacyQuantLinearConvBaseMixin, nn.ConvTranspose3d):
    """Quantized 3D transposed convolution."""

    default_quant_desc_weight = _QuantConvTranspose3d.default_quant_desc_weight


# Define alias with Quant prefix
Conv1d = QuantConv1d
Conv2d = QuantConv2d
Conv3d = QuantConv3d
ConvTranspose1d = QuantConvTranspose1d
ConvTranspose2d = QuantConvTranspose2d
ConvTranspose3d = QuantConvTranspose3d
