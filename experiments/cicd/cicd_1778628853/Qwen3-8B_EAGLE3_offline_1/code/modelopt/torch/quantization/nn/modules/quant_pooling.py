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

"""Quantized Pooling modules."""

from torch.nn.modules import pooling

from .quant_module import QuantInputBase, QuantModuleRegistry, _LegacyQuantInputBaseMixin

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "QuantAdaptiveAvgPool1d",
    "QuantAdaptiveAvgPool2d",
    "QuantAdaptiveAvgPool3d",
    "QuantAvgPool1d",
    "QuantAvgPool2d",
    "QuantAvgPool3d",
    "QuantMaxPool1d",
    "QuantMaxPool2d",
    "QuantMaxPool3d",
]


class QuantMaxPool1d(_LegacyQuantInputBaseMixin, pooling.MaxPool1d):
    """Quantized version of nn.MaxPool1d."""


class QuantMaxPool2d(_LegacyQuantInputBaseMixin, pooling.MaxPool2d):
    """Quantized version of nn.MaxPool2d."""


class QuantMaxPool3d(_LegacyQuantInputBaseMixin, pooling.MaxPool3d):
    """Quantized version of nn.MaxPool3d."""


class QuantAvgPool1d(_LegacyQuantInputBaseMixin, pooling.AvgPool1d):
    """Quantized version of nn.AvgPool1d."""


class QuantAvgPool2d(_LegacyQuantInputBaseMixin, pooling.AvgPool2d):
    """Quantized version of nn.AvgPool2d."""


class QuantAvgPool3d(_LegacyQuantInputBaseMixin, pooling.AvgPool3d):
    """Quantized version of nn.AvgPool3d."""


class QuantAdaptiveAvgPool1d(_LegacyQuantInputBaseMixin, pooling.AdaptiveAvgPool1d):
    """Quantized version of nn.AdaptiveAvgPool1d."""


class QuantAdaptiveAvgPool2d(_LegacyQuantInputBaseMixin, pooling.AdaptiveAvgPool2d):
    """Quantized version of nn.AdaptiveAvgPool2d."""


class QuantAdaptiveAvgPool3d(_LegacyQuantInputBaseMixin, pooling.AdaptiveAvgPool3d):
    """Quantized version of nn.AdaptiveAvgPool3d."""


# Define alias with Quant prefix
MaxPool1d = QuantMaxPool1d
MaxPool2d = QuantMaxPool2d
MaxPool3d = QuantMaxPool3d
AvgPool1d = QuantAvgPool1d
AvgPool2d = QuantAvgPool2d
AvgPool3d = QuantAvgPool3d
AdaptiveAvgPool1d = QuantAdaptiveAvgPool1d
AdaptiveAvgPool2d = QuantAdaptiveAvgPool2d
AdaptiveAvgPool3d = QuantAdaptiveAvgPool3d

QuantModuleRegistry.register({pooling.MaxPool1d: "nn.MaxPool1d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.MaxPool2d: "nn.MaxPool2d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.MaxPool3d: "nn.MaxPool3d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AvgPool1d: "nn.AvgPool1d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AvgPool2d: "nn.AvgPool2d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AvgPool3d: "nn.AvgPool3d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AdaptiveAvgPool1d: "nn.AdaptiveAvgPool1d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AdaptiveAvgPool2d: "nn.AdaptiveAvgPool2d"})(QuantInputBase)
QuantModuleRegistry.register({pooling.AdaptiveAvgPool3d: "nn.AdaptiveAvgPool3d"})(QuantInputBase)
