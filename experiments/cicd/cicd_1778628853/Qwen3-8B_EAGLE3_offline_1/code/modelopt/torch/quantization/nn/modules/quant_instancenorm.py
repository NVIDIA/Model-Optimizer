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

"""Quantized instance normalization module."""

import torch.nn as nn

from .quant_module import QuantInputBase, QuantModuleRegistry, _LegacyQuantInputBaseMixin

__all__ = ["QuantInstanceNorm1d", "QuantInstanceNorm2d", "QuantInstanceNorm3d"]


class QuantInstanceNorm1d(_LegacyQuantInputBaseMixin, nn.InstanceNorm1d):
    """Applies Quantized Instance Normalization over a 3D input."""


class QuantInstanceNorm2d(_LegacyQuantInputBaseMixin, nn.InstanceNorm2d):
    """Applies Quantized Instance Normalization over a 4D input."""


class QuantInstanceNorm3d(_LegacyQuantInputBaseMixin, nn.InstanceNorm3d):
    """Applies Quantized Instance Normalization over a 5D input."""


QuantModuleRegistry.register({nn.InstanceNorm1d: "nn.InstanceNorm1d"})(QuantInputBase)
QuantModuleRegistry.register({nn.InstanceNorm2d: "nn.InstanceNorm2d"})(QuantInputBase)
QuantModuleRegistry.register({nn.InstanceNorm3d: "nn.InstanceNorm3d"})(QuantInputBase)
