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

"""Support quantization for FastVideo layers."""

import torch
import torch.nn.functional as F
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.vaes.wanvae import WanCausalConv3d

from ...nn import QuantLinearConvBase, QuantModuleRegistry
from ...nn.modules.quant_conv import _QuantConv3d
from ...nn.modules.quant_linear import _QuantLinear
from ...utils import is_torch_export_mode


@QuantModuleRegistry.register({WanCausalConv3d: "WanCausalConv3d"})
class _QuantWanCausalConv3d(_QuantConv3d):
    @staticmethod
    def _get_quantized_weight(module: "QuantLinearConvBase", weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight in linear format for proper block-wise FP4 quantization."""
        if module._enable_weight_quantization or is_torch_export_mode():
            # Quantize in linear format (block-wise quantization works correctly here)
            return module.weight_quantizer(weight)

        return weight

    def forward(self, x, cache_x=None):
        from fastvideo.platforms import current_platform

        with self.quantize_weight():
            padding = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
            x = F.pad(x, padding)
            x = (
                x.to(self.weight.dtype) if current_platform.is_mps() else x
            )  # casting needed for mps since amp isn't supported

            input = self.input_quantizer(x)
            output = super(WanCausalConv3d, self).forward(input)

            if isinstance(output, tuple):
                return (self.output_quantizer(output[0]), *output[1:])
            return self.output_quantizer(output)


@QuantModuleRegistry.register({ReplicatedLinear: "ReplicatedLinear"})
class _QuantReplicatedLinear(_QuantLinear):
    pass
