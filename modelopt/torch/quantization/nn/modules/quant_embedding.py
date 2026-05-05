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

"""Quantized Embedding.

``nn.Embedding`` quantization is weight-only: only the lookup table (``weight``) is
fake-quantized. Embedding inputs are integer indices — their ``input_quantizer`` is
registered (so config entries like ``"*input_quantizer"`` can still target it) but is
disabled by default so integer tensors pass through untouched.
"""

import torch.nn as nn

from ... import tensor_quant
from .quant_module import QuantLinearConvBase, QuantModuleRegistry

__all__ = ["QuantEmbedding"]


@QuantModuleRegistry.register({nn.Embedding: "nn.Embedding"})
class _QuantEmbedding(QuantLinearConvBase):
    """Quantized base class for ``nn.Embedding``.

    Weight-only quantization. Input/output quantizers are created (so wildcard configs
    still resolve cleanly) but are disabled — an embedding's input is an index tensor.
    """

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def _setup(self):
        super()._setup()
        # Embedding inputs are integer indices; never fake-quantize them.
        self.input_quantizer.disable()
        # output_quantizer is already disabled by QuantInputBase._setup().


# Alias to follow the naming convention of QuantLinear.
QuantEmbedding = _QuantEmbedding
