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

"""Tests for ``torch.nn.LayerNorm`` being registered in ``QuantModuleRegistry``."""

import torch
import torch.nn as nn

from modelopt.torch.quantization.nn import QuantModuleRegistry


def test_layernorm_quant_wrapper_is_identity_when_quantizers_disabled():
    qln = QuantModuleRegistry.convert(nn.LayerNorm(8))
    qln.input_quantizer.disable()
    qln.output_quantizer.disable()

    x = torch.randn(2, 8)
    ref = nn.functional.layer_norm(x, (8,), qln.weight, qln.bias, eps=qln.eps)
    assert torch.allclose(qln(x), ref, rtol=0, atol=0)
