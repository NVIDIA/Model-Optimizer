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

import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention.utils import _state_qdq, state_quantizer_config
from modelopt.torch.quantization.nn import TensorQuantizer


def test_nvfp4_state_qdq_and_identity_gradient():
    torch.manual_seed(191)
    value = torch.randn(2, 2, 32, 32, device="cuda", requires_grad=True)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            type="dynamic",
            pass_through_bwd=True,
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        )
    ).cuda()
    assert state_quantizer_config(quantizer) == ("nvfp4", 16)
    result = _state_qdq(value, quantizer)
    expected = torch.stack(
        [quantizer(matrix) for matrix in value.reshape(-1, value.shape[-2], value.shape[-1])]
    ).reshape_as(value)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.isfinite(result).all() and not torch.equal(result, value)

    # A much larger co-scheduled request or head must not change the first matrix's scale.
    small = value.detach()[:1, :1]
    for dim in (0, 1):
        mixed = torch.cat((small, small * 4096), dim=dim)
        first = _state_qdq(mixed, quantizer)[:1, :1]
        torch.testing.assert_close(first, _state_qdq(small, quantizer), rtol=0, atol=0)
    probe = torch.randn_like(value)
    (gradient,) = torch.autograd.grad((result * probe).sum(), value)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)
