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

from modelopt.torch.quantization.linear_attention.decode import _encode, _hadamard32


def _rotate(value):
    # Independent dense Sylvester matrix; production uses a butterfly transform.
    matrix = value.new_tensor([[(-1) ** (i & j).bit_count() for j in range(32)] for i in range(32)])
    return (value.reshape(*value.shape[:-1], -1, 32) @ (matrix / 32**0.5)).reshape_as(value)


def _qdq(value):
    with torch.no_grad():
        groups = value.float().unflatten(-1, (-1, 32))
        scale = (groups.abs().amax(-1, keepdim=True) / 127).clamp_min(6e-8)
        normalized = groups / scale
        codes = torch.where(normalized >= 0, (normalized + 0.5).floor(), (normalized - 0.5).ceil())
        rounded = (codes.clamp(-127, 127) * scale.half().float()).flatten(-2).to(value.dtype)
    return (value - value.detach()) + rounded


def test_hadamard_codec_matches_dense_oracle_and_identity_ste():
    torch.manual_seed(321)
    state = torch.randn(2, 4, 64, dtype=torch.float64, requires_grad=True)
    encoded = _encode(
        _hadamard32(state),
        True,
        64,
        state=True,
        state_format="int8",
        state_codec="int8_hadamard32",
    )
    actual = _hadamard32(encoded.values)
    expected = _rotate(_qdq(_rotate(state)))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert encoded.scales.shape == (2, 4, 2)
    assert encoded.scales.dtype == torch.float16
    probe = torch.randn_like(state)
    (gradient,) = torch.autograd.grad((actual * probe).sum(), state)
    torch.testing.assert_close(gradient, probe, rtol=1e-10, atol=1e-10)
