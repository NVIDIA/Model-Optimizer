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

from modelopt.torch.quantization.linear_attention.decode import _hadamard32


def _rotate(value):
    # Independent dense Sylvester matrix; production uses a butterfly transform.
    matrix = value.new_tensor([[(-1) ** (i & j).bit_count() for j in range(32)] for i in range(32)])
    return (value.reshape(*value.shape[:-1], -1, 32) @ (matrix / 32**0.5)).reshape_as(value)


def test_hadamard_matches_dense_oracle_and_backward():
    torch.manual_seed(321)
    state = torch.randn(2, 4, 64, dtype=torch.float64, requires_grad=True)
    actual = _hadamard32(state)
    expected = _rotate(state)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(state)
    (gradient,) = torch.autograd.grad((actual * probe).sum(), state)
    torch.testing.assert_close(gradient, _rotate(probe), rtol=1e-10, atol=1e-10)
