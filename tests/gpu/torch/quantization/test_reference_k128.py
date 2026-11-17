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

import pytest
import torch

from modelopt.torch.kernels.quantization.linear_attention.reference_k128 import reference_k128_qdq


def test_reference_k128_quantization_and_state_gradient():
    torch.manual_seed(47)
    state = torch.randn(2, 3, 128, 16, device="cuda", requires_grad=True)
    caps = torch.full((3,), 100.0, device="cuda", requires_grad=True)
    output = reference_k128_qdq(state, caps)
    blocks = state.detach().transpose(-1, -2)
    scale = blocks.abs().amax(-1, keepdim=True) / 6
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    nearest = grid[torch.bucketize(blocks.abs() / scale, bounds, right=True)]
    expected = (nearest.copysign(blocks) * scale).transpose(-1, -2)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    probe = torch.randn_like(output)
    state_grad, cap_grad = torch.autograd.grad((output * probe).sum(), (state, caps))
    torch.testing.assert_close(state_grad, probe, rtol=0, atol=0)
    assert torch.isfinite(cap_grad).all()
    with pytest.raises(ValueError, match="one per head"):
        reference_k128_qdq(state, caps[:1])
