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

"""CPU unit tests for quantization functions."""

import torch

from modelopt.torch.quantization.nn.functional import clip


def test_clip_backward_with_broadcast_constant_bounds():
    inputs = torch.tensor([[-2.0, 0.5, 4.0], [-0.5, 2.0, 2.0]], requires_grad=True)
    clip_value_min = torch.tensor([-1.0, 0.0, 1.0])
    clip_value_max = torch.tensor([0.0, 1.0, 3.0])

    outputs = clip(inputs, clip_value_min, clip_value_max)
    outputs.sum().backward()

    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_outputs = torch.maximum(
        torch.minimum(reference_inputs, clip_value_max), clip_value_min
    )
    reference_outputs.sum().backward()

    torch.testing.assert_close(outputs, reference_outputs)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
