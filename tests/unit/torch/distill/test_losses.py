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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.distill.losses import _chunked_logits_kl


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.7, 2.0])
def test_chunked_logits_kl_matches_dense_loss_and_masked_gradient(dtype, temperature):
    torch.manual_seed(1234)
    student = torch.randn(13, 2, 37, dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student)
    mask = torch.rand(13, 2)
    p = student.float() / temperature
    q = teacher.float() / temperature
    dense = F.kl_div(
        p - p.logsumexp(-1, keepdim=True),
        q - q.logsumexp(-1, keepdim=True),
        reduction="none",
        log_target=True,
    ).sum(-1)
    (expected_gradient,) = torch.autograd.grad((dense * mask).sum(), student)
    chunked = _chunked_logits_kl(student, teacher, temperature, 5)
    (actual_gradient,) = torch.autograd.grad((chunked * mask).sum(), student)
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_chunked_logits_kl_retains_only_original_logits():
    student = torch.randn(17, 2, 37, requires_grad=True)
    teacher = torch.randn_like(student)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        _chunked_logits_kl(student, teacher, 1.0, 5).sum().backward()
    assert len(saved) == 2
    assert saved[0] is student and saved[1] is teacher
