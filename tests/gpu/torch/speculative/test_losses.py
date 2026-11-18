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

"""CUDA compilation and precision coverage for the real LK autograd path."""

import pytest
import torch

from modelopt.torch.speculative.losses import masked_lk_lambda_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["binary", "fractional", "empty"])
def test_lk_compiled_cuda_matches_fp32_reference(dtype, mask_kind):
    torch.manual_seed(31)
    sequence_length = 1025
    teacher = torch.randn(2, sequence_length, 128, device="cuda", dtype=dtype)
    teacher[..., ::7] = -torch.inf
    student = torch.randn(2, sequence_length, 128, device="cuda", dtype=dtype)
    student.masked_fill_(torch.isneginf(teacher), -torch.inf)
    mask = torch.randint(0, 2, (2, sequence_length), device="cuda").float()
    if mask_kind == "fractional":
        mask /= mask.numel()
    elif mask_kind == "empty":
        mask.zero_()
    # Undefined distributions at masked tokens must not poison the loss or gradients.
    if mask_kind != "fractional":
        student.masked_fill_((mask == 0).unsqueeze(-1), -torch.inf)
    if mask_kind != "binary":
        teacher.masked_fill_((mask == 0).unsqueeze(-1), torch.nan)
    reference_student = student.float().requires_grad_()
    expected = masked_lk_lambda_loss(teacher.float(), reference_student, mask)
    expected.backward()
    assert torch.isfinite(expected)
    assert torch.isfinite(reference_student.grad).all()
    assert torch.count_nonzero(reference_student.grad[mask == 0]) == 0
    compiled = torch.compile(masked_lk_lambda_loss, fullgraph=True)
    for loss_fn in (masked_lk_lambda_loss, compiled):
        actual_student = student.detach().clone().requires_grad_()
        actual = loss_fn(teacher, actual_student, mask)
        actual.backward()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            actual_student.grad.float(), reference_student.grad, atol=1e-5, rtol=0.02
        )
