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

"""Paper-reference coverage for LK lambda values, gradients and retained memory."""

import pytest
import torch

from modelopt.torch.speculative.losses import masked_lk_lambda_loss


def _reference(teacher, student, mask, eta, indices=None):
    p = teacher.detach().float().log_softmax(-1).exp()
    teacher = teacher.detach().float()
    if indices is not None:
        p = p[..., indices]
        teacher = teacher[..., indices]
    log_p_conditional = teacher.log_softmax(-1)
    log_q = student.float().log_softmax(-1)
    q = log_q.exp()
    overlap = torch.where(q < p, q, p).sum(-1)
    p_conditional = log_p_conditional.exp()
    kl = torch.special.xlogy(p_conditional, p_conditional).sum(-1) - (p_conditional * log_q).sum(-1)
    count = mask.sum() if mask.any() else 1
    blend = torch.exp(-eta * (overlap * mask).sum() / count).detach()
    return ((blend * kl + (1 - blend) * (1 - overlap)) * mask).sum() / count


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("eta", [0.0, 3.0])
@pytest.mark.parametrize(
    "case",
    ["random", "identical", "empty_mask", "extreme", "zero_probability", "fractional_weights"],
)
def test_lk_loss_matches_reference_and_only_differentiates_student(truncated, eta, case):
    torch.manual_seed(17)
    teacher = torch.randn(2, 5, 11)
    if case in ("zero_probability", "empty_mask"):
        teacher[..., 2] = -torch.inf
    teacher.requires_grad_()
    indices = torch.tensor([8, 2, 6, 0, 10]) if truncated else None
    student = torch.randn(2, 5, 5 if truncated else 11)
    if case == "identical":
        student = teacher.detach().clone() if indices is None else teacher.detach()[..., indices]
    elif case == "extreme":
        student *= 1000
    student = student.requires_grad_()
    reference_student = student.detach().clone().requires_grad_()
    mask = torch.tensor([[0, 1, 1, 0, 1], [1, 0, 1, 1, 0]], dtype=torch.float32)
    if case == "empty_mask":
        mask.zero_()
    elif case == "fractional_weights":
        mask *= torch.linspace(0.01, 0.09, mask.numel()).reshape_as(mask)
    expected = _reference(teacher, reference_student, mask, eta, indices)
    actual = masked_lk_lambda_loss(teacher, student, mask, eta=eta, draft_vocab_indices=indices)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    (actual * 1.7).backward()
    (expected * 1.7).backward()
    torch.testing.assert_close(student.grad, reference_student.grad, atol=1e-6, rtol=1e-5)
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[mask == 0]) == 0
    if case == "empty_mask" or (case == "identical" and not truncated):
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-6, rtol=0)


def test_lk_truncated_overlap_keeps_missing_teacher_mass():
    teacher = torch.tensor([[[0.1, 0.3, 0.2, 0.4]]]).log().requires_grad_()
    indices = torch.tensor([3, 0, 1])
    student = (teacher.detach()[..., indices].exp() / 0.8).log().requires_grad_()
    actual = masked_lk_lambda_loss(teacher, student, torch.ones(1, 1), draft_vocab_indices=indices)
    # Conditional KL is zero, but the true acceptance ceiling is 0.8, not 1.
    expected = (1 - torch.exp(torch.tensor(-3.0 * 0.8))) * 0.2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=1e-6, rtol=0)
    assert teacher.grad is None


def test_lk_probability_ties_match_reference_subgradient():
    teacher = torch.tensor([[[0.25, 0.25, 0.5]]]).log()
    student = torch.tensor([[[0.25, 0.5, 0.25]]]).log().requires_grad_()
    reference_student = student.detach().clone().requires_grad_()
    mask = torch.ones(1, 1)
    masked_lk_lambda_loss(teacher, student, mask).backward()
    _reference(teacher, reference_student, mask, 3.0).backward()
    torch.testing.assert_close(student.grad, reference_student.grad)


@pytest.mark.parametrize("vocab_size", [64, 1024])
def test_lk_retains_only_inputs_and_compact_statistics(vocab_size):
    teacher = torch.randn(2, 5, vocab_size)
    student = torch.randn_like(teacher, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = masked_lk_lambda_loss(teacher, student, torch.ones(2, 5))
    inputs = {teacher.data_ptr(), student.data_ptr()}
    assert all(t.data_ptr() in inputs or t.numel() <= 10 for t in saved)
    loss.backward()
