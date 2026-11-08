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

"""Shared speculative distillation objectives."""

import math

import torch

__all__ = ["masked_lk_lambda_loss", "masked_soft_target_cross_entropy"]


class _LKLambdaLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher_logits, student_logits, loss_mask, eta, draft_vocab_indices):
        teacher_normalizer = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
        selected_teacher = (
            teacher_logits
            if draft_vocab_indices is None
            else teacher_logits.index_select(-1, draft_vocab_indices)
        ).float()
        conditional_normalizer = (
            teacher_normalizer
            if draft_vocab_indices is None
            else torch.logsumexp(selected_teacher, dim=-1, keepdim=True)
        )
        student_normalizer = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
        log_q = student_logits.float() - student_normalizer
        log_p_conditional = selected_teacher - conditional_normalizer
        p = (selected_teacher - teacher_normalizer).exp()
        q = log_q.exp()
        student_is_lower = q < p
        overlap = torch.where(student_is_lower, q, p).sum(-1)
        picked_mass = (q * student_is_lower).sum(-1, keepdim=True)
        del selected_teacher, p, q, student_is_lower
        kl = torch.nn.functional.kl_div(log_q, log_p_conditional.exp(), reduction="none").sum(-1)
        weights = loss_mask.float()
        weight_sum = weights.sum()
        weights = weights / weight_sum.masked_fill(weight_sum == 0, 1)
        blend = torch.exp(-eta * (weights * overlap).sum())
        ctx.save_for_backward(
            teacher_logits,
            student_logits,
            draft_vocab_indices,
            teacher_normalizer,
            conditional_normalizer,
            student_normalizer,
            picked_mass,
            weights,
            blend,
        )
        return (weights * (blend * kl + (1 - blend) * (1 - overlap))).sum()

    @staticmethod
    def backward(ctx, grad_output):
        (
            teacher_logits,
            student_logits,
            draft_vocab_indices,
            teacher_normalizer,
            conditional_normalizer,
            student_normalizer,
            picked_mass,
            weights,
            blend,
        ) = ctx.saved_tensors
        selected_teacher = (
            teacher_logits
            if draft_vocab_indices is None
            else teacher_logits.index_select(-1, draft_vocab_indices)
        ).float()
        p = (selected_teacher - teacher_normalizer).exp()
        p_conditional = (selected_teacher - conditional_normalizer).exp()
        q = (student_logits.float() - student_normalizer).exp()
        # Reconstruct probabilities instead of retaining full-width FP32 activations per round.
        grad = blend * (q - p_conditional)
        grad += (blend - 1) * q * ((q < p).to(q.dtype) - picked_mass)
        grad *= (grad_output * weights).unsqueeze(-1)
        return None, grad.to(student_logits.dtype), None, None, None


def masked_lk_lambda_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eta: float = 3.0,
    draft_vocab_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute LK lambda with a frozen teacher and one detached blend weight per call.

    Implements https://arxiv.org/abs/2602.23881, sections 4.2 and 4.4. Call once per
    aligned draft depth. The mask averages over the local microbatch, without DP reductions.

    Args:
        teacher_logits: Full-vocabulary target logits, shaped [batch, sequence, vocabulary].
        student_logits: Aligned draft logits, with the same batch and sequence dimensions.
        loss_mask: Nonnegative valid-token weights shaped [batch, sequence].
        eta: Nonnegative decay rate; zero selects pure KL.
        draft_vocab_indices: Target token IDs in draft-vocabulary order. KL conditions the
            teacher on these IDs; overlap retains its original full-vocabulary normalization.

    Returns:
        FP32 masked loss. Only student logits receive gradients; an empty mask gives zero.
    """
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("LK lambda eta must be finite and nonnegative")
    if (
        teacher_logits.ndim != 3
        or student_logits.ndim != 3
        or teacher_logits.shape[:2] != student_logits.shape[:2]
        or loss_mask.shape != student_logits.shape[:2]
    ):
        raise ValueError("LK lambda requires aligned [batch, sequence] logits and loss_mask")
    if draft_vocab_indices is None:
        if teacher_logits.shape[-1] != student_logits.shape[-1]:
            raise ValueError("Different vocabularies require draft_vocab_indices")
    elif draft_vocab_indices.ndim != 1 or draft_vocab_indices.numel() != student_logits.shape[-1]:
        raise ValueError("draft_vocab_indices must contain one target token ID per draft token")
    return _LKLambdaLoss.apply(
        teacher_logits.detach(), student_logits, loss_mask.detach(), eta, draft_vocab_indices
    )


def masked_soft_target_cross_entropy(
    teacher_probabilities: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute EAGLE's masked CE from normalized teacher probabilities and student logits.

    Preserves gradients through the teacher for EAGLE's base-LoRA training.
    Callers with a frozen teacher must detach its probabilities.
    """
    loss_mask = loss_mask[:, : student_logits.shape[1], None]
    student_log_probabilities = torch.log_softmax(student_logits, dim=2)
    return -torch.sum(
        torch.sum(loss_mask * teacher_probabilities * student_log_probabilities, dim=2)
    ) / (loss_mask.sum() + 1e-5)
