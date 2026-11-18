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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["masked_lk_lambda_loss", "masked_soft_target_cross_entropy"]

_LK_TOKEN_CHUNK_SIZE = 1024


def _lk_lambda_loss(
    divergence: torch.Tensor,
    overlap: torch.Tensor,
    weights: torch.Tensor,
    weight_sum: torch.Tensor,
    *,
    eta: float,
    scale: float = 1.0,
    mean_acceptance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend divergence and acceptance losses for soft targets and ``HFDFlash2Model``.

    DFlash2 supplies its unweighted mean acceptance; soft targets use the loss weights.
    """
    # A masked token can have infinite KL when only the student assigns zero mass.
    divergence_loss = (divergence.masked_fill(weights == 0, 0) * weights).sum() / weight_sum
    tv_loss = ((1.0 - overlap) * weights).sum() / weight_sum
    if mean_acceptance is None:
        mean_acceptance = (overlap.detach() * weights).sum() / weight_sum
    blend = scale * torch.exp(-eta * mean_acceptance.detach())
    return blend * divergence_loss + (1.0 - blend) * tv_loss


def _soft_target_lk_statistics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute full-vocabulary KL and overlap for each token."""
    # Mask before normalization: masking a NaN loss later still leaves NaN gradients.
    masked_tokens = (loss_mask == 0).unsqueeze(-1)
    log_p = torch.where(masked_tokens, 0, teacher_logits).float().log_softmax(-1)
    p = log_p.exp()
    log_q = torch.where(masked_tokens, 0, student_logits).float().log_softmax(-1)
    q = log_q.exp()
    # Keep the repaired objective's subgradient at equal probabilities.
    overlap = torch.where(q < p, q, p).sum(-1)
    # Zero teacher mass contributes zero KL, including when the student mass is zero.
    kl = F.kl_div(log_q.masked_fill(p == 0, 0), p, reduction="none").sum(-1)
    return kl, overlap


def _soft_target_lk_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    kl, overlap = _soft_target_lk_statistics(teacher_logits, student_logits, loss_mask)
    weights = loss_mask.float()
    weight_sum = weights.sum()
    return _lk_lambda_loss(
        kl, overlap, weights, weight_sum.masked_fill(weight_sum == 0, 1), eta=eta
    )


def masked_lk_lambda_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eta: float = 3.0,
) -> torch.Tensor:
    """Compute LK lambda with a frozen teacher and one detached blend weight per call.

    Implements https://arxiv.org/abs/2602.23881, sections 4.2 and 4.4. Call once per
    aligned draft depth. The mask averages over the local microbatch, without DP reductions.

    Args:
        teacher_logits: Full-vocabulary target logits, shaped [batch, sequence, vocabulary].
        student_logits: Aligned draft logits over the same full vocabulary as the teacher.
        loss_mask: Nonnegative valid-token weights shaped [batch, sequence].
        eta: Nonnegative decay rate; zero selects pure KL.

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
    if teacher_logits.shape[-1] != student_logits.shape[-1]:
        raise ValueError("LK lambda requires the same full vocabulary for teacher and student")
    if teacher_logits.shape[0] * teacher_logits.shape[1] > _LK_TOKEN_CHUNK_SIZE:
        # Bound full-vocabulary temporaries without copying noncontiguous aligned logits.
        divergences, overlaps = [], []
        for teacher, student, mask in zip(
            teacher_logits.detach().unbind(), student_logits.unbind(), loss_mask.detach().unbind()
        ):
            for teacher_chunk, student_chunk, mask_chunk in zip(
                teacher.split(_LK_TOKEN_CHUNK_SIZE),
                student.split(_LK_TOKEN_CHUNK_SIZE),
                mask.split(_LK_TOKEN_CHUNK_SIZE),
            ):
                divergence, overlap = checkpoint(
                    _soft_target_lk_statistics,
                    teacher_chunk,
                    student_chunk,
                    mask_chunk,
                    use_reentrant=False,
                )
                divergences.append(divergence)
                overlaps.append(overlap)
        weights = loss_mask.detach().float()
        weight_sum = weights.sum()
        # One detached blend over the whole microbatch preserves the LK objective.
        return _lk_lambda_loss(
            torch.cat(divergences).view_as(weights),
            torch.cat(overlaps).view_as(weights),
            weights,
            weight_sum.masked_fill(weight_sum == 0, 1),
            eta=eta,
        )
    # Recompute vocabulary-sized probabilities rather than retain them across draft depths.
    return checkpoint(
        _soft_target_lk_loss,
        teacher_logits.detach(),
        student_logits,
        loss_mask.detach(),
        eta,
        use_reentrant=False,
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
