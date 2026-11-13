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
    """Blend weighted divergence and acceptance losses as in DFlash2.

    DFlash2 supplies its unweighted mean acceptance; soft targets use the loss weights.
    """
    # A masked token can have infinite KL when only the student assigns zero mass.
    divergence_loss = (divergence.masked_fill(weights == 0, 0) * weights).sum() / weight_sum
    tv_loss = ((1.0 - overlap) * weights).sum() / weight_sum
    if mean_acceptance is None:
        mean_acceptance = (overlap.detach() * weights).sum() / weight_sum
    blend = scale * torch.exp(-eta * mean_acceptance.detach())
    return blend * divergence_loss + (1.0 - blend) * tv_loss


def _soft_target_lk_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    eta: float,
    draft_vocab_indices: torch.Tensor | None,
) -> torch.Tensor:
    log_p = teacher_logits.float().log_softmax(-1)
    if draft_vocab_indices is not None:
        log_p = log_p.index_select(-1, draft_vocab_indices)
    p = log_p.exp()
    p_conditional = p if draft_vocab_indices is None else log_p.softmax(-1)
    log_q = student_logits.float().log_softmax(-1)
    q = log_q.exp()
    # Keep the repaired objective's subgradient at equal probabilities.
    overlap = torch.where(q < p, q, p).sum(-1)
    # Zero teacher mass contributes zero KL, including when the student mass is zero.
    kl = F.kl_div(log_q.masked_fill(p_conditional == 0, 0), p_conditional, reduction="none").sum(-1)
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
    # Recompute vocabulary-sized probabilities rather than retain them across draft depths.
    return checkpoint(
        _soft_target_lk_loss,
        teacher_logits.detach(),
        student_logits,
        loss_mask.detach(),
        eta,
        draft_vocab_indices,
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
