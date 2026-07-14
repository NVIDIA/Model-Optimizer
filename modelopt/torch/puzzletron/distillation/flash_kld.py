# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact memory-bounded CE and KLD from hidden states and LM heads."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from .loss import (
    _all_reduce_forward,
    _distributed_log_softmax,
    _infer_tp_group_from_dtensor,
    _vocab_shard_offset,
)

__all__ = ["TrainingFlashKLD"]


class TrainingFlashKLD(nn.Module):
    """Compute exact CE and KLD without retaining sequence-by-vocabulary logits.

    The student LM head is applied once per token chunk. CE and KLD share that
    projection, while a disabled KLD term avoids the teacher projection
    entirely. Vocabulary-sharded DTensors remain sharded throughout the loss.

    Args:
        token_chunk_size: Maximum number of tokens projected at once.
        temperature: KLD softmax temperature.
        ignore_index: Label value excluded from both CE and KLD.
        fp32_upcast: Compute probability reductions in float32.
        checkpoint_chunks: Recompute projections during backward to bound
            activation memory.
    """

    def __init__(
        self,
        *,
        token_chunk_size: int = 128,
        temperature: float = 1.0,
        ignore_index: int = -100,
        fp32_upcast: bool = True,
        checkpoint_chunks: bool = True,
    ) -> None:
        super().__init__()
        if token_chunk_size <= 0:
            raise ValueError("TrainingFlashKLD token_chunk_size must be positive")
        if temperature <= 0:
            raise ValueError("TrainingFlashKLD temperature must be positive")
        self.token_chunk_size = int(token_chunk_size)
        self.temperature = float(temperature)
        self.ignore_index = int(ignore_index)
        self.fp32_upcast = bool(fp32_upcast)
        self.checkpoint_chunks = bool(checkpoint_chunks)

    @staticmethod
    def _local_labels(labels: torch.Tensor, tp_group) -> torch.Tensor:
        if not isinstance(labels, DTensor):
            return labels
        return labels.to_local() if tp_group is not None else labels.full_tensor()

    def _local_chunk_losses(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        *,
        compute_ce: bool,
        compute_kd: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student = student_logits.float() if self.fp32_upcast else student_logits
        valid = labels != self.ignore_index
        zero = student.new_zeros(())
        ce = (
            F.cross_entropy(
                student,
                labels,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            if compute_ce
            else zero
        )
        if not compute_kd or not torch.any(valid):
            return ce, zero
        if teacher_logits is None:
            raise ValueError("KLD requires teacher logits")
        teacher = teacher_logits.float() if self.fp32_upcast else teacher_logits
        student_logprob = F.log_softmax(student / self.temperature, dim=-1)
        teacher_logprob = F.log_softmax(teacher / self.temperature, dim=-1)
        kd = F.kl_div(
            student_logprob,
            teacher_logprob,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        kd = kd.masked_select(valid).sum() * self.temperature**2
        return ce, kd

    def _tp_chunk_losses(
        self,
        student_logits: DTensor,
        teacher_logits: torch.Tensor | None,
        labels: torch.Tensor,
        *,
        compute_ce: bool,
        compute_kd: bool,
        tp_group,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_offset, _ = _vocab_shard_offset(student_logits, tp_group)
        student = student_logits.to_local()
        if self.fp32_upcast:
            student = student.float()
        labels = self._local_labels(labels, tp_group)
        valid = labels != self.ignore_index
        zero = student.new_zeros(())

        student_logprob = None
        if compute_ce:
            student_logprob = _distributed_log_softmax(student, tp_group)
            local_target = labels - vocab_offset
            local_mask = valid & (local_target >= 0) & (local_target < student.shape[-1])
            safe_target = local_target.clamp(0, student.shape[-1] - 1)
            selected = student_logprob.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
            selected = torch.where(local_mask, selected, torch.zeros_like(selected))
            selected = _all_reduce_forward(
                selected,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )
            ce = -selected.masked_select(valid).sum()
        else:
            ce = zero

        if not compute_kd or not torch.any(valid):
            return ce, zero
        if teacher_logits is None:
            raise ValueError("KLD requires teacher logits")
        if not isinstance(teacher_logits, DTensor):
            raise ValueError("TP KLD requires teacher logits aligned to the student DTensor")
        teacher = teacher_logits.to_local()
        if teacher.shape != student.shape:
            raise ValueError(
                "TP student and teacher vocabulary shards differ: "
                f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
            )
        if self.fp32_upcast:
            teacher = teacher.float()
        if self.temperature == 1.0 and student_logprob is not None:
            student_kd_logprob = student_logprob
        else:
            student_kd_logprob = _distributed_log_softmax(
                student / self.temperature,
                tp_group,
            )
        teacher_logprob = _distributed_log_softmax(
            teacher / self.temperature,
            tp_group,
        )
        local_kd = (
            teacher_logprob.exp() * (teacher_logprob - student_kd_logprob)
        ).sum(dim=-1)
        kd = _all_reduce_forward(
            local_kd,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
        kd = kd.masked_select(valid).sum() * self.temperature**2
        return ce, kd

    def _chunk_losses(
        self,
        student_hidden: torch.Tensor,
        student_head: nn.Module,
        labels: torch.Tensor,
        *,
        teacher_hidden: torch.Tensor | None,
        teacher_project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        compute_ce: bool,
        compute_kd: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projection = getattr(student_head, "_puzzletron_projection_forward", None)
        student_logits = (
            projection(student_hidden)
            if projection is not None
            else student_head(student_hidden)
        )
        teacher_logits = None
        if compute_kd:
            if teacher_hidden is None or teacher_project is None:
                raise ValueError("KLD requires teacher hidden states and a teacher projection")
            with torch.no_grad():
                teacher_logits = teacher_project(teacher_hidden, student_logits)
        tp_group = _infer_tp_group_from_dtensor(student_logits)
        if tp_group is not None:
            return self._tp_chunk_losses(
                student_logits,
                teacher_logits,
                labels,
                compute_ce=compute_ce,
                compute_kd=compute_kd,
                tp_group=tp_group,
            )
        if isinstance(student_logits, DTensor):
            student_logits = student_logits.full_tensor()
        if isinstance(teacher_logits, DTensor):
            teacher_logits = teacher_logits.full_tensor()
        labels = self._local_labels(labels, None)
        return self._local_chunk_losses(
            student_logits,
            teacher_logits,
            labels,
            compute_ce=compute_ce,
            compute_kd=compute_kd,
        )

    def forward(
        self,
        student_hidden: torch.Tensor,
        student_head: nn.Module,
        labels: torch.Tensor,
        *,
        teacher_hidden: torch.Tensor | None = None,
        teacher_project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        compute_ce: bool = True,
        compute_kd: bool = True,
        num_label_tokens: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return independently normalized exact CE and KLD scalars."""
        if not compute_ce and not compute_kd:
            raise ValueError("TrainingFlashKLD requires at least one enabled objective")
        student_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
        labels_flat = labels.reshape(-1)
        if student_flat.shape[0] != labels_flat.shape[0]:
            raise ValueError(
                "Hidden states and labels must contain the same number of tokens; "
                f"got {student_flat.shape[0]} and {labels_flat.shape[0]}"
            )
        teacher_flat = None
        if compute_kd:
            if teacher_hidden is None:
                raise ValueError("KLD requires teacher hidden states")
            teacher_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
            if teacher_flat.shape[0] != student_flat.shape[0]:
                raise ValueError(
                    "Student and teacher hidden states must contain the same number of tokens; "
                    f"got {student_flat.shape[0]} and {teacher_flat.shape[0]}"
                )

        ce_sum = None
        kd_sum = None
        use_checkpoint = self.checkpoint_chunks and torch.is_grad_enabled()
        for start in range(0, student_flat.shape[0], self.token_chunk_size):
            stop = min(start + self.token_chunk_size, student_flat.shape[0])
            student_chunk = student_flat[start:stop]
            labels_chunk = labels_flat[start:stop]
            teacher_chunk = (
                teacher_flat[start:stop]
                if teacher_flat is not None
                else student_chunk.new_empty((0, student_chunk.shape[-1]))
            )

            def chunk_fn(student_value, teacher_value, label_value):
                return self._chunk_losses(
                    student_value,
                    student_head,
                    label_value,
                    teacher_hidden=teacher_value if compute_kd else None,
                    teacher_project=teacher_project,
                    compute_ce=compute_ce,
                    compute_kd=compute_kd,
                )

            if use_checkpoint:
                ce_chunk, kd_chunk = checkpoint(
                    chunk_fn,
                    student_chunk,
                    teacher_chunk,
                    labels_chunk,
                    use_reentrant=False,
                )
            else:
                ce_chunk, kd_chunk = chunk_fn(
                    student_chunk,
                    teacher_chunk,
                    labels_chunk,
                )
            ce_sum = ce_chunk if ce_sum is None else ce_sum + ce_chunk
            kd_sum = kd_chunk if kd_sum is None else kd_sum + kd_chunk

        if ce_sum is None or kd_sum is None:
            local_hidden = (
                student_hidden.to_local()
                if isinstance(student_hidden, DTensor)
                else student_hidden
            )
            zero = local_hidden.sum() * 0.0
            return zero, zero
        denominator = (
            (labels_flat != self.ignore_index).sum()
            if num_label_tokens is None
            else num_label_tokens
        )
        if isinstance(denominator, int) and denominator == 0:
            return ce_sum * 0.0, kd_sum * 0.0
        denominator = torch.as_tensor(
            denominator,
            device=ce_sum.device,
            dtype=ce_sum.dtype,
        ).clamp_min(1)
        return ce_sum / denominator, kd_sum / denominator
