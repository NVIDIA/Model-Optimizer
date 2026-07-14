# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint

__all__ = ["ChunkedCrossEntropy", "KDLoss", "TVDLoss", "build_distribution_loss"]


def _infer_tp_group_from_dtensor(tensor: torch.Tensor):
    """Return the DTensor mesh group when logits are vocab-sharded."""
    if not isinstance(tensor, DTensor):
        return None
    vocab_dim = tensor.ndim - 1
    for mesh_dim, placement in enumerate(tensor.placements):
        if isinstance(placement, Shard) and placement.dim in (-1, vocab_dim):
            return tensor.device_mesh.get_group(mesh_dim)
    return None


def _local_logits(tensor: torch.Tensor, tp_group):
    if not isinstance(tensor, DTensor):
        return tensor
    return tensor.to_local() if tp_group is not None else tensor.full_tensor()


def _vocab_shard_offset(tensor: DTensor, tp_group) -> tuple[int, int]:
    """Return this rank's contiguous vocabulary offset and the global size."""
    global_vocab = int(tensor.shape[-1])
    world_size = dist.get_world_size(tp_group)
    group_rank = dist.get_rank(tp_group)
    base, remainder = divmod(global_vocab, world_size)
    local_vocab = base + int(group_rank < remainder)
    offset = group_rank * base + min(group_rank, remainder)
    if local_vocab != tensor.to_local().shape[-1]:
        raise ValueError(
            "Unsupported non-contiguous vocabulary DTensor shard: "
            f"expected local size {local_vocab}, got {tensor.to_local().shape[-1]}"
        )
    return offset, global_vocab


class _AllReduceForward(torch.autograd.Function):
    """All-reduce values while keeping rank-local SPMD gradient semantics.

    Every TP rank backpropagates the same globally reduced scalar. An autograd
    all-reduce would therefore sum identical upstream gradients once more and
    multiply TP gradients by the group size. The correct vocabulary-parallel
    primitive reduces in forward and is the identity in backward.
    """

    @staticmethod
    def forward(ctx, tensor, op, group):
        ctx.op = op
        ctx.group = group
        output = tensor.clone()
        dist.all_reduce(output, op=op, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def _all_reduce_forward(tensor, *, op, group):
    return _AllReduceForward.apply(tensor, op, group)


def _distributed_log_softmax(logits: torch.Tensor, tp_group):
    maximum = logits.max(dim=-1, keepdim=True).values.detach()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = logits - maximum
    denominator = shifted.exp().sum(dim=-1, keepdim=True)
    denominator = _all_reduce_forward(
        denominator, op=dist.ReduceOp.SUM, group=tp_group
    )
    return shifted - denominator.clamp_min(1.0e-12).log()


class ChunkedCrossEntropy(nn.Module):
    """Memory-bounded CE for replicated or vocabulary-TP logits.

    Unlike AutoModel's generic masked CE, a vocabulary-sharded DTensor remains
    sharded throughout the calculation. Token chunks bound the fp32 softmax
    workspace, and checkpointing recomputes that workspace during backward
    instead of retaining one vocabulary-sized buffer per token.
    """

    def __init__(
        self,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
        chunk_size: int = 0,
        checkpoint_chunks: bool = False,
    ):
        super().__init__()
        if reduction != "sum":
            raise ValueError("ChunkedCrossEntropy currently requires reduction='sum'")
        if chunk_size < 0:
            raise ValueError("CE chunk_size cannot be negative")
        self.fp32_upcast = bool(fp32_upcast)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.chunk_size = int(chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)

    def _tp_chunk_sum(self, logits, labels, tp_group, vocab_offset):
        if self.fp32_upcast:
            logits = logits.float()
        maximum = logits.max(dim=-1).values.detach()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
        shifted = logits - maximum.unsqueeze(-1)
        denominator = shifted.exp().sum(dim=-1)
        denominator = _all_reduce_forward(
            denominator, op=dist.ReduceOp.SUM, group=tp_group
        )

        valid = labels != self.ignore_index
        local_target = labels - vocab_offset
        local_mask = valid & (local_target >= 0) & (local_target < logits.shape[-1])
        safe_target = local_target.clamp(0, logits.shape[-1] - 1)
        selected = logits.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
        selected = torch.where(local_mask, selected, torch.zeros_like(selected))
        selected = _all_reduce_forward(
            selected, op=dist.ReduceOp.SUM, group=tp_group
        )
        per_token = maximum + denominator.clamp_min(1.0e-12).log() - selected
        return per_token.masked_select(valid).sum()

    def _local_chunk_sum(self, logits, labels):
        if self.fp32_upcast:
            logits = logits.float()
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction="sum",
        )

    def _forward_impl(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_label_tokens: int | None,
        *,
        checkpoint_chunks: bool,
    ) -> torch.Tensor:
        tp_group = _infer_tp_group_from_dtensor(logits)
        vocab_offset = 0
        if isinstance(logits, DTensor):
            if tp_group is not None:
                vocab_offset, _ = _vocab_shard_offset(logits, tp_group)
                logits = logits.to_local()
            else:
                logits = logits.full_tensor()
        if isinstance(labels, DTensor):
            labels = labels.to_local() if tp_group is not None else labels.full_tensor()
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "Logits and labels must contain the same number of tokens; "
                f"got {logits.shape[0]} and {labels.shape[0]}"
            )

        chunk_size = self.chunk_size or logits.shape[0]
        loss_sum = logits.new_zeros((), dtype=torch.float32 if self.fp32_upcast else None)
        for start in range(0, logits.shape[0], chunk_size):
            stop = min(start + chunk_size, logits.shape[0])
            logits_chunk = logits[start:stop]
            labels_chunk = labels[start:stop]

            def _chunk_sum(chunk_logits, chunk_labels):
                if tp_group is not None:
                    return self._tp_chunk_sum(
                        chunk_logits, chunk_labels, tp_group, vocab_offset
                    )
                return self._local_chunk_sum(chunk_logits, chunk_labels)

            if checkpoint_chunks and torch.is_grad_enabled() and logits_chunk.requires_grad:
                chunk_loss = checkpoint(
                    _chunk_sum,
                    logits_chunk,
                    labels_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss = _chunk_sum(logits_chunk, labels_chunk)
            loss_sum = loss_sum + chunk_loss

        if num_label_tokens is not None:
            if num_label_tokens == 0:
                return loss_sum * 0.0
            loss_sum = loss_sum / num_label_tokens
        return loss_sum

    def forward(self, logits, labels, mask=None, num_label_tokens=None):
        if mask is not None:
            labels = labels.clone()
            labels.masked_fill_(mask == 0, self.ignore_index)
        return self._forward_impl(
            logits,
            labels,
            num_label_tokens,
            checkpoint_chunks=self.checkpoint_chunks,
        )

    def forward_no_checkpoint(self, logits, labels, num_label_tokens=None):
        return self._forward_impl(
            logits,
            labels,
            num_label_tokens,
            checkpoint_chunks=False,
        )


class _DistributionLoss(nn.Module):
    metric: Literal["kld", "tvd"]

    def __init__(
        self,
        ignore_index: int = -100,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        tp_group=None,
        chunk_size: int = 0,
        checkpoint_chunks: bool = False,
        **_kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("KD temperature must be greater than zero")
        if chunk_size < 0:
            raise ValueError("KD chunk_size cannot be negative")
        self.ignore_index = ignore_index
        self.temperature = float(temperature)
        self.fp32_upcast = fp32_upcast
        self.tp_group = tp_group
        self.chunk_size = int(chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)

    def _per_token(self, student_logits, teacher_logits, tp_group):
        teacher_logprob = (
            _distributed_log_softmax(teacher_logits, tp_group)
            if tp_group is not None
            else teacher_logits.log_softmax(dim=-1)
        )
        student_logprob = (
            _distributed_log_softmax(student_logits, tp_group)
            if tp_group is not None
            else student_logits.log_softmax(dim=-1)
        )
        teacher_prob = teacher_logprob.exp()
        if self.metric == "kld":
            local = (teacher_prob * (teacher_logprob - student_logprob)).sum(dim=-1)
        else:
            local = 0.5 * (teacher_prob - student_logprob.exp()).abs().sum(dim=-1)
        if tp_group is not None:
            local = _all_reduce_forward(
                local, op=dist.ReduceOp.SUM, group=tp_group
            )
        return local

    def _forward_impl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
        *,
        checkpoint_chunks: bool,
    ) -> torch.Tensor:
        if student_logits.ndim > 2:
            student_logits = student_logits.reshape(-1, student_logits.shape[-1])
        if teacher_logits.ndim > 2:
            teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        tp_group = (
            self.tp_group
            if self.tp_group is not None
            else _infer_tp_group_from_dtensor(student_logits)
        )
        student_logits = _local_logits(student_logits, tp_group)
        teacher_logits = _local_logits(teacher_logits, tp_group)
        if isinstance(labels, DTensor):
            labels = labels.to_local() if tp_group is not None else labels.full_tensor()
        labels = labels.reshape(-1)
        valid_mask = labels != self.ignore_index
        if not torch.any(valid_mask):
            return student_logits.new_zeros(())
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "Student and teacher logits must have matching local shapes; "
                f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
            )

        # Slice before masking, fp32 conversion, and temperature scaling. Doing
        # those operations on the complete token x vocabulary matrix defeats
        # chunking and can add tens of GiB for long-context, large-vocabulary
        # KD. Each chunk is numerically independent before the final sum.
        chunk_size = self.chunk_size or student_logits.shape[0]
        loss_sum = student_logits.new_zeros((), dtype=torch.float32 if self.fp32_upcast else None)
        for start in range(0, student_logits.shape[0], chunk_size):
            stop = min(start + chunk_size, student_logits.shape[0])
            chunk_mask = valid_mask[start:stop]
            if not torch.any(chunk_mask):
                continue
            student_chunk = student_logits[start:stop][chunk_mask]
            teacher_chunk = teacher_logits[start:stop][chunk_mask]
            if self.fp32_upcast:
                student_chunk = student_chunk.float()
                teacher_chunk = teacher_chunk.float()
            student_chunk = student_chunk / self.temperature
            teacher_chunk = teacher_chunk / self.temperature

            def _chunk_loss(student, teacher):
                return self._per_token(student, teacher, tp_group).sum()

            if checkpoint_chunks and torch.is_grad_enabled() and student_chunk.requires_grad:
                chunk_loss = checkpoint(
                    _chunk_loss,
                    student_chunk,
                    teacher_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss = _chunk_loss(student_chunk, teacher_chunk)
            loss_sum = loss_sum + chunk_loss

        denominator = num_batch_labels if num_batch_labels is not None else valid_mask.sum()
        loss = loss_sum / denominator
        # The conventional T^2 gradient correction applies to KLD, not TVD.
        if self.metric == "kld":
            loss = loss * self.temperature**2
        return loss

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            student_logits,
            teacher_logits,
            labels,
            num_batch_labels,
            checkpoint_chunks=self.checkpoint_chunks,
        )

    def forward_no_checkpoint(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            student_logits,
            teacher_logits,
            labels,
            num_batch_labels,
            checkpoint_chunks=False,
        )


class KDLoss(_DistributionLoss):
    """TP-aware teacher-to-student Kullback-Leibler divergence."""

    metric = "kld"


class TVDLoss(_DistributionLoss):
    """TP-aware total-variation distance between teacher and student."""

    metric = "tvd"


def build_distribution_loss(metric: str, **kwargs) -> _DistributionLoss:
    metric = metric.lower()
    if metric == "kld":
        return KDLoss(**kwargs)
    if metric == "tvd":
        return TVDLoss(**kwargs)
    raise ValueError(f"Unsupported KD metric {metric!r}; expected 'kld' or 'tvd'")
