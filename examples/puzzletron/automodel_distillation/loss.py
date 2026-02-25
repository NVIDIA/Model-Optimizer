# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard


def _infer_tp_group_from_dtensor(tensor: "torch.Tensor"):
    """Return device_mesh process group if tensor is a DTensor sharded on vocab (logits last dim, lm_head dim 0)."""
    if not isinstance(tensor, DTensor):
        return None
    # Vocab sharding: Shard on last dim (logits) or Shard(0) (weight matrix)
    has_shard = any(isinstance(p, Shard) for p in tensor.placements)
    if not has_shard:
        return None
    return tensor.device_mesh.get_group()


def _kl_forward_tp(
    t_logits: torch.Tensor,
    s_logits: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """
    Compute KL (negative cross entropy sum(P*log Q)) with tensor parallelism.
    Returns per-token negative cross entropy (sum over vocab).
    """
    teacher_max = t_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(teacher_max, op=dist.ReduceOp.MAX, group=tp_group)
    output_teacher = t_logits - teacher_max

    denom_teacher = torch.exp(output_teacher).sum(dim=-1, keepdim=True)
    dist.all_reduce(denom_teacher, op=dist.ReduceOp.SUM, group=tp_group)
    teacher_prob = torch.exp(output_teacher - torch.log(denom_teacher.clamp(min=1e-12)))

    student_max = s_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(student_max, op=dist.ReduceOp.MAX, group=tp_group)
    output_student = s_logits - student_max.detach()

    denom_student = torch.exp(output_student).sum(dim=-1, keepdim=True)
    dist.all_reduce(denom_student, op=dist.ReduceOp.SUM, group=tp_group)
    student_log_prob = output_student - torch.log(denom_student.clamp(min=1e-12))

    term = teacher_prob * student_log_prob
    inf_mask = torch.isinf(s_logits)
    term = torch.masked_fill(term, inf_mask, 0.0)
    ce_local = term.sum(dim=-1)
    dist.all_reduce(ce_local, op=dist.ReduceOp.SUM, group=tp_group)
    return ce_local.view(-1)


class KDLoss(nn.Module):
    """TP-aware KD on precomputed logits."""

    def __init__(
        self,
        ignore_index: int = -100,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        tp_group=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast
        self.tp_group = tp_group

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        valid_mask = (labels != self.ignore_index).view(-1)
        if valid_mask.sum() == 0:
            return student_logits.new_tensor(0.0)

        if student_logits.ndim > 2:
            student_logits = student_logits.view(-1, student_logits.shape[-1])
        if teacher_logits.ndim > 2:
            teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        if labels.ndim > 1:
            labels = labels.view(-1)

        tp_group = self.tp_group
        if isinstance(student_logits, DTensor) and tp_group is None:
            tp_group = _infer_tp_group_from_dtensor(student_logits)

        if tp_group is not None:
            if isinstance(student_logits, DTensor):
                student_logits = student_logits.to_local()
            if isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.to_local()
        else:
            if isinstance(student_logits, DTensor):
                student_logits = student_logits.full_tensor()
            if isinstance(teacher_logits, DTensor):
                teacher_logits = teacher_logits.full_tensor()

        t_logits = teacher_logits[valid_mask]
        s_logits = student_logits[valid_mask]

        if self.fp32_upcast:
            t_logits = t_logits.float()
            s_logits = s_logits.float()
        if self.temperature != 1.0:
            t_logits = t_logits.mul(1.0 / self.temperature)
            s_logits = s_logits.mul(1.0 / self.temperature)

        if tp_group is not None:
            kl_per_token = _kl_forward_tp(t_logits, s_logits, tp_group)
        else:
            teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
            student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)
            inf_mask = torch.isinf(s_logits)
            kl_per_token = (
                torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0.0).sum(-1).view(-1)
            )

        if self.temperature != 1.0:
            kl_per_token = kl_per_token * (self.temperature**2)

        if num_batch_labels is not None:
            return -torch.sum(kl_per_token) / num_batch_labels
        return -torch.mean(kl_per_token)
