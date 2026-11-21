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

"""Offline teacher for logit distillation: train against a Megatron-LM offline KD V3 corpus.

The corpus (``modelopt.torch.distill.plugins.offline_kd_v3``, written for instance by
``dump_teacher_v3.py`` in ``examples/llm_qat``) holds each sample's inputs next to the teacher's
top-k log-probs and the log of its remaining mass at every loss position. The KD loss is the forward
KL(teacher || student) over the k stored tokens plus one bucket for the rest of the vocabulary,
which equals the full-vocabulary KL when the teacher's tail mass is zero.
"""

import torch
from torch.utils.data import Dataset

__all__ = [
    "OFFLINE_TEACHER_KEYS",
    "OfflineTeacherCollator",
    "OfflineTeacherDataset",
    "fused_linear_topk_kl",
    "topk_kl",
]

TEACHER_POSITIONS_KEY = "teacher_topk_positions"
TEACHER_IDS_KEY = "teacher_topk_ids"
TEACHER_LOGPROBS_KEY = "teacher_topk_logprobs"
TEACHER_TAIL_KEY = "teacher_topk_tail"
OFFLINE_TEACHER_KEYS = (
    TEACHER_POSITIONS_KEY,
    TEACHER_IDS_KEY,
    TEACHER_LOGPROBS_KEY,
    TEACHER_TAIL_KEY,
)

IGNORE_INDEX = -100


def _tail_terms(log_q_k, log_q_tail, teacher_logprobs, teacher_tail):
    """Per-row forward KL over the k stored ids plus the remaining-vocabulary bucket."""
    p = teacher_logprobs.exp()
    p_tail = teacher_tail.exp()
    kl = (p * (teacher_logprobs - log_q_k)).sum(-1)
    # An empty tail (none left in the teacher, or k == V) contributes 0, not 0 * inf.
    has_tail = (p_tail > 0) & torch.isfinite(log_q_tail)
    tail = torch.where(has_tail, p_tail * (teacher_tail - log_q_tail), torch.zeros_like(p_tail))
    return kl + tail, p, p_tail, has_tail


def _student_log_probs(logits, ids):
    """Log q at ``ids``, log of the student mass outside them, and the log-partition (fp32)."""
    lse = logits.logsumexp(-1)
    log_q_k = logits.gather(-1, ids) - lse[:, None]
    # Log-sum-exp over the complement instead of log(1 - sum q_k), which cancels when q_k ~ 1.
    log_q_tail = logits.scatter(-1, ids, float("-inf")).logsumexp(-1) - lse
    return log_q_k, log_q_tail, lse


def topk_kl(student_logits, teacher_ids, teacher_logprobs, teacher_tail):
    """Summed top-k forward KL over rows, differentiable through ``student_logits`` [N, V]."""
    logits = student_logits.float()
    ids = teacher_ids.long()
    log_q_k, log_q_tail, _ = _student_log_probs(logits, ids)
    kl, *_ = _tail_terms(log_q_k, log_q_tail, teacher_logprobs.float(), teacher_tail.float())
    return kl.sum()


class _FusedLinearTopKKL(torch.autograd.Function):
    """``topk_kl(hidden @ weight[:vocab_size].T, ...)`` without materializing [N, V] logits.

    Like Liger's fused linear losses, the gradients are computed chunk by chunk in the forward and
    only scaled in the backward, so the sharded ``weight`` is needed only while forward runs.
    """

    @staticmethod
    def forward(
        ctx, hidden, weight, teacher_ids, teacher_logprobs, teacher_tail, vocab_size, chunk
    ):
        # Rows past the tokenizer's vocabulary are output-layer padding, which the targets exclude.
        vocab_weight = weight[:vocab_size]
        grad_hidden = torch.empty_like(hidden) if hidden.requires_grad else None
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if weight.requires_grad
            else None
        )
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], chunk):
            end = start + chunk
            h = hidden[start:end]
            # bf16 matmul then fp32, as Liger computes logits for the online KD loss.
            logits = (h @ vocab_weight.t()).float()
            ids = teacher_ids[start:end].long()
            log_q_k, log_q_tail, lse = _student_log_probs(logits, ids)
            tail_logprob = teacher_tail[start:end].float()
            kl, p, p_tail, has_tail = _tail_terms(
                log_q_k, log_q_tail, teacher_logprobs[start:end].float(), tail_logprob
            )
            loss += kl.sum()

            # d kl / d logits = mass * q - target: target is p on the stored ids and spreads the
            # teacher tail over the student's own tail in proportion to q, and mass is the target's
            # total, which the fp16 log-probs and fp32 tail leave a hair off 1.
            log_q = logits - lse[:, None]
            # In log space, so an empty tail gives exp(-inf) = 0 instead of 0 * inf.
            tail_shift = torch.where(
                has_tail, tail_logprob - log_q_tail, torch.full_like(p_tail, float("-inf"))
            )
            target = (log_q + tail_shift[:, None]).exp()
            target.scatter_(-1, ids, p)
            mass = p.sum(-1) + torch.where(has_tail, p_tail, torch.zeros_like(p_tail))
            grad_logits = log_q.exp_().mul_(mass[:, None]).sub_(target).to(h.dtype)
            if grad_hidden is not None:
                grad_hidden[start:end] = grad_logits @ vocab_weight
            if grad_weight is not None:
                grad_weight[:vocab_size].addmm_(grad_logits.t().float(), h.float())
        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.weight_dtype = weight.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        if grad_hidden is not None:
            grad_hidden = grad_hidden * grad_output.to(grad_hidden.dtype)
        if grad_weight is not None:
            grad_weight = (grad_weight * grad_output).to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None, None, None, None, None


def fused_linear_topk_kl(
    hidden, weight, teacher_ids, teacher_logprobs, teacher_tail, vocab_size=None, chunk_size=512
):
    """Summed top-k forward KL of ``hidden @ weight[:vocab_size].T`` against the teacher rows."""
    vocab_size = weight.shape[0] if vocab_size is None else vocab_size
    return _FusedLinearTopKKL.apply(
        hidden, weight, teacher_ids, teacher_logprobs, teacher_tail, vocab_size, chunk_size
    )


class OfflineTeacherDataset(Dataset):
    """Training rows with their teacher top-k targets, both read from one V3 corpus source.

    Each sample becomes the Hugging Face row it was written from, ending at its last loss token:
    input ids are its tokens plus its last label, labels are its labels shifted back by one, and the
    row is padded to ``max_length`` like the ``examples/llm_qat`` rows. A sample without loss becomes
    a fully padded row, as ``examples/llm_qat`` makes for such rows, so the batches match those of
    online KD. The teacher targets cover the loss positions t, whose next token t + 1 carries loss;
    only those are carried, as a dense [L, k] tensor would be mostly unread at large k.
    """

    def __init__(self, source: str, tokenizer, max_length: int, top_k: int | None = None):
        """Index ``source``; ``top_k`` keeps the first k targets, folding the rest into the tail."""
        from .offline_kd_v3 import CorpusSource  # needs pyarrow, which online KD does not

        self.source = CorpusSource(source, tokenizer)
        if self.source.settings["contract"]["attention_mode"] != "isolated_segments":
            raise ValueError(f"{source} packs samples, which Hugging Face rows cannot isolate.")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        self.top_k = top_k
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        sample = self.source[i]
        tokens = torch.from_numpy(sample["tokens"]).long()
        labels = torch.from_numpy(sample["labels"]).long()
        loss_mask = torch.from_numpy(sample["loss_mask"])
        length = tokens.numel() + 1 if tokens.numel() else 0
        if length > self.max_length:
            raise ValueError(
                f"Sample {i} needs {length} tokens, over max_length {self.max_length}."
            )
        if length and (len(sample["segments"]) != 1 or not torch.equal(labels[:-1], tokens[1:])):
            raise ValueError(f"Sample {i} is not one unpacked row with next-token labels.")
        input_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        row_labels = torch.full((self.max_length,), IGNORE_INDEX, dtype=torch.long)
        if length:
            input_ids[: length - 1] = tokens
            input_ids[length - 1] = labels[-1]
            attention_mask[:length] = 1
            row_labels[1:length] = torch.where(loss_mask, labels, IGNORE_INDEX)

        values = torch.from_numpy(sample["values"])
        indices = torch.from_numpy(sample["indices"])
        tail = torch.from_numpy(sample["tail"])
        if self.top_k is not None and self.top_k > values.shape[-1]:
            raise ValueError(f"top_k {self.top_k} exceeds the {values.shape[-1]} stored targets.")
        if self.top_k is not None and self.top_k < values.shape[-1]:
            # Targets are sorted by log-prob, so the first k are a top-k dump with a larger tail.
            tail = torch.logaddexp(tail, values[:, self.top_k :].float().logsumexp(-1))
            values, indices = values[:, : self.top_k], indices[:, : self.top_k]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": row_labels,
            TEACHER_POSITIONS_KEY: loss_mask.nonzero().flatten(),
            TEACHER_IDS_KEY: indices,
            TEACHER_LOGPROBS_KEY: values,
            TEACHER_TAIL_KEY: tail,
        }


class OfflineTeacherCollator:
    """Wraps a collator and pads the per-row teacher top-k to the batch's most loss positions.

    Padded entries have position -1.
    """

    def __init__(self, collator):
        """Wrap ``collator``, which batches everything but the teacher columns."""
        self.collator = collator

    def __call__(self, features):
        """Collate ``features`` and add the padded teacher positions, ids, log-probs and tails."""
        teacher = [{k: f.pop(k) for k in OFFLINE_TEACHER_KEYS} for f in features]
        batch = self.collator(features)
        num = max(t[TEACHER_POSITIONS_KEY].numel() for t in teacher)
        positions = torch.full((len(teacher), num), -1, dtype=torch.int64)
        padded = {
            key: teacher[0][key].new_zeros(len(teacher), num, *teacher[0][key].shape[1:])
            for key in (TEACHER_IDS_KEY, TEACHER_LOGPROBS_KEY, TEACHER_TAIL_KEY)
        }
        for b, t in enumerate(teacher):
            n = t[TEACHER_POSITIONS_KEY].numel()
            positions[b, :n] = t[TEACHER_POSITIONS_KEY]
            for key, value in padded.items():
                value[b, :n] = t[key]
        batch[TEACHER_POSITIONS_KEY] = positions
        batch.update(padded)
        return batch
