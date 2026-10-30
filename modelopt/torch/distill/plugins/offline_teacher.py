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

"""Offline teacher for logit distillation: precomputed top-k teacher log-probs.

The teacher's next-token distribution is stored as its top-k token ids and log-probs per loss
position (``dump_teacher_topk.py`` in ``examples/llm_qat`` writes them). The KD loss is the forward
KL(teacher || student) over the k stored tokens plus one bucket for the rest of the vocabulary,
which equals the full-vocabulary KL when the teacher's tail mass is zero.

Dump layout (one directory per split, plus ``meta.json`` at the root)::

    <dir>/meta.json
    <dir>/<split>/shard-00000.safetensors   rows [0, shard_size), shard-00001 the next ones, ...

Each shard holds, for its ``n`` dataset rows and ``P`` loss positions in total:

    row        int64 [n]       dataset row index
    num_tokens int64 [n]       unpadded length of the row
    checksum   int64 [n]       ``input_checksum`` of the unpadded input ids
    offsets    int64 [n + 1]   row i owns positions[offsets[i]:offsets[i + 1]]
    positions  int64 [P]       position t whose next token t + 1 is a loss token
    ids        int32 [P, k]    teacher top-k token ids at t, by descending log-prob
    logprobs   fp16  [P, k]    teacher log-probs of those ids (temperature 1)
"""

import json
import os
from functools import lru_cache

import torch
from torch.utils.data import Dataset

__all__ = [
    "OFFLINE_TEACHER_KEYS",
    "OfflineTeacherCollator",
    "OfflineTeacherDataset",
    "fused_linear_topk_kl",
    "input_checksum",
    "topk_kl",
]

TEACHER_POSITIONS_KEY = "teacher_topk_positions"
TEACHER_IDS_KEY = "teacher_topk_ids"
TEACHER_LOGPROBS_KEY = "teacher_topk_logprobs"
OFFLINE_TEACHER_KEYS = (TEACHER_POSITIONS_KEY, TEACHER_IDS_KEY, TEACHER_LOGPROBS_KEY)

_CHECKSUM_MOD = (1 << 61) - 1


def input_checksum(input_ids: torch.Tensor) -> int:
    """Position-weighted checksum of unpadded input ids, to catch a dump paired with other data."""
    ids = input_ids.to(torch.int64).flatten()
    weights = torch.arange(1, ids.numel() + 1, dtype=torch.int64)
    return int(((ids * weights) % _CHECKSUM_MOD).sum() % _CHECKSUM_MOD)


def _tail_terms(log_q_k, log_q_tail, teacher_logprobs):
    """Per-row forward KL over the k stored ids plus the remaining-vocabulary bucket."""
    p = teacher_logprobs.exp()
    # fp16 log-probs can sum a hair above 1; the tail is then empty.
    p_tail = (1.0 - p.sum(-1)).clamp_min(0.0)
    kl = (p * (teacher_logprobs - log_q_k)).sum(-1)
    # An empty tail (none left in the teacher, or k == V) contributes 0, not 0 * inf.
    has_tail = (p_tail > 0) & torch.isfinite(log_q_tail)
    tail = torch.where(has_tail, p_tail * (p_tail.log() - log_q_tail), torch.zeros_like(p_tail))
    return kl + tail, p, p_tail, has_tail


def _student_log_probs(logits, ids):
    """Log q at ``ids``, log of the student mass outside them, and the log-partition (fp32)."""
    lse = logits.logsumexp(-1)
    log_q_k = logits.gather(-1, ids) - lse[:, None]
    # Log-sum-exp over the complement instead of log(1 - sum q_k), which cancels when q_k ~ 1.
    log_q_tail = logits.scatter(-1, ids, float("-inf")).logsumexp(-1) - lse
    return log_q_k, log_q_tail, lse


def topk_kl(student_logits, teacher_ids, teacher_logprobs):
    """Summed top-k forward KL over rows, differentiable through ``student_logits`` [N, V]."""
    logits = student_logits.float()
    ids = teacher_ids.long()
    log_q_k, log_q_tail, _ = _student_log_probs(logits, ids)
    kl, *_ = _tail_terms(log_q_k, log_q_tail, teacher_logprobs.float())
    return kl.sum()


class _FusedLinearTopKKL(torch.autograd.Function):
    """``topk_kl(hidden @ weight.T, ...)`` without materializing [N, V] logits.

    Like Liger's fused linear losses, the gradients are computed chunk by chunk in the forward and
    only scaled in the backward, so the sharded ``weight`` is needed only while forward runs.
    """

    @staticmethod
    def forward(ctx, hidden, weight, teacher_ids, teacher_logprobs, chunk_size):
        grad_hidden = torch.empty_like(hidden) if hidden.requires_grad else None
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if weight.requires_grad
            else None
        )
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], chunk_size):
            end = start + chunk_size
            h = hidden[start:end]
            # bf16 matmul then fp32, as Liger computes logits for the online KD loss.
            logits = (h @ weight.t()).float()
            ids = teacher_ids[start:end].long()
            log_q_k, log_q_tail, lse = _student_log_probs(logits, ids)
            kl, p, p_tail, has_tail = _tail_terms(
                log_q_k, log_q_tail, teacher_logprobs[start:end].float()
            )
            loss += kl.sum()

            # d kl / d logits = q - target, where target = p on the stored ids and spreads the
            # teacher tail over the student's own tail in proportion to q.
            log_q = logits - lse[:, None]
            # In log space, so an empty tail gives exp(-inf) = 0 instead of 0 * inf.
            tail_shift = torch.where(
                has_tail, p_tail.log() - log_q_tail, torch.full_like(p_tail, float("-inf"))
            )
            q = log_q.exp()
            target = (log_q + tail_shift[:, None]).exp()
            target.scatter_(-1, ids, p)
            grad_logits = (q - target).to(h.dtype)
            if grad_hidden is not None:
                grad_hidden[start:end] = grad_logits @ weight
            if grad_weight is not None:
                grad_weight.addmm_(grad_logits.t().float(), h.float())
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
        return grad_hidden, grad_weight, None, None, None


def fused_linear_topk_kl(hidden, weight, teacher_ids, teacher_logprobs, chunk_size=512):
    """Summed top-k forward KL of ``hidden @ weight.T`` against the stored teacher rows."""
    return _FusedLinearTopKKL.apply(hidden, weight, teacher_ids, teacher_logprobs, chunk_size)


@lru_cache(maxsize=8)
def _open_shard(path):
    from safetensors import safe_open

    return safe_open(path, framework="pt")


class OfflineTeacherDataset(Dataset):
    """A tokenized dataset with each row's dumped teacher top-k attached.

    Adds the loss positions t (the teacher distribution for token t + 1, matching the causal shift
    of the KD loss) and their [P, k] top-k ids and log-probs. Only loss positions are carried: a
    dense [L, k] tensor would be mostly unread rows at large k.
    """

    def __init__(self, base, teacher_dir: str, split: str, top_k: int | None = None):
        """Index ``teacher_dir/<split>`` by dataset row; ``top_k`` keeps only the first k of each row."""
        self.base = base
        with open(os.path.join(teacher_dir, "meta.json")) as f:
            self.meta = json.load(f)
        # The dump keeps each position's ids sorted by log-prob and its log-probs normalized over the full
        # vocabulary, so its first k columns are exactly a top-k dump; the loss rebuilds the tail.
        if top_k is not None and not 0 < top_k <= self.meta["top_k"]:
            raise ValueError(
                f"top_k must be in [1, {self.meta['top_k']}] for {teacher_dir}, got {top_k}"
            )
        self.top_k = top_k
        split_dir = os.path.join(teacher_dir, split)
        self._index = {}
        for name in sorted(os.listdir(split_dir)):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(split_dir, name)
            shard = _open_shard(path)
            rows = shard.get_tensor("row").tolist()
            for local, row in enumerate(rows):
                self._index[row] = (path, local)
        missing = [i for i in range(len(base)) if i not in self._index]
        if missing:
            raise ValueError(
                f"{split_dir} has no teacher outputs for {len(missing)} of {len(base)} rows "
                f"(first: {missing[:5]}); rerun the dump."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        path, local = self._index[i]
        shard = _open_shard(path)
        input_ids = torch.as_tensor(item["input_ids"])
        num_tokens = int(torch.as_tensor(item["attention_mask"]).sum())
        expected = (
            int(shard.get_slice("num_tokens")[local : local + 1][0]),
            int(shard.get_slice("checksum")[local : local + 1][0]),
        )
        if expected != (num_tokens, input_checksum(input_ids[:num_tokens])):
            raise ValueError(f"Teacher dump row {i} ({path}) does not match the dataset row.")
        lo, hi = shard.get_slice("offsets")[local : local + 2].tolist()
        item[TEACHER_POSITIONS_KEY] = shard.get_slice("positions")[lo:hi]
        k = self.top_k or self.meta["top_k"]
        item[TEACHER_IDS_KEY] = shard.get_slice("ids")[lo:hi, :k]
        item[TEACHER_LOGPROBS_KEY] = shard.get_slice("logprobs")[lo:hi, :k]
        return item


class OfflineTeacherCollator:
    """Wraps a collator and pads the per-row teacher top-k to the batch's most loss positions.

    Padded entries have position -1.
    """

    def __init__(self, collator):
        """Wrap ``collator``, which batches everything but the teacher columns."""
        self.collator = collator

    def __call__(self, features):
        """Collate ``features`` and add the padded teacher positions, ids and log-probs."""
        teacher = [{k: f.pop(k) for k in OFFLINE_TEACHER_KEYS} for f in features]
        batch = self.collator(features)
        num = max(t[TEACHER_POSITIONS_KEY].numel() for t in teacher)
        first_ids, first_logprobs = teacher[0][TEACHER_IDS_KEY], teacher[0][TEACHER_LOGPROBS_KEY]
        positions = torch.full((len(teacher), num), -1, dtype=torch.int64)
        ids = first_ids.new_zeros(len(teacher), num, first_ids.shape[-1])
        logprobs = first_logprobs.new_zeros(len(teacher), num, first_logprobs.shape[-1])
        for b, t in enumerate(teacher):
            n = t[TEACHER_POSITIONS_KEY].numel()
            positions[b, :n] = t[TEACHER_POSITIONS_KEY]
            ids[b, :n] = t[TEACHER_IDS_KEY]
            logprobs[b, :n] = t[TEACHER_LOGPROBS_KEY]
        batch[TEACHER_POSITIONS_KEY] = positions
        batch[TEACHER_IDS_KEY] = ids
        batch[TEACHER_LOGPROBS_KEY] = logprobs
        return batch
