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

"""Memory-efficient (vocab-chunked) cross-entropy + knowledge-distillation losses.

This is a simplified, **forward-only** distillation of the flash-KD kernel: it
computes per-token cross-entropy (student vs. ground-truth labels) and KL
divergence ``KL(P_teacher || P_student)`` without ever materializing the full
``[num_tokens, vocab_size]`` logits/probability tensors in fp32.

Motivation: for large-vocab models (e.g. Qwen3.5 with vocab ~248k) the naive
similarity-metric path materializes several full-vocab fp32 tensors per sample
(``softmax``/``log_softmax`` run in fp32 under autocast), which OOMs at long
sequence lengths. Here we stream over the vocab dimension in chunks and keep
only ``O(num_tokens)`` running statistics, using the online-softmax (running
max + rescale) trick so the result is numerically identical to the full
computation.

Only the forward pass is implemented (scoring runs under ``torch.no_grad``);
there are no gradients, no Triton, and no tensor-parallel sharding — the loss is
computed on a single rank that holds the full vocab.
"""

import torch

__all__ = ["flash_ce_kd_loss"]


@torch.no_grad()
def flash_ce_kd_loss(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    ignore_index: int = -1,
    chunk_size: int = 16384,
    upcast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token CE and KD losses by streaming over the vocab dimension.

    The student side uses already-materialized ``student_logits`` (sliced per
    chunk); the teacher side recomputes its logits chunk-by-chunk as
    ``teacher_hidden @ teacher_lm_head_weight[chunk].T`` so the full teacher
    logits tensor is never materialized.

    Args:
        student_logits: ``[N, V]`` student logits (N = number of tokens).
        teacher_hidden: ``[N, D]`` teacher hidden states feeding its LM head.
        teacher_lm_head_weight: ``[V, D]`` teacher LM-head weight (no bias).
        labels: ``[N]`` ground-truth next-token ids; ``ignore_index`` entries
            get a CE of 0 (matching ``F.cross_entropy(..., reduction="none")``).
        temperature: KD softmax temperature (1.0 reproduces the plain ``kl_div``
            scoring metric: mean-per-token ``KL(P_T || P_S)``).
        ignore_index: label value to exclude from cross-entropy.
        chunk_size: number of vocab columns processed per step (memory knob).
        upcast: accumulate in fp32 (recommended; matches the autocast path which
            runs ``softmax``/``log_softmax`` in fp32).

    Returns:
        ``(ce_per_token, kd_per_token)``, each shape ``[N]`` in the accumulation
        dtype. ``ce_per_token`` is the standard cross-entropy; ``kd_per_token``
        is ``KL(P_T || P_S)`` per token (NOT multiplied by ``temperature**2``).
    """
    if student_logits.ndim != 2:
        raise ValueError(f"student_logits must be [N, V], got {tuple(student_logits.shape)}")
    n_tokens, vocab_size = student_logits.shape
    if teacher_lm_head_weight.shape[0] != vocab_size:
        raise ValueError(
            f"teacher_lm_head_weight vocab dim {teacher_lm_head_weight.shape[0]} "
            f"!= student vocab {vocab_size}"
        )
    device = student_logits.device
    acc_dtype = torch.float32 if upcast else student_logits.dtype

    # Running statistics (one scalar per token) -- O(N) memory, no O(N*V) tensors.
    # CE: online log-sum-exp of student logits + the student logit at the label.
    m_ce = torch.full((n_tokens,), float("-inf"), device=device, dtype=acc_dtype)
    s_ce = torch.zeros((n_tokens,), device=device, dtype=acc_dtype)
    target_logit = torch.zeros((n_tokens,), device=device, dtype=acc_dtype)
    # KD: online log-sum-exp of temperature-scaled student & teacher logits,
    # plus the online "cross" accumulator sum_v exp(z_T/T - m_T) * (z_T - z_S)/T.
    m_skd = torch.full((n_tokens,), float("-inf"), device=device, dtype=acc_dtype)
    s_skd = torch.zeros((n_tokens,), device=device, dtype=acc_dtype)
    m_tkd = torch.full((n_tokens,), float("-inf"), device=device, dtype=acc_dtype)
    s_tkd = torch.zeros((n_tokens,), device=device, dtype=acc_dtype)
    cross_tkd = torch.zeros((n_tokens,), device=device, dtype=acc_dtype)

    rows = torch.arange(n_tokens, device=device)
    valid_label = labels != ignore_index

    for c in range(0, vocab_size, chunk_size):
        cs = min(chunk_size, vocab_size - c)

        z_s = student_logits[:, c : c + cs]
        if upcast:
            z_s = z_s.float()
        # Teacher logits for this vocab slice: [N, D] @ [D, cs] -> [N, cs].
        z_t = torch.matmul(teacher_hidden, teacher_lm_head_weight[c : c + cs].transpose(0, 1))
        if upcast:
            z_t = z_t.float()

        # ---------------- Cross-entropy (student vs labels, no temperature) ----
        chunk_max = z_s.amax(dim=1)
        new_m = torch.maximum(m_ce, chunk_max)
        s_ce = s_ce * torch.exp(m_ce - new_m) + torch.exp(z_s - new_m.unsqueeze(1)).sum(dim=1)
        m_ce = new_m
        # Capture the student logit at the true label when it lands in this chunk.
        local = labels - c
        in_chunk = (local >= 0) & (local < cs) & valid_label
        if in_chunk.any():
            sel = in_chunk
            target_logit[rows[sel]] = z_s[rows[sel], local[sel]]

        # ---------------- KD: KL(P_T || P_S) at temperature ---------------------
        z_s_t = z_s / temperature
        z_t_t = z_t / temperature

        # Student temperature-scaled log-sum-exp.
        chunk_max_s = z_s_t.amax(dim=1)
        new_m_s = torch.maximum(m_skd, chunk_max_s)
        s_skd = s_skd * torch.exp(m_skd - new_m_s) + torch.exp(z_s_t - new_m_s.unsqueeze(1)).sum(dim=1)
        m_skd = new_m_s

        # Teacher temperature-scaled log-sum-exp + cross accumulator.
        chunk_max_t = z_t_t.amax(dim=1)
        new_m_t = torch.maximum(m_tkd, chunk_max_t)
        scale_prev = torch.exp(m_tkd - new_m_t)
        exp_t = torch.exp(z_t_t - new_m_t.unsqueeze(1))
        s_tkd = s_tkd * scale_prev + exp_t.sum(dim=1)
        # diff is finite everywhere here (exact slice -> no padding/-inf), so no
        # masking is needed (unlike the padded Triton kernel).
        diff = z_t_t - z_s_t
        cross_tkd = cross_tkd * scale_prev + (exp_t * diff).sum(dim=1)
        m_tkd = new_m_t

    lse_ce = m_ce + torch.log(s_ce)
    ce_per_token = torch.where(valid_label, lse_ce - target_logit, torch.zeros_like(lse_ce))

    lse_skd = m_skd + torch.log(s_skd)
    lse_tkd = m_tkd + torch.log(s_tkd)
    # KL(P_T||P_S) = E_{P_T}[(z_T - z_S)/T] + lse_S - lse_T, where the expectation
    # equals cross_tkd / s_tkd (the normalized teacher-weighted mean of diff).
    kd_per_token = (cross_tkd / s_tkd) + lse_skd - lse_tkd

    return ce_per_token, kd_per_token
