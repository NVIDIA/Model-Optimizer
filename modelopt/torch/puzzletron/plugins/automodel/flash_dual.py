# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dual-side vocab-streaming CE / KD / top-k for AutoModel replace-1-block scoring.

Generalizes ``utils.flash_kd.flash_ce_kd_loss`` (which streams only the teacher
side and needs the student logits materialized) so that **both** sides are
reconstructed chunk-by-chunk from final hidden states and LM-head weights. This is
what the AutoModel path needs: under NeMo the model produces final hidden states
(logits may never be materialized, and under TP would be vocab-sharded), so we
capture ``hidden`` + the ``lm_head`` weight and never build a full ``[N, V]`` tensor
for either model.

All reductions match the legacy metrics:
* ``ce_per_token`` = standard cross-entropy (``ignore_index`` -> 0),
* ``kd_per_token`` = ``KL(P_T || P_S)`` per token (no ``temperature**2`` factor),
* ``topk_agreement`` = fraction of the teacher's global top-k retained by the student.

Online log-sum-exp keeps everything O(N) in memory. Verified on CPU against a naive
full-logit reference (test_automodel_flash_dual.py).
"""

import torch
import torch.nn.functional as F

__all__ = [
    "flash_dual_ce_kd",
    "flash_kld_ce_topk",
    "topk_accuracy_and_agreement_from_hidden",
    "topk_agreement_from_hidden",
    "topk_hit_from_hidden",
]


@torch.no_grad()
def flash_kld_ce_topk(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    ignore_index: int = -1,
    teacher_identity: bool = False,
    tp_group=None,
    token_chunk_size: int | None = None,
    reduction_backend: str = "fla",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    """Exact forward-only FLA FlashKLD with CE and top-k from the same logits.

    FLA's public ``FusedKLDivLoss`` also allocates backward buffers.  Scoring
    needs no gradients, so this wrapper calls its Triton forward kernel directly
    and retains only per-token losses.  Tokens are chunked while the complete LM
    head remains supported; no vocabulary approximation is used.
    """
    import triton

    from fla.modules.fused_cross_entropy import fused_cross_entropy_forward
    from fla.modules.fused_kl_div import MAX_FUSED_SIZE, STATIC_WARPS, kl_div_kernel

    import torch.distributed as torch_dist

    n_tokens, hidden_size = student_hidden.shape
    vocab_size = student_lm_head_weight.shape[0]
    if teacher_lm_head_weight.shape[0] != vocab_size:
        raise ValueError(
            "FlashKLD requires matching teacher/student vocabularies, got "
            f"student={vocab_size} teacher={teacher_lm_head_weight.shape[0]}"
        )
    if teacher_hidden.shape[0] != n_tokens:
        raise ValueError(
            "FlashKLD requires matching teacher/student token counts, got "
            f"student={n_tokens} teacher={teacher_hidden.shape[0]}"
        )
    if student_lm_head_weight.shape[1] != hidden_size:
        raise ValueError(
            "FlashKLD student hidden/head width mismatch: "
            f"hidden={hidden_size} head={student_lm_head_weight.shape[1]}"
        )
    if teacher_lm_head_weight.shape[1] != teacher_hidden.shape[1]:
        raise ValueError(
            "FlashKLD teacher hidden/head width mismatch: "
            f"hidden={teacher_hidden.shape[1]} head={teacher_lm_head_weight.shape[1]}"
        )

    tp_size = torch_dist.get_world_size(tp_group) if tp_group is not None else 1
    tp_rank = torch_dist.get_rank(tp_group) if tp_group is not None else 0
    tokens_per_tp = (n_tokens + tp_size - 1) // tp_size
    tp_start = tp_rank * tokens_per_tp
    tp_end = min(tp_start + tokens_per_tp, n_tokens)
    local_tokens = tp_end - tp_start
    student_hidden = student_hidden[tp_start:tp_end]
    teacher_hidden = teacher_hidden[tp_start:tp_end]
    labels = labels[tp_start:tp_end]

    block_vocab = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
    if token_chunk_size is None:
        num_chunks = min(8, triton.cdiv(vocab_size, hidden_size))
        token_chunk = triton.next_power_of_2(triton.cdiv(local_tokens, num_chunks))
    else:
        token_chunk = min(max(1, int(token_chunk_size)), local_tokens)
    num_chunks = triton.cdiv(local_tokens, token_chunk)

    ce = torch.empty(local_tokens, dtype=torch.float32, device=student_hidden.device)
    kd = torch.zeros(local_tokens, dtype=torch.float32, device=student_hidden.device)
    hits = {
        1: torch.empty(local_tokens, dtype=torch.float32, device=student_hidden.device),
        5: torch.empty(local_tokens, dtype=torch.float32, device=student_hidden.device),
        10: torch.empty(local_tokens, dtype=torch.float32, device=student_hidden.device),
    }
    label_hits = {top_k: torch.empty_like(hit) for top_k, hit in hits.items()}

    inv_temperature = 1.0 / float(temperature)
    for chunk_index in range(num_chunks):
        start = chunk_index * token_chunk
        end = min((chunk_index + 1) * token_chunk, local_tokens)
        student_logits = F.linear(student_hidden[start:end], student_lm_head_weight)

        chunk_ce, _, _, _, _ = fused_cross_entropy_forward(
            student_logits,
            labels[start:end],
            ignore_index=ignore_index,
        )
        ce[start:end] = chunk_ce

        teacher_logits = (
            student_logits
            if teacher_identity
            else F.linear(teacher_hidden[start:end], teacher_lm_head_weight)
        )
        student_top = student_logits.topk(10, dim=-1).indices
        teacher_top = teacher_logits.topk(10, dim=-1).indices
        for top_k in hits:
            overlap = (
                student_top[:, :top_k, None] == teacher_top[:, None, :top_k]
            ).any(dim=-1)
            hits[top_k][start:end] = overlap.float().mean(dim=-1)
            label_hits[top_k][start:end] = (
                student_top[:, :top_k] == labels[start:end, None]
            ).any(dim=-1).float()

        if teacher_identity:
            continue
        if temperature != 1.0:
            student_logits.mul_(inv_temperature)
            teacher_logits.mul_(inv_temperature)
        chunk_kd = kd[start:end]
        if reduction_backend == "fla":
            kl_div_kernel[(end - start,)](
                logits=student_logits,
                target_logits=teacher_logits,
                loss=chunk_kd,
                s_logits=student_logits.stride(-2),
                s_loss=chunk_kd.stride(-1),
                reduction="none",
                N=local_tokens,
                V=vocab_size,
                BV=block_vocab,
                num_warps=STATIC_WARPS,
            )
        elif reduction_backend == "vectorized_exact":
            student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
            chunk_kd.copy_(
                F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)
            )
        else:
            raise ValueError(f"Unknown FlashKLD reduction_backend={reduction_backend!r}")

    if tp_size > 1:
        def gather_tokens(value: torch.Tensor) -> torch.Tensor:
            if value.shape[0] < tokens_per_tp:
                value = F.pad(value, (0, tokens_per_tp - value.shape[0]))
            gathered = [torch.empty_like(value) for _ in range(tp_size)]
            torch_dist.all_gather(gathered, value.contiguous(), group=tp_group)
            return torch.cat(gathered, dim=0)[:n_tokens]

        ce = gather_tokens(ce)
        kd = gather_tokens(kd)
        hits = {top_k: gather_tokens(hit) for top_k, hit in hits.items()}
        label_hits = {
            top_k: gather_tokens(hit) for top_k, hit in label_hits.items()
        }

    return ce, kd, hits, label_hits


def _chunk_logits(hidden: torch.Tensor, lm_head_weight: torch.Tensor, c: int, cs: int, upcast):
    """Student/teacher logits for vocab slice ``[c, c+cs)``: ``hidden @ W[c:c+cs].T``."""
    z = torch.matmul(hidden, lm_head_weight[c : c + cs].transpose(0, 1))
    return z.float() if upcast else z


def flash_dual_ce_kd(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    ignore_index: int = -1,
    chunk_size: int = 16384,
    upcast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token CE (student vs labels) and KD ``KL(P_T||P_S)``, streaming both sides over vocab.

    Args:
        student_hidden / teacher_hidden: ``[N, D]`` final hidden states feeding each LM head.
        student_lm_head_weight / teacher_lm_head_weight: ``[V, D]`` LM-head weights (no bias).
        labels: ``[N]`` next-token ids; ``ignore_index`` -> CE 0.
        temperature: KD softmax temperature (1.0 reproduces the plain ``kl_div`` metric).
    Returns:
        ``(ce_per_token, kd_per_token)`` each ``[N]`` in the accumulation dtype.
    """
    n_tokens, vocab_size = student_hidden.shape[0], student_lm_head_weight.shape[0]
    if teacher_lm_head_weight.shape[0] != vocab_size:
        raise ValueError(
            f"teacher vocab {teacher_lm_head_weight.shape[0]} != student vocab {vocab_size}"
        )
    device = student_hidden.device
    acc = torch.float32 if upcast else student_hidden.dtype

    # CE online log-sum-exp over student logits + the student logit at the label.
    m_ce = torch.full((n_tokens,), float("-inf"), device=device, dtype=acc)
    s_ce = torch.zeros((n_tokens,), device=device, dtype=acc)
    target_logit = torch.zeros((n_tokens,), device=device, dtype=acc)
    # KD: online lse of temperature-scaled student & teacher logits + the cross term
    # sum_v softmax(z_T/T)_v * (z_T - z_S)/T  (= KL(P_T||P_S) after normalization).
    m_s = torch.full((n_tokens,), float("-inf"), device=device, dtype=acc)
    s_s = torch.zeros((n_tokens,), device=device, dtype=acc)
    m_t = torch.full((n_tokens,), float("-inf"), device=device, dtype=acc)
    s_t = torch.zeros((n_tokens,), device=device, dtype=acc)
    cross = torch.zeros((n_tokens,), device=device, dtype=acc)

    rows = torch.arange(n_tokens, device=device)
    valid = labels != ignore_index
    inv_t = 1.0 / temperature

    for c in range(0, vocab_size, chunk_size):
        cs = min(chunk_size, vocab_size - c)
        z_s = _chunk_logits(student_hidden, student_lm_head_weight, c, cs, upcast)
        z_t = _chunk_logits(teacher_hidden, teacher_lm_head_weight, c, cs, upcast)

        # CE (student, no temperature).
        cmax = z_s.amax(dim=1)
        new_m = torch.maximum(m_ce, cmax)
        s_ce = s_ce * torch.exp(m_ce - new_m) + torch.exp(z_s - new_m.unsqueeze(1)).sum(dim=1)
        m_ce = new_m
        in_chunk = valid & (labels >= c) & (labels < c + cs)
        if in_chunk.any():
            idx = labels[in_chunk] - c
            target_logit[in_chunk] = z_s[in_chunk, idx]

        # KD (temperature-scaled). Maintain separate running lse for student & teacher,
        # and a running cross accumulator rescaled to the teacher's running max.
        zs_t = z_s * inv_t
        zt_t = z_t * inv_t
        # student lse
        cmax_s = zs_t.amax(dim=1)
        nm_s = torch.maximum(m_s, cmax_s)
        s_s = s_s * torch.exp(m_s - nm_s) + torch.exp(zs_t - nm_s.unsqueeze(1)).sum(dim=1)
        m_s = nm_s
        # teacher lse + cross term sum_v exp(zt_t - m_t) * (zt_t - zs_t)
        cmax_t = zt_t.amax(dim=1)
        nm_t = torch.maximum(m_t, cmax_t)
        rescale = torch.exp(m_t - nm_t)
        s_t = s_t * rescale + torch.exp(zt_t - nm_t.unsqueeze(1)).sum(dim=1)
        w_t = torch.exp(zt_t - nm_t.unsqueeze(1))  # [N, cs] unnormalized teacher weights
        cross = cross * rescale + (w_t * (zt_t - zs_t)).sum(dim=1)
        m_t = nm_t

    log_Z_ce = m_ce + torch.log(s_ce)
    ce_per_token = torch.where(valid, log_Z_ce - target_logit, torch.zeros_like(target_logit))

    # KL(P_T||P_S) = sum_v p_T (log p_T - log p_S)
    #             = (1/Z_T) sum_v exp(zt_t - m_t) * (zt_t - zs_t) + log_Z_S - log_Z_T
    log_Z_s = m_s + torch.log(s_s)
    log_Z_t = m_t + torch.log(s_t)
    kd_per_token = cross / s_t + log_Z_s - log_Z_t
    return ce_per_token, kd_per_token


def topk_hit_from_hidden(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
    *,
    chunk_size: int = 16384,
    upcast: bool = True,
) -> torch.Tensor:
    """Per-token boolean: is ``labels`` within the student's global top-``k`` over the vocab.

    Online merge of per-chunk top-k logits/indices so no full ``[N, V]`` tensor is built.
    """
    n_tokens = student_hidden.shape[0]
    vocab_size = student_lm_head_weight.shape[0]
    device = student_hidden.device
    best_vals: torch.Tensor | None = None
    best_idx: torch.Tensor | None = None
    for c in range(0, vocab_size, chunk_size):
        cs = min(chunk_size, vocab_size - c)
        z = _chunk_logits(student_hidden, student_lm_head_weight, c, cs, upcast)
        k = min(top_k, cs)
        vals, idx = z.topk(k, dim=-1)
        idx = idx + c
        if best_vals is None:
            best_vals, best_idx = vals, idx
        else:
            cat_v = torch.cat([best_vals, vals], dim=-1)
            cat_i = torch.cat([best_idx, idx], dim=-1)
            kk = min(top_k, cat_v.shape[-1])
            top_v, sel = cat_v.topk(kk, dim=-1)
            best_vals = top_v
            best_idx = torch.gather(cat_i, -1, sel)
    rows = torch.arange(n_tokens, device=device)  # noqa: F841 (kept for clarity)
    return (best_idx == labels.unsqueeze(-1)).any(dim=-1)


def _topk_indices_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    top_k: int,
    *,
    chunk_size: int,
    upcast: bool = True,
) -> torch.Tensor:
    vocab_size = lm_head_weight.shape[0]
    best_vals: torch.Tensor | None = None
    best_idx: torch.Tensor | None = None
    for start in range(0, vocab_size, chunk_size):
        size = min(chunk_size, vocab_size - start)
        logits = _chunk_logits(hidden, lm_head_weight, start, size, upcast)
        values, indices = logits.topk(min(top_k, size), dim=-1)
        indices = indices + start
        if best_vals is None:
            best_vals, best_idx = values, indices
            continue
        merged_values = torch.cat((best_vals, values), dim=-1)
        merged_indices = torch.cat((best_idx, indices), dim=-1)
        selected_values, selected = merged_values.topk(
            min(top_k, merged_values.shape[-1]), dim=-1
        )
        best_vals = selected_values
        best_idx = torch.gather(merged_indices, -1, selected)
    if best_idx is None:
        raise ValueError("LM head has an empty vocabulary")
    return best_idx


def topk_agreement_from_hidden(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    top_k: int,
    *,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Per-token fraction of teacher top-k token IDs present in student top-k."""
    student_top = _topk_indices_from_hidden(
        student_hidden,
        student_lm_head_weight,
        top_k,
        chunk_size=chunk_size,
    )
    teacher_top = _topk_indices_from_hidden(
        teacher_hidden,
        teacher_lm_head_weight,
        top_k,
        chunk_size=chunk_size,
    )
    return (
        student_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)
    ).any(dim=-1).float().mean(dim=-1)


def topk_accuracy_and_agreement_from_hidden(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    top_ks: tuple[int, ...] = (1, 5, 10),
    *,
    chunk_size: int = 16384,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Compute label accuracy and teacher/student consistency from one top-k sweep."""
    max_top_k = max(top_ks)
    student_top = _topk_indices_from_hidden(
        student_hidden,
        student_lm_head_weight,
        max_top_k,
        chunk_size=chunk_size,
    )
    teacher_top = _topk_indices_from_hidden(
        teacher_hidden,
        teacher_lm_head_weight,
        max_top_k,
        chunk_size=chunk_size,
    )
    agreements = {}
    accuracies = {}
    for top_k in top_ks:
        student_k = student_top[:, :top_k]
        teacher_k = teacher_top[:, :top_k]
        agreements[top_k] = (
            student_k.unsqueeze(-1) == teacher_k.unsqueeze(-2)
        ).any(dim=-1).float().mean(dim=-1)
        accuracies[top_k] = (student_k == labels.unsqueeze(-1)).any(dim=-1).float()
    return agreements, accuracies
