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

"""Fused, row-chunked lm_head statistics for DFlash-family training.

The draft's vocab-wide logits are read only through per-row statistics: distillation
cross-entropy and total variation against a teacher, label cross-entropy, logsumexp, argmax
and top-k. Each chunk of rows is projected and reduced to those statistics, and the
hidden-space gradient of every differentiable one is written in the same pass, so no
``[rows, vocab]`` tensor outlives its chunk or is recomputed in backward.
"""

import functools
from collections.abc import Iterable
from typing import NamedTuple

import torch
import torch.nn.functional as F

__all__ = ["TERMS", "HeadStats", "fused_head_stats"]

# Per-row statistics that can carry a gradient back to the hidden states.
TERMS = ("kd_ce", "tv", "hard_ce", "lse")


class HeadStats(NamedTuple):
    """Per-row statistics of the head's logits; a statistic that was not requested is ``None``."""

    kd_ce: torch.Tensor | None  # -sum(p * log q) against the teacher distribution p
    tv: torch.Tensor | None  # 0.5 * |p - q|_1: one minus the acceptance rate of q against p
    hard_ce: torch.Tensor | None  # -log q[target]
    lse: torch.Tensor | None  # logsumexp of the logits
    top1: torch.Tensor  # argmax
    top_ids: torch.Tensor | None  # top-k ids, descending
    top_logits: torch.Tensor | None  # their logits, differentiable


def _transform(raw, scale, cap):
    """``cap * tanh(scale * raw / cap)`` computed in fp32 and returned in ``raw``'s dtype."""
    if scale is None and cap is None:
        return raw
    out = raw.float()
    if scale is not None:
        out = out * scale
    if cap is not None:
        out = cap * torch.tanh(out / cap)
    return out.to(raw.dtype)


def _chunk_stats(raw, teacher, targets, teacher_is_logits, scale, cap, want, grads, top_k):
    """Statistics of one chunk of rows, and each differentiable one's gradient w.r.t. ``raw``."""
    logits = _transform(raw, scale, cap)
    z = logits.float()
    lse = torch.logsumexp(z, dim=-1, keepdim=True)
    log_q = z - lse
    q = torch.exp(log_q)
    vals, dz = {}, {}
    if teacher is not None:
        zt = (teacher if teacher_is_logits else _transform(teacher, scale, cap)).float()
        p = torch.softmax(zt, dim=-1)
        if "kd_ce" in want:
            vals["kd_ce"] = -(p * log_q).sum(dim=-1)
            if "kd_ce" in grads:
                dz["kd_ce"] = q - p
        if "tv" in want:
            vals["tv"] = 0.5 * (q - p).abs().sum(dim=-1)
            if "tv" in grads:
                sign = torch.sign(q - p)
                dz["tv"] = 0.5 * q * (sign - (sign * q).sum(dim=-1, keepdim=True))
    if "hard_ce" in want:
        vals["hard_ce"] = -log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if "hard_ce" in grads:
            label = torch.arange(z.shape[-1], device=z.device) == targets.unsqueeze(-1)
            dz["hard_ce"] = q - label.float()
    if "lse" in want:
        vals["lse"] = lse.squeeze(-1)
        if "lse" in grads:
            dz["lse"] = q

    # d logits / d raw through the transform; its dtype casts pass gradients through unchanged.
    factor = 1.0 if scale is None else scale
    if cap is not None:
        th = torch.tanh(raw.float() * (factor / cap))
        factor = factor * (1.0 - th * th)
    top_ids = top_logits = top_factor = None
    if top_k:
        top_logits, top_ids = logits.topk(top_k, dim=-1)
        if isinstance(factor, torch.Tensor):
            top_factor = factor.gather(-1, top_ids)
        elif scale is not None:
            top_factor = torch.full(top_ids.shape, scale, device=raw.device)
    jac_inputs = tuple((dz[t] * factor).to(raw.dtype) for t in grads)
    stats = tuple(vals.get(t) for t in TERMS)
    return jac_inputs, stats, logits.argmax(dim=-1), top_ids, top_logits, top_factor


@functools.cache
def _compiled_chunk_stats():
    return torch.compile(_chunk_stats)


def _cat(parts):
    return None if parts[0] is None else torch.cat(parts)


class _FusedHead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        teacher,
        targets,
        teacher_is_logits,
        scale,
        cap,
        want,
        grads,
        top_k,
        chunk,
    ):
        stats_fn = _compiled_chunk_stats() if hidden.device.type == "cuda" else _chunk_stats
        # The head runs in its own dtype and the statistics in fp32, whatever autocast says.
        with torch.autocast(hidden.device.type, enabled=False):
            h = hidden.to(weight.dtype)
            jac = {t: torch.empty_like(h) for t in grads}
            parts = []
            for s in range(0, h.shape[0], chunk):
                e = min(s + chunk, h.shape[0])
                rows = None
                if teacher is not None:
                    rows = teacher[s:e]
                    if not teacher_is_logits:
                        rows = F.linear(rows.to(weight.dtype), weight)
                jac_inputs, *outs = stats_fn(
                    F.linear(h[s:e], weight),
                    rows,
                    None if targets is None else targets[s:e],
                    teacher_is_logits,
                    scale,
                    cap,
                    want,
                    grads,
                    top_k,
                )
                for t, dz in zip(grads, jac_inputs):
                    torch.mm(dz, weight, out=jac[t][s:e])
                parts.append(outs)
        stats = [_cat([part[0][i] for part in parts]) for i in range(len(TERMS))]
        top1, top_ids, top_logits, top_factor = (
            _cat([part[i] for part in parts]) for i in range(1, 5)
        )
        ctx.save_for_backward(*(jac[t] for t in grads), top_ids, top_factor, weight)
        ctx.set_materialize_grads(False)  # statistics the loss never reads cost nothing in backward
        ctx.grads, ctx.chunk, ctx.hidden_dtype = grads, chunk, hidden.dtype
        ctx.mark_non_differentiable(
            *(v for t, v in zip(TERMS, stats) if v is not None and t not in grads),
            top1,
            *(() if top_ids is None else (top_ids,)),
        )
        return (*stats, top1, top_ids, top_logits)

    @staticmethod
    def backward(ctx, *grad_outs):
        *jacs, top_ids, top_factor, weight = ctx.saved_tensors
        grad = None
        for t, jac in zip(ctx.grads, jacs):
            g = grad_outs[TERMS.index(t)]
            if g is not None:
                part = g.float().unsqueeze(-1) * jac.float()
                grad = part if grad is None else grad + part
        g_top = grad_outs[-1]
        if g_top is not None:
            g_top = g_top.float() if top_factor is None else g_top.float() * top_factor
            # sum_k g[r, k] * weight[id[r, k]], gathering the head rows one chunk at a time.
            part = torch.cat(
                [
                    torch.bmm(
                        g_top[s : s + ctx.chunk].unsqueeze(1).to(weight.dtype),
                        weight[top_ids[s : s + ctx.chunk]],
                    ).squeeze(1)
                    for s in range(0, g_top.shape[0], ctx.chunk)
                ]
            ).float()
            grad = part if grad is None else grad + part
        return (None if grad is None else grad.to(ctx.hidden_dtype), *([None] * 10))


def fused_head_stats(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    teacher_hidden: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    want: Iterable[str] = (),
    grads: Iterable[str] = (),
    top_k: int = 0,
    logit_scale: float | None = None,
    soft_cap: float | None = None,
    chunk: int = 2048,
) -> HeadStats:
    """Per-row statistics of the head's logits ``hidden @ weight.T``, never materialized whole.

    On CUDA the per-chunk statistics run through ``torch.compile``.

    Args:
        hidden: ``[R, H]`` rows. The ``grads`` statistics and ``top_logits`` backpropagate into
            them, and nothing into ``weight``, which must be frozen.
        weight: ``[V, H]`` head weight.
        teacher_hidden: ``[R, H]`` teacher rows to project through the same head. ``kd_ce``
            and ``tv`` need this or ``teacher_logits``.
        teacher_logits: ``[R, V]`` teacher logits, used as given.
        targets: ``[R]`` label ids, needed for ``hard_ce``.
        want: Statistics to return, named as in :data:`TERMS`; ``grads`` are returned too.
        grads: Statistics to differentiate w.r.t. ``hidden``.
        top_k: Also return each row's ``top_k`` ids and their logits.
        logit_scale: Multiplier the head applies to its projection, if any.
        soft_cap: ``c`` of a ``c * tanh(x / c)`` cap the head applies after ``logit_scale``.
        chunk: Rows per projection; bounds the vocab-wide temporaries.
    """
    want, grads = set(want), set(grads)
    if unknown := (want | grads) - set(TERMS):
        raise ValueError(f"Unknown statistics {sorted(unknown)}; choose from {TERMS}.")
    if weight.requires_grad:
        raise ValueError("fused_head_stats computes no gradient for the head weight; freeze it.")
    if teacher_hidden is not None and teacher_logits is not None:
        raise ValueError("Pass teacher_hidden or teacher_logits, not both.")
    if top_k > weight.shape[0]:
        raise ValueError(f"top_k={top_k} exceeds the vocabulary size {weight.shape[0]}.")
    teacher = teacher_hidden if teacher_logits is None else teacher_logits
    want |= grads
    if not (torch.is_grad_enabled() and hidden.requires_grad):
        grads = set()
    if teacher is None and want & {"kd_ce", "tv"}:
        raise ValueError("kd_ce and tv need teacher_hidden or teacher_logits.")
    if targets is None and "hard_ce" in want:
        raise ValueError("hard_ce needs targets.")
    return HeadStats(
        *_FusedHead.apply(
            hidden,
            weight,
            teacher,
            targets,
            teacher_logits is not None,
            logit_scale,
            soft_cap,
            tuple(t for t in TERMS if t in want),
            tuple(t for t in TERMS if t in grads),
            top_k,
            chunk,
        )
    )
