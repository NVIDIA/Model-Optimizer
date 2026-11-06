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

"""CPU tests for both backends of the lm_head statistics behind the DFlash-family loss."""

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.speculative.plugins.dflash_head_stats import (
    TERMS,
    HeadRequest,
    fused_head_stats,
    materialized_head_stats,
)

ROWS, HIDDEN, VOCAB, TOP_K = 37, 16, 50, 4
CHUNK = 8  # several chunks and a partial last one
BACKENDS = ["fused", "materialized"]


def _transform(raw, scale, cap):
    if scale is not None:
        raw = raw * scale
    return raw if cap is None else cap * torch.tanh(raw / cap)


def _stats(backend, hidden, weight, request, scale=None, cap=None, **inputs):
    """The statistics of the head ``_transform(hidden @ weight.T)``, from ``backend``."""
    if backend == "fused":
        return fused_head_stats(
            hidden, weight, request, logit_scale=scale, soft_cap=cap, chunk=CHUNK, **inputs
        )
    if (teacher_hidden := inputs.pop("teacher_hidden", None)) is not None:
        inputs["teacher_logits"] = _transform(teacher_hidden @ weight.T, scale, cap)
    return materialized_head_stats(_transform(hidden @ weight.T, scale, cap), request, **inputs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("teacher_kind", ["hidden", "logits"])
@pytest.mark.parametrize(("scale", "cap"), [(None, None), (0.7, None), (0.7, 2.0)])
def test_matches_autograd_on_the_logits(backend, teacher_kind, scale, cap):
    """Every statistic, its gradient under arbitrary per-row upstream weights, and the top-k."""
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, HIDDEN)
    hidden = torch.randn(ROWS, HIDDEN)
    targets = torch.randint(0, VOCAB, (ROWS,))
    teacher_hidden = torch.randn(ROWS, HIDDEN)
    teacher_logits = 3 * torch.randn(ROWS, VOCAB)  # handed over as is, never transformed
    coefs = {term: torch.randn(ROWS) for term in TERMS}
    top_coefs = torch.randn(ROWS, TOP_K)

    ref_hidden = hidden.clone().requires_grad_(True)
    logits = _transform(ref_hidden @ weight.T, scale, cap)
    if teacher_kind == "hidden":
        teacher = {"teacher_hidden": teacher_hidden}
        log_p = _transform(teacher_hidden @ weight.T, scale, cap).log_softmax(-1)
    else:
        teacher = {"teacher_logits": teacher_logits}
        log_p = teacher_logits.log_softmax(-1)
    log_q = logits.log_softmax(-1)
    ref = {
        "kd_ce": -(log_p.exp() * log_q).sum(-1),
        "hard_ce": F.cross_entropy(logits, targets, reduction="none"),
        "lse": logits.logsumexp(-1),
    }
    top_logits, top_ids = logits.topk(TOP_K, dim=-1)
    ref_loss = sum((coefs[t] * ref[t]).sum() for t in TERMS) + (top_coefs * top_logits).sum()
    ref_loss.backward()

    request = HeadRequest(grads=frozenset(TERMS), top_k=TOP_K)
    test_hidden = hidden.clone().requires_grad_(True)
    stats = _stats(backend, test_hidden, weight, request, scale, cap, targets=targets, **teacher)
    loss = sum((coefs[t] * getattr(stats, t)).sum() for t in TERMS)
    (loss + (top_coefs * stats.top_logits).sum()).backward()
    with torch.no_grad():  # values without the gradient work, e.g. an eval pass in train mode
        no_grad_stats = _stats(
            backend, test_hidden, weight, request, scale, cap, targets=targets, **teacher
        )

    for term in TERMS:
        torch.testing.assert_close(getattr(stats, term), ref[term].detach(), msg=term)
        torch.testing.assert_close(getattr(no_grad_stats, term), ref[term].detach(), msg=term)
    assert torch.equal(stats.top1, logits.argmax(-1))
    assert torch.equal(stats.top_ids, top_ids)
    torch.testing.assert_close(stats.top_logits, top_logits.detach())
    torch.testing.assert_close(test_hidden.grad, ref_hidden.grad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_values_carry_no_gradient_from_either_backend(backend):
    """Only ``grads`` statistics are differentiable, so the backends cannot train differently."""
    torch.manual_seed(0)
    hidden = torch.randn(ROWS, HIDDEN, requires_grad=True)
    targets = torch.randint(0, VOCAB, (ROWS,))
    request = HeadRequest(grads=frozenset({"lse"}), values=frozenset({"hard_ce"}))
    stats = _stats(backend, hidden, torch.randn(VOCAB, HIDDEN), request, targets=targets)
    assert stats.lse.requires_grad
    assert not stats.hard_ce.requires_grad
    assert stats.kd_ce is None
