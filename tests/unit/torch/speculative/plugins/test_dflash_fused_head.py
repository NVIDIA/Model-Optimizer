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

"""CPU tests for the fused lm_head statistics behind ``dflash_fused_loss``."""

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.speculative.plugins.dflash_fused_head import TERMS, fused_head_stats

ROWS, HIDDEN, VOCAB, TOP_K = 37, 16, 50, 4
CHUNK = 8  # several chunks and a partial last one


def _transform(raw, scale, cap):
    if scale is not None:
        raw = raw * scale
    return raw if cap is None else cap * torch.tanh(raw / cap)


@pytest.mark.parametrize("teacher_kind", ["hidden", "logits"])
@pytest.mark.parametrize(("scale", "cap"), [(None, None), (0.7, None), (0.7, 2.0)])
def test_matches_autograd_on_the_materialized_logits(teacher_kind, scale, cap):
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
        "tv": 0.5 * (log_q.exp() - log_p.exp()).abs().sum(-1),
        "hard_ce": F.cross_entropy(logits, targets, reduction="none"),
        "lse": logits.logsumexp(-1),
    }
    top_logits, top_ids = logits.topk(TOP_K, dim=-1)
    ref_loss = sum((coefs[t] * ref[t]).sum() for t in TERMS) + (top_coefs * top_logits).sum()
    ref_loss.backward()

    fused_hidden = hidden.clone().requires_grad_(True)
    kwargs = dict(
        targets=targets,
        grads=TERMS,
        top_k=TOP_K,
        logit_scale=scale,
        soft_cap=cap,
        chunk=CHUNK,
        **teacher,
    )
    stats = fused_head_stats(fused_hidden, weight, **kwargs)
    loss = sum((coefs[t] * getattr(stats, t)).sum() for t in TERMS)
    (loss + (top_coefs * stats.top_logits).sum()).backward()
    with torch.no_grad():  # values without the gradient work, e.g. an eval pass in train mode
        no_grad_stats = fused_head_stats(fused_hidden, weight, **kwargs)

    for term in TERMS:
        torch.testing.assert_close(getattr(stats, term), ref[term].detach(), msg=term)
        torch.testing.assert_close(getattr(no_grad_stats, term), ref[term].detach(), msg=term)
    assert torch.equal(stats.top1, logits.argmax(-1))
    assert torch.equal(stats.top_ids, top_ids)
    torch.testing.assert_close(stats.top_logits, top_logits.detach())
    torch.testing.assert_close(fused_hidden.grad, ref_hidden.grad)
