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

"""CPU parity tests for the dual-side vocab-streaming CE/KD/top-k kernel.

The kernel must equal a naive full-logit reference regardless of chunk size, so
the AutoModel replace-1-block metric matches the legacy scoring math.
"""

import torch

from modelopt.torch.puzzletron.plugins.automodel.flash_dual import (
    flash_dual_ce_kd,
    topk_hit_from_hidden,
)


def _reference(student_hidden, student_w, teacher_hidden, teacher_w, labels, temperature, ignore_index):
    zs = student_hidden.float() @ student_w.float().t()  # [N, V]
    zt = teacher_hidden.float() @ teacher_w.float().t()
    ce = torch.nn.functional.cross_entropy(zs, labels, ignore_index=ignore_index, reduction="none")
    p_t = torch.softmax(zt / temperature, dim=-1)
    logp_t = torch.log_softmax(zt / temperature, dim=-1)
    logp_s = torch.log_softmax(zs / temperature, dim=-1)
    kd = (p_t * (logp_t - logp_s)).sum(dim=-1)
    return ce, kd


def test_flash_dual_matches_full_logit_reference():
    torch.manual_seed(0)
    n, d, v = 7, 16, 50
    sh = torch.randn(n, d)
    th = torch.randn(n, d)
    sw = torch.randn(v, d)
    tw = torch.randn(v, d)
    labels = torch.randint(0, v, (n,))
    labels[0] = -1  # ignore_index token contributes 0 CE

    ref_ce, ref_kd = _reference(sh, sw, th, tw, labels, temperature=1.0, ignore_index=-1)
    # chunk_size < vocab forces the multi-chunk online path.
    ce, kd = flash_dual_ce_kd(sh, sw, th, tw, labels, chunk_size=8, ignore_index=-1)

    assert torch.allclose(ce, ref_ce, atol=1e-4), (ce, ref_ce)
    assert torch.allclose(kd, ref_kd, atol=1e-4), (kd, ref_kd)
    assert ce[0].item() == 0.0  # ignore_index


def test_flash_dual_temperature_matches_reference():
    torch.manual_seed(1)
    n, d, v = 5, 8, 32
    sh, th = torch.randn(n, d), torch.randn(n, d)
    sw, tw = torch.randn(v, d), torch.randn(v, d)
    labels = torch.randint(0, v, (n,))
    ref_ce, ref_kd = _reference(sh, sw, th, tw, labels, temperature=2.0, ignore_index=-1)
    ce, kd = flash_dual_ce_kd(sh, sw, th, tw, labels, temperature=2.0, chunk_size=7)
    assert torch.allclose(ce, ref_ce, atol=1e-4)
    assert torch.allclose(kd, ref_kd, atol=1e-4)


def test_topk_hit_matches_full_topk():
    torch.manual_seed(2)
    n, d, v = 6, 12, 40
    sh = torch.randn(n, d)
    sw = torch.randn(v, d)
    labels = torch.randint(0, v, (n,))
    full = (sh.float() @ sw.float().t())
    for top_k in (1, 5, 10):
        ref = (full.topk(top_k, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1)
        got = topk_hit_from_hidden(sh, sw, labels, top_k, chunk_size=9)
        assert torch.equal(got, ref), (top_k, got, ref)
