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

"""Tests for the KD rotation objective: learn_rotations(teacher=...)."""

import copy
import sys
import traceback

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from modelopt.torch.quantization.rotation import QuantObjective, learn_rotations

VOCAB = 128
HIDDEN = 64
HEAD_DIM = 32

TINY_W4A4 = QuantObjective(
    name="tiny_w4a4", w_bits=4, w_group=16, a_bits=4, a_mode="per_token_dynamic"
)


def _tiny_llama():
    torch.manual_seed(1234)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)
    return model


def _batches(n=1, bs=2, seq=16, seed=7):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (bs, seq)) for _ in range(n)]


def test_kd_wiring_no_quant_kl_vanishes():
    """With ALL quantizers off and teacher = an exact functional copy, the rotated
    student computes the teacher's function -> KL term ~ 0 and the KD loss equals
    (1 - kd_alpha) * CE of the plain run at step 0 (lr=0 isolates step-0)."""
    batches = _batches()
    ref = learn_rotations(
        _tiny_llama(), batches, steps=1, lr=0.0, objective_cfg=None, seed=3, log_every=0
    )
    model = _tiny_llama()
    teacher = copy.deepcopy(model)
    kd = learn_rotations(
        model,
        batches,
        steps=1,
        lr=0.0,
        objective_cfg=None,
        seed=3,
        log_every=0,
        teacher=teacher,
        kd_alpha=0.5,
        kd_temp=2.0,
    )
    ce_ref = ref.history[0]["loss"]
    got = kd.history[0]["loss"]
    want = 0.5 * ce_ref  # (1-a)*CE + a*T^2*KL, KL ~ 0
    assert abs(got - want) < 5e-3 * max(1.0, abs(want)), (
        f"KD step-0 loss {got} != (1-a)*CE = {want} (CE {ce_ref}) — wiring or a "
        "non-vanishing KL where the student equals the teacher"
    )
    assert kd.meta["kd"] == {"alpha": 0.5, "temp": 2.0}
    assert ref.meta["kd"] is None


def test_kd_objective_trains_and_stays_orthogonal():
    """The KD path is live and Cayley-safe: rotations MOVE, stay orthogonal, and the KD
    term is genuinely nonzero against a perturbed teacher.

    Deliberately does NOT assert "final loss < first loss": at this toy scale the
    trajectory is noise (plain CE on the same recipe also ends higher than it starts), so
    such an assertion passes or fails on seed luck rather than on training behavior.
    """
    torch.manual_seed(11)
    batch = _batches(n=1)
    model = _tiny_llama()
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():  # a teacher that genuinely differs -> KL > 0
        p.data.add_(0.05 * torch.randn_like(p.data))
    teacher.eval().requires_grad_(False)

    ref = learn_rotations(
        _tiny_llama(), batch, steps=0, lr=0.5, objective_cfg=TINY_W4A4, seed=5, log_every=0
    )
    rs = learn_rotations(
        model,
        batch,
        steps=10,
        lr=0.5,
        objective_cfg=TINY_W4A4,
        seed=5,
        log_every=0,
        teacher=teacher,
    )
    # (a) training actually moved the rotations away from the shared seeded init
    moved = (rs.R1 - ref.R1).abs().max().item()
    assert moved > 1e-4, f"rotations did not move under the KD objective (max delta {moved:.2e})"
    # (b) every iterate is still on the manifold after the Cayley steps + retraction
    assert max(rs.ortho_audit().values()) < 1e-4
    # (c) the KD term is live: an alpha=1 run against this teacher has a positive loss
    kd_only = learn_rotations(
        _tiny_llama(),
        batch,
        steps=1,
        lr=0.0,
        objective_cfg=TINY_W4A4,
        seed=5,
        log_every=0,
        teacher=teacher,
        kd_alpha=1.0,
    )
    assert kd_only.history[0]["loss"] > 0.0, "KD term is identically zero (KD path is dead)"


def test_teacher_untouched():
    model = _tiny_llama()
    teacher = copy.deepcopy(model)
    before = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    learn_rotations(
        model,
        _batches(),
        steps=3,
        lr=0.5,
        objective_cfg=TINY_W4A4,
        seed=5,
        log_every=0,
        teacher=teacher,
    )
    for n, p in teacher.named_parameters():
        assert torch.equal(p.detach(), before[n]), f"teacher param {n} changed"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
