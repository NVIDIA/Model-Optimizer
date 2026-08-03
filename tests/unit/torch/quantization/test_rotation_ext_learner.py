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

"""Learner-semantics tests (T20.4) for modelopt.torch.quantization.rotation.learn.

Covers:
  1. steps=0 == fold_rotations seed draws, bitwise, seeds {0, 3, 7}, both archs.
  2. Gradient exclusivity at step 0 for every objective preset (W4A4_G128,
     INT8_DEFAULT, int8 per-token-dynamic): grads reach all R params, none reach
     model params.
  3. Loss decreases on a repeated batch (25 steps) for each of the 3 objectives.
  4. a_static_scope semantics: "run" scale = monotone running max, "batch" scale
     tracks per-batch amax (can decrease); telemetry monotone; both stay orthogonal.
  5. Warm start from a prior RotationSet: step-0 loss continues the donor's final
     loss; the ortho gate rejects a corrupted warm start.
  6. Gradient checkpointing: recompute-during-backward still sees the effective
     weights (_reparametrize_module spans backward), grads reach R, loss finite.
  7. Objective coverage: activation hooks on exactly 7*n_layers linears;
     lm_head/embeddings never hooked and never fake-quantized.

Plain test_* functions with asserts: collectable by pytest, and also runnable without
it via ``python test_rotation_ext_learner.py`` (the __main__ driver runs every test
function and exits nonzero on any failure). CPU-only, tiny models, seconds per test.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # HARD CONSTRAINT: CPU-only, never touch the GPU

import itertools
import sys
import traceback
import types

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import stateless
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modelopt.torch.quantization.rotation import (
    INT8_DEFAULT_OBJECTIVE,
    W4A4_G128_OBJECTIVE,
    QuantObjective,
    fold_rotations,
    learn_rotations,
)
from modelopt.torch.quantization.rotation.learn import (
    _ATTN_PROJS,
    _MLP_PROJS,
    _ActQuantHooks,
    _assemble_effective_weights,
    _fq_act,
)

VOCAB = 128
# Standard tiny fixture (same as test_rotation_{fold,learn}.py): HEAD_DIM deliberately
# decoupled from HIDDEN // num_attention_heads (32 vs 64//4 = 16), like Qwen3-0.6B.
HIDDEN = 64
HEAD_DIM = 32
N_LAYERS = 2

# Grouped fixture for W4A4_G128_OBJECTIVE: w_group=128 must divide every quantized
# in_features. hidden=128 (q/k/v/gate/up in), heads=4 * head_dim=64 -> o_proj in 256,
# intermediate=2*hidden=256 (down_proj in) — all divisible by 128. head_dim stays
# decoupled (64 != 128//4 = 32).
G_HIDDEN = 128
G_HEAD_DIM = 64

#: The exact three presets of the task: the module's two shipped presets plus the
#: int8 per-token-dynamic recipe.
INT8_PTDYN = QuantObjective(
    name="ptdyn", w_bits=8, w_group=None, a_bits=8, a_mode="per_token_dynamic"
)
PRESET_OBJECTIVES = (W4A4_G128_OBJECTIVE, INT8_DEFAULT_OBJECTIVE, INT8_PTDYN)

# Tiny-model W4A4 for the standard fixture (g=16 divides 64/128/128), as in
# test_rotation_learn.py — used where the objective flavor is not the thing under test.
TINY_W4A4 = QuantObjective(
    name="tiny_w4a4", w_bits=4, w_group=16, a_bits=4, a_mode="per_token_dynamic"
)

# fp32 Cayley iterates drift off the manifold at ~1e-7/step (measured); a handful of
# steps stays well under 1e-5. Post-retraction audits sit at fp64-SVD level (<1e-10).
ORTHO_TOL_FP32_STEPS = 1e-5


def _randomize_rmsnorm_gains(model):
    """Non-one RMSNorm gains so norm-fusion effects are not vacuous (fresh HF models
    initialize all norm weights to ones)."""
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)


def _build(cfg_cls, model_cls, hidden, head_dim, tie=False):
    torch.manual_seed(1234)
    cfg = cfg_cls(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attn_implementation="eager",
    )
    model = model_cls(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _tiny_llama(tie=False):
    return _build(LlamaConfig, LlamaForCausalLM, HIDDEN, HEAD_DIM, tie)


def _tiny_qwen3(tie=False):
    return _build(Qwen3Config, Qwen3ForCausalLM, HIDDEN, HEAD_DIM, tie)


def _grouped_llama(tie=False):
    return _build(LlamaConfig, LlamaForCausalLM, G_HIDDEN, G_HEAD_DIM, tie)


def _grouped_qwen3(tie=False):
    return _build(Qwen3Config, Qwen3ForCausalLM, G_HIDDEN, G_HEAD_DIM, tie)


def _calib_batches(n_batches=2, bs=2, seq=16, seed=7):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (bs, seq)) for _ in range(n_batches)]


# --------------------------------------------------------------------------------------
# 1. steps=0 == fold seed draws, bitwise, seeds {0, 3, 7}, both archs
# --------------------------------------------------------------------------------------


def test_steps0_matches_fold_seed_draws_bitwise():
    """learn_rotations(steps=0, seed=s) returns exactly (torch.equal, bitwise) the
    matrices fold_rotations(mode, seed=s) draws — for seeds {0, 3, 7} on both archs
    (hadamard everywhere; "random" mode additionally checked at seed 0). One RNG
    contract: R1 first, then R2 by ascending layer, from the seeded global CPU RNG."""
    for build in (_tiny_llama, _tiny_qwen3):
        for seed in (0, 3, 7):
            modes = ("hadamard", "random") if seed == 0 else ("hadamard",)
            for mode in modes:
                init = learn_rotations(
                    build(),
                    [],
                    steps=0,
                    mode=mode,
                    objective_cfg=None,
                    seed=seed,
                    log_every=0,
                )
                folded = fold_rotations(build(), mode=mode, seed=seed, use_r2=True)
                assert set(init.rotations) == set(folded), (
                    f"{build.__name__} seed={seed} mode={mode}: key sets differ"
                )
                for k in folded:
                    assert torch.equal(init.rotations[k], folded[k]), (
                        f"{build.__name__} seed={seed} mode={mode}: {k} not bitwise equal"
                    )


# --------------------------------------------------------------------------------------
# 2. Gradient exclusivity at step 0 for every objective preset
# --------------------------------------------------------------------------------------


def _one_training_backward(model, init, objective):
    """Mirror one step-0 forward/backward of learn_rotations' loop using the module's
    own building blocks (_assemble_effective_weights + _ActQuantHooks +
    stateless._reparametrize_module) with the R matrices as fresh leaves, so gradients
    are observable from the outside. ``model`` must already be prepared (untied, fused,
    frozen) — learn_rotations(steps=0) does exactly that."""
    n_layers = len(model.model.layers)
    head_dim = model.config.head_dim
    R1 = nn.Parameter(init.R1.to(torch.float32))
    R2s = [
        nn.Parameter(init.rotations[f"model.layers.{i}.self_attn.R2"].to(torch.float32))
        for i in range(n_layers)
    ]
    sd = dict(model.named_parameters())
    base = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].data,
        "lm_head.weight": sd["lm_head.weight"].data,
    }
    for i in range(n_layers):
        for proj in _ATTN_PROJS + _MLP_PROJS:
            name = f"model.layers.{i}.{proj}.weight"
            base[name] = sd[name].data

    hooks = None
    if objective is not None and objective.a_bits is not None:
        hooks = _ActQuantHooks(objective)
        n_hooked = hooks.attach(model)
        assert n_hooked == 7 * n_layers, f"hooked {n_hooked}, expected {7 * n_layers}"
    torch.manual_seed(42)
    ids = torch.randint(0, VOCAB, (2, 16))
    try:
        eff = _assemble_effective_weights(
            base,
            R1,
            R2s,
            n_layers,
            head_dim,
            objective,
            model.model.embed_tokens.weight.dtype,
        )
        with stateless._reparametrize_module(model, eff):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
    finally:
        if hooks is not None:
            hooks.remove()
    return loss, R1, R2s


def test_gradient_exclusivity_every_objective():
    """For each of the 3 objective presets on both archs: one forward/backward at the
    seed init gives (i) a finite loss, (ii) a non-None, nonzero, finite gradient on R1
    and on EVERY per-layer R2, and (iii) no gradient on any model parameter."""
    for build in (_grouped_llama, _grouped_qwen3):
        for obj in PRESET_OBJECTIVES:
            tag = f"{build.__name__}/{obj.name}"
            model = build()
            init = learn_rotations(model, [], steps=0, objective_cfg=None, seed=0, log_every=0)
            loss, R1, R2s = _one_training_backward(model, init, obj)
            assert torch.isfinite(loss), f"{tag}: loss not finite: {loss}"
            assert R1.grad is not None, f"{tag}: no grad reached R1"
            assert torch.isfinite(R1.grad).all(), f"{tag}: R1 grad not finite"
            assert R1.grad.abs().max().item() > 0, f"{tag}: R1 grad identically zero"
            for i, r2 in enumerate(R2s):
                assert r2.grad is not None, f"{tag}: no grad reached R2[{i}]"
                assert torch.isfinite(r2.grad).all(), f"{tag}: R2[{i}] grad not finite"
                assert r2.grad.abs().max().item() > 0, f"{tag}: R2[{i}] grad zero"
            for name, p in model.named_parameters():
                assert p.grad is None, f"{tag}: model param {name} received a gradient"


# --------------------------------------------------------------------------------------
# 3. Loss decreases on a repeated batch for each of the 3 objectives
# --------------------------------------------------------------------------------------


def test_loss_decreases_each_objective():
    """25 Cayley steps overfitting a single repeated batch must reduce the loss for
    every preset: min of the last 5 recorded losses < the step-0 loss. (The loop is
    deterministic on a repeated batch, so any decrease is real descent, not noise.)"""
    for obj in PRESET_OBJECTIVES:
        model = _grouped_qwen3()
        batch = _calib_batches(n_batches=1, bs=2, seq=32, seed=11)
        rs = learn_rotations(model, batch, steps=25, lr=1.0, objective_cfg=obj, seed=0, log_every=0)
        losses = [r["loss"] for r in rs.history]
        assert len(losses) == 25
        assert all(torch.isfinite(torch.tensor(v)) for v in losses), (
            f"{obj.name}: non-finite loss in {losses}"
        )
        assert min(losses[-5:]) < losses[0], (
            f"{obj.name}: no decrease — first {losses[0]:.6f}, last5 {losses[-5:]}"
        )


# --------------------------------------------------------------------------------------
# 4. a_static_scope semantics ("run" vs "batch")
# --------------------------------------------------------------------------------------


class _HookProbe(nn.Module):
    """Minimal module whose single Linear is named ``model.layers.0.self_attn.q_proj``
    (the attach filter needs ``.layers.`` in the name + a target suffix), plus one
    decoy Linear (``model.head``) that must NOT be hooked."""

    def __init__(self, in_f=8):
        super().__init__()

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(in_f, 4, bias=False)

            def forward(self, x):
                return self.q_proj(x)

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _Attn()

            def forward(self, x):
                return self.self_attn(x)

        class _Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Layer()])
                # Decoy: an nn.Linear named "model.head" — no ".layers." in its name,
                # so the attach filter must skip it.
                self.head = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.head(self.layers[0](x))

        self.model = _Core()

    def forward(self, x):
        return self.model(x)


def test_a_static_scope_run_vs_batch_semantics():
    """Step-resolved semantics of the per-tensor-static activation scale:

    - scope="run": the effective scale is the monotone running max of batch amaxes —
      a later small batch is quantized on the stale coarse grid (3/127 -> 0).
    - scope="batch": the effective scale tracks each batch's own amax and DECREASES
      when a smaller batch arrives — the small batch reconstructs exactly on its grid.
    - static_amax telemetry is the monotone non-decreasing observed max in BOTH scopes
      (the module records the running max as telemetry either way, by design).

    Verified bitwise against the module's own _fq_act with independently computed
    expected scales, via a capture hook registered after the quant hook."""
    key = "model.layers.0.self_attn.q_proj"
    qpos = 127.0
    # amax sequence 8 -> 1 -> 5 (decrease then partial recovery); b2/b3 values sit
    # exactly on their own amax/127 grids so batch-scope fq is an exact identity.
    b1 = torch.tensor([[8.0, -8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0]])
    b2 = torch.tensor([[3.0, -5.0, 1.0, 127.0, -127.0, 2.0, 64.0, 0.0]]) / 127.0
    b3 = torch.tensor([[5.0, -5.0, 2.5, 1.25, 0.5, 0.25, 3.0, 0.0]])
    batches = [b1, b2, b3]
    amaxes = [8.0, 1.0, 5.0]

    for scope in ("run", "batch"):
        probe = _HookProbe()
        cfg = QuantObjective(
            name=f"probe_{scope}",
            w_bits=None,
            w_group=None,
            a_bits=8,
            a_mode="per_tensor_static",
            a_static_scope=scope,
        )
        hooks = _ActQuantHooks(cfg)
        n = hooks.attach(probe)
        assert n == 1, f"scope={scope}: attached {n} hooks, expected 1 (decoy hooked?)"

        captured = []
        lin = probe.model.layers[0].self_attn.q_proj
        cap_handle = lin.register_forward_pre_hook(
            lambda m, inp: captured.append(inp[0].detach().clone())
        )
        telemetry = []
        with torch.no_grad():
            for x in batches:
                probe(x)
                telemetry.append(float(hooks.static_amax[key]))
        cap_handle.remove()
        hooks.remove()

        # Telemetry: monotone non-decreasing running max in BOTH scopes.
        assert telemetry == [8.0, 8.0, 8.0], f"scope={scope}: telemetry {telemetry}"
        assert all(b >= a for a, b in itertools.pairwise(telemetry))

        # Effective scale per batch, replicated with the module's own op sequence.
        prev = None
        for i, (x, cap) in enumerate(zip(batches, captured)):
            batch_amax = x.detach().abs().amax()
            assert batch_amax.item() == amaxes[i], f"fixture broke: batch {i} amax"
            run_max = batch_amax if prev is None else torch.maximum(prev, batch_amax)
            prev = run_max
            amax = batch_amax if scope == "batch" else run_max
            s = (amax / qpos).clamp_min(1e-12)
            expected = _fq_act(x, s, 8)
            assert torch.equal(cap, expected), (
                f"scope={scope} batch {i}: fq output does not match the "
                f"{'per-batch' if scope == 'batch' else 'running-max'} scale "
                f"(amax used should be {amax.item():.6f})"
            )
        if scope == "run":
            # Small batch on the stale coarse grid: 3/127 quantizes to 0 with s=8/127.
            assert captured[1][0, 0].item() == 0.0, "run scope: expected 3/127 -> 0"
            assert not torch.allclose(captured[1], b2), (
                "run scope: small batch must be distorted by the stale scale"
            )
        else:
            # Per-batch scale decreased 8/127 -> 1/127: exact-grid identity on b2.
            assert torch.allclose(captured[1], b2, rtol=0, atol=1e-9), (
                "batch scope: small batch must reconstruct exactly on its own grid"
            )

    # Both scopes stay orthogonal end-to-end through learn_rotations.
    for scope in ("batch", "run"):
        obj = QuantObjective(
            name=f"int8_{scope}",
            w_bits=8,
            w_group=None,
            a_bits=8,
            a_mode="per_tensor_static",
            a_static_scope=scope,
        )
        rs = learn_rotations(
            _tiny_llama(),
            _calib_batches(),
            steps=4,
            lr=1.0,
            objective_cfg=obj,
            seed=0,
            log_every=0,
        )
        audit = rs.ortho_audit()
        assert max(audit.values()) < 1e-10, f"scope={scope}: post-retraction {audit}"
        assert all(r["r1_ortho"] < ORTHO_TOL_FP32_STEPS for r in rs.history), (
            f"scope={scope}: raw drift {[r['r1_ortho'] for r in rs.history]}"
        )
        amax = rs.meta.get("static_act_amax", {})
        assert len(amax) == 7 * N_LAYERS and all(v > 0 for v in amax.values())


# --------------------------------------------------------------------------------------
# 5. Warm start
# --------------------------------------------------------------------------------------


def test_warm_start_continues_donor_loss_and_gate_rejects_corruption():
    """init_rotations from a donor RotationSet: the warm run's step-0 loss (same
    repeated batch, identically built model) lands in the ballpark of the donor's
    final loss — training progress transfers through save/warm-start — and is closer
    to the donor's end than to its start. A deliberately corrupted warm start (R1
    scaled by 1.01, ortho residual ~2e-2 >> 1e-4) is rejected by the ortho gate."""
    batch = _calib_batches(n_batches=1, bs=2, seq=32, seed=11)
    donor = learn_rotations(
        _tiny_qwen3(),
        batch,
        steps=12,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    donor_first = donor.history[0]["loss"]
    donor_last = donor.history[-1]["loss"]
    assert donor_last < donor_first, (
        f"donor did not descend ({donor_first:.6f} -> {donor_last:.6f}); "
        "warm-start continuity check would be vacuous"
    )

    warm = learn_rotations(
        _tiny_qwen3(),
        batch,
        steps=1,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        init_rotations=donor.rotations,
        log_every=0,
    )
    assert warm.meta["warm_start"] is True
    w0 = warm.history[0]["loss"]
    # Ballpark: the only differences vs. donor_last are the donor's final near-zero-lr
    # Cayley step (cosine lr at step 11/12 is 1.7% of peak), the final polar retraction
    # (~1e-6 entry delta), and fp64->fp32 casts. Budget 20% of the donor's total
    # improvement, floored at 0.02 absolute.
    tol = max(0.02, 0.2 * (donor_first - donor_last))
    assert abs(w0 - donor_last) < tol, (
        f"warm step-0 loss {w0:.6f} not in the donor-final ballpark "
        f"{donor_last:.6f} (donor first {donor_first:.6f}, tol {tol:.6f})"
    )
    assert abs(w0 - donor_last) < abs(w0 - donor_first), (
        f"warm step-0 loss {w0:.6f} closer to donor start {donor_first:.6f} "
        f"than to donor end {donor_last:.6f}"
    )

    # Ortho gate: corrupted warm start must be refused.
    bad = dict(donor.rotations)
    bad["R1"] = bad["R1"] * 1.01
    with pytest.raises(AssertionError, match="not orthogonal"):
        learn_rotations(
            _tiny_qwen3(),
            batch,
            steps=1,
            objective_cfg=TINY_W4A4,
            seed=0,
            init_rotations=bad,
            log_every=0,
        )


# --------------------------------------------------------------------------------------
# 6. Gradient checkpointing — reparametrize must span backward
# --------------------------------------------------------------------------------------


def test_gradient_checkpointing_grads_still_reach_rotations():
    """gradient_checkpointing_enable() + 3 learn steps: the checkpointed recompute
    happens during backward and must still see the effective weights (loss.backward()
    runs inside the _reparametrize_module context — the design property). Grads reach
    R (the trainer's internal step-0 exclusivity assert would raise otherwise, and the
    returned R1 visibly moved from its init), every recorded loss is finite, and the
    checkpoint function demonstrably ran (call counter == steps * n_layers).

    transformers' GradientCheckpointingLayer only checkpoints when ``self.training``
    is True, but learn_rotations calls model.eval(); the test force-keeps train mode
    via a no-op eval override so the checkpointed path actually executes (the tiny
    models have no active dropout, so train-mode forward is deterministic)."""
    steps = 3
    model = _tiny_qwen3()
    model.gradient_checkpointing_enable()
    model.train()
    model.eval = types.MethodType(lambda self: self, model)  # keep checkpointing active

    calls = {"n": 0}
    wrapped_any = False
    for m in model.modules():
        f = getattr(m, "_gradient_checkpointing_func", None)
        if f is not None and getattr(m, "gradient_checkpointing", False):

            def counting(*a, __f=f, **k):
                calls["n"] += 1
                return __f(*a, **k)

            m._gradient_checkpointing_func = counting
            wrapped_any = True
    assert wrapped_any, "gradient_checkpointing_enable() installed no checkpoint funcs"

    rs = learn_rotations(
        model,
        _calib_batches(),
        steps=steps,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    assert calls["n"] == steps * N_LAYERS, (
        f"checkpointing did not run as expected: {calls['n']} checkpoint calls, "
        f"expected {steps * N_LAYERS} (steps * n_layers)"
    )
    assert len(rs.history) == steps
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rs.history), (
        f"non-finite loss under checkpointing: {[r['loss'] for r in rs.history]}"
    )
    # Grads reached R: the returned R1 moved away from the seed-0 init draw.
    init = learn_rotations(_tiny_qwen3(), [], steps=0, objective_cfg=None, seed=0, log_every=0)
    dmax = (rs.R1 - init.R1).abs().max().item()
    assert dmax > 1e-5, f"R1 did not move under checkpointing (max delta {dmax:.3e})"
    assert max(rs.ortho_audit().values()) < 1e-10


# --------------------------------------------------------------------------------------
# 7. Objective coverage — hook targets and never-quantized embeddings/lm_head
# --------------------------------------------------------------------------------------


def test_hook_coverage_and_embeddings_never_quantized():
    """Activation hooks attach to exactly the 7*n_layers decoder-layer linears on both
    archs — never lm_head (an nn.Linear!) or embed_tokens — and are fully removed both
    by hooks.remove() and after a learn_rotations run. In the effective-weight
    assembly, embed_tokens/lm_head are rotated but NEVER weight-fake-quantized, while
    decoder linears are."""
    expected = {f"model.layers.{i}.{p}" for i in range(N_LAYERS) for p in _ATTN_PROJS + _MLP_PROJS}
    for build in (_tiny_llama, _tiny_qwen3):
        model = build()
        hooks = _ActQuantHooks(INT8_PTDYN)
        n = hooks.attach(model)
        assert n == 7 * N_LAYERS, f"{build.__name__}: {n} hooks, expected {7 * N_LAYERS}"
        hooked = {name for name, m in model.named_modules() if len(m._forward_pre_hooks) > 0}
        assert hooked == expected, (
            f"{build.__name__}: hooked set mismatch: extra {hooked - expected}, "
            f"missing {expected - hooked}"
        )
        assert "lm_head" not in hooked and "model.embed_tokens" not in hooked
        hooks.remove()
        assert all(len(m._forward_pre_hooks) == 0 for _, m in model.named_modules()), (
            f"{build.__name__}: hooks leaked after remove()"
        )

    # learn_rotations leaves no hooks behind (removed in its finally block).
    model = _tiny_llama()
    learn_rotations(
        model,
        _calib_batches(),
        steps=1,
        lr=1.0,
        objective_cfg=INT8_PTDYN,
        seed=0,
        log_every=0,
    )
    assert all(len(m._forward_pre_hooks) == 0 for _, m in model.named_modules()), (
        "learn_rotations left activation hooks attached"
    )

    # Assembly: embeddings/lm_head rotated but never fake-quantized; decoder linears are.
    model = _tiny_qwen3()
    init = learn_rotations(model, [], steps=0, objective_cfg=None, seed=0, log_every=0)
    n_layers = len(model.model.layers)
    R1 = init.R1.to(torch.float32)
    R2s = [
        init.rotations[f"model.layers.{i}.self_attn.R2"].to(torch.float32) for i in range(n_layers)
    ]
    sd = dict(model.named_parameters())
    base = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].data,
        "lm_head.weight": sd["lm_head.weight"].data,
    }
    for i in range(n_layers):
        for proj in _ATTN_PROJS + _MLP_PROJS:
            name = f"model.layers.{i}.{proj}.weight"
            base[name] = sd[name].data
    # Brutal 2-bit weight quant makes any accidental embed/lm_head quantization obvious.
    obj = QuantObjective(name="w2_probe", w_bits=2, w_group=None, a_bits=None)
    eff = _assemble_effective_weights(
        base, R1, R2s, n_layers, model.config.head_dim, obj, torch.float32
    )
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        plain_rot = (base[name].to(torch.float32) @ R1).to(torch.float32)
        assert torch.equal(eff[name], plain_rot), (
            f"{name}: effective weight != plain rotation — it was fake-quantized"
        )
    q_name = "model.layers.0.self_attn.q_proj.weight"
    plain_q = (base[q_name].to(torch.float32) @ R1).to(torch.float32)
    assert not torch.equal(eff[q_name], plain_q), (
        "q_proj effective weight was NOT fake-quantized — objective not applied"
    )


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(
        f"\n{len(tests) - len(failed)}/{len(tests)} tests passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    sys.exit(1 if failed else 0)
