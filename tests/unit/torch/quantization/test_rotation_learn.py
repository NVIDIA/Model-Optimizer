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

"""Tests for modelopt.torch.quantization.rotation.learn (Cayley-SGD learned R1/R2) and the
external-matrix path of fold_rotations.

Plain test_* functions with asserts: collectable by pytest, and also runnable without it
via ``python test_rotation_learn.py`` (the __main__ driver runs every test function and
exits nonzero on any failure). CPU-only, tiny models, seconds per test.
"""

import os
import sys
import tempfile
import traceback

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modelopt.torch.quantization.rotation import (
    INT8_DEFAULT_OBJECTIVE,
    SGDG,
    QuantObjective,
    RotationSet,
    fold_rotations,
    learn_rotations,
)

VOCAB = 128
HIDDEN = 64
# Same deliberate decoupling as test_rotation_fold.py: HEAD_DIM != HIDDEN //
# num_attention_heads (32 vs 16), like Qwen3-0.6B — a coincident config would let a
# head_dim-resolution regression pass silently.
HEAD_DIM = 32
N_LAYERS = 2

# Tiny-model objective: per-group weights must divide every in_features
# ({64 (hidden), 128 (num_q_heads*head_dim, o_proj), 128 (2*hidden, down_proj)}), so g=16.
TINY_W4A4 = QuantObjective(
    name="tiny_w4a4", w_bits=4, w_group=16, a_bits=4, a_mode="per_token_dynamic"
)

# Orthonormality gate for fp32 Cayley iterates after a handful of steps: each step
# perturbs |R^T R - I| at the fp32-rounding scale (~1e-7 per step, measured); 1e-5 keeps
# margin while still catching any real manifold-departure bug (plain SGD drifts to O(lr)).
ORTHO_TOL_FP32_STEPS = 1e-5


def _randomize_rmsnorm_gains(model):
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)


def _tiny_llama(tie=False):
    torch.manual_seed(1234)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _tiny_qwen3(tie=False):
    torch.manual_seed(1234)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _calib_batches(n_batches=2, bs=2, seq=16, seed=7):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (bs, seq)) for _ in range(n_batches)]


def _logits(model):
    torch.manual_seed(99)
    ids = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        return model(ids).logits


def _ortho_err(R):
    Rd = R.detach().to(torch.float64)
    eye = torch.eye(Rd.shape[0], dtype=torch.float64)
    return (Rd.t() @ Rd - eye).abs().max().item()


# --------------------------------------------------------------------------------------
# 1. The Cayley step itself preserves orthonormality (optimizer in isolation)
# --------------------------------------------------------------------------------------


def test_sgdg_cayley_preserves_orthonormality():
    """10 SGDG steps on a square Stiefel parameter with adversarially LARGE random
    gradients (unit-scale randn — far harsher than real CE gradients): every iterate stays
    orthonormal at trained-rotation tolerance, the parameter actually moves, and — the
    contrast that makes the gate meaningful — a single plain-SGD step with the same
    lr/gradient leaves the manifold by >3 orders of magnitude more. Per-step Cayley drift
    is fp32-rounding + 5-iteration fixed-point truncation (~1e-6/step here; a full-scale
    reference run measures 4.6e-5 after 150 steps on a 2048-dim R1)."""
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))
    P = torch.nn.Parameter(q.to(torch.float32))
    P0 = P.detach().clone()
    opt = SGDG([P], lr=0.5, stiefel=True)
    gen = torch.Generator().manual_seed(1)
    G0 = None
    for step in range(10):
        G = torch.randn(32, 32, generator=gen)
        G0 = G if G0 is None else G0
        loss = (P * G).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        err = _ortho_err(P)
        assert err < 5e-5, f"step {step}: |P^T P - I| = {err:.3e}"
    assert (P.detach() - P0).abs().max().item() > 1e-3, "parameter never moved"
    # Contrast: one EUCLIDEAN SGD step (same lr, same first gradient) departs the
    # manifold by O(lr * ||G||) — the Cayley step is what preserves it.
    err_sgd = _ortho_err(P0 - 0.5 * G0)
    assert err_sgd > 1e-1, f"contrast broken: plain-SGD ortho err only {err_sgd:.3e}"


# --------------------------------------------------------------------------------------
# 2. learn_rotations end-to-end on a tiny model
# --------------------------------------------------------------------------------------


def test_learn_rotations_orthonormal_and_moved():
    """A short quantized-objective run returns the full fold-convention key set, float64
    CPU matrices, all orthonormal after training, visibly moved from the init draws, and a
    complete per-step history."""
    model = _tiny_qwen3()
    steps = 6
    rs = learn_rotations(
        model,
        _calib_batches(),
        steps=steps,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    expected = {"R1"} | {f"model.layers.{i}.self_attn.R2" for i in range(N_LAYERS)}
    assert set(rs.rotations) == expected
    for name, R in rs.rotations.items():
        size = HIDDEN if name == "R1" else HEAD_DIM
        assert R.dtype == torch.float64 and R.device.type == "cpu"
        assert R.shape == (size, size)
    # Trained sets are polar-retracted on return: BOTH residual forms at fp64 SVD level.
    audit = rs.ortho_audit()
    assert max(audit.values()) < 1e-10, f"ortho audit (post-retraction): {audit}"
    assert len(rs.history) == steps
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rs.history)
    # Moved from init: compare against the seed-0 draws (== steps=0 output).
    init = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    dmax = (rs.R1 - init.R1).abs().max().item()
    assert dmax > 1e-4, f"R1 did not move from its init (max delta {dmax:.3e})"


def test_init_matches_fold_seed_draws():
    """steps=0 returns exactly (bitwise) the matrices fold_rotations draws for the same
    seed/mode — the trainer's init and the validated fold path share one RNG contract."""
    init = learn_rotations(
        _tiny_qwen3(), _calib_batches(), steps=0, objective_cfg=None, seed=5, log_every=0
    )
    folded = fold_rotations(_tiny_qwen3(), mode="hadamard", seed=5, use_r2=True)
    assert set(init.rotations) == set(folded)
    for k in folded:
        assert torch.equal(init.rotations[k], folded[k]), f"{k}: init != fold draw"


def test_loss_decreases_tiny_overfit():
    """Strict loss decrease on a single repeated batch (overfit): with the fake-quant
    objective on, CE depends on R through the quantization error, and a few Cayley steps
    must reduce it below the step-0 value."""
    model = _tiny_qwen3()
    batch = _calib_batches(n_batches=1, bs=2, seq=32, seed=11)
    rs = learn_rotations(
        model, batch, steps=25, lr=1.0, objective_cfg=TINY_W4A4, seed=0, log_every=0
    )
    losses = [r["loss"] for r in rs.history]
    assert min(losses[-5:]) < losses[0], (
        f"no strict decrease: first {losses[0]:.6f}, last5 {losses[-5:]}"
    )
    assert rs.history[-1]["r1_ortho"] < ORTHO_TOL_FP32_STEPS


def test_learned_fold_fp_equivalence():
    """Learned R fed through fold_rotations(R1=..., R2=...) keeps fp-equivalence with
    quantization off. The tolerance budgets the trained matrices' manifold drift (~1e-6
    after 6 fp32 steps) on O(1) logits; a wrong orientation or a missed fusion is O(1)."""
    rs = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=6,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    model = _tiny_qwen3()  # fresh, un-mutated model
    before = _logits(model)
    applied = fold_rotations(model, R1=rs.R1, R2=rs.R2)
    after = _logits(model)
    max_diff = (after - before).abs().max().item()
    assert torch.allclose(after, before, rtol=0, atol=1e-3), (
        f"max |delta logit| = {max_diff:.3e} > 1e-3"
    )
    # The returned dict is the applied (float64) matrices — bitwise the learned ones.
    for k, v in rs.rotations.items():
        assert torch.equal(applied[k], v), f"{k}: applied != learned"


def test_final_retraction_closes_rrt_gap():
    """The returned matrices are the polar retraction of the raw fp32 iterates: meta
    records per-matrix raw drift in BOTH residual forms plus the entry-wise projection
    distance; the post-retraction audit is at fp64-SVD level, far below the raw drift.
    (Motivating field measurement: raw 150-step R1 passes R^T R at ~5e-5 but sits at
    ~1e-3 in the R R^T form the fold consumes — basis-dependent max-entry residuals.)"""
    rs = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=8,
        lr=1.5,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    log = rs.meta["final_retraction"]
    assert set(log) == set(rs.rotations)
    audit = rs.ortho_audit()
    for k, rec in log.items():
        assert rec["raw_rtr"] > 0 and rec["raw_rrt"] > 0 and rec["delta_max"] > 0
        # projection moved entries on the order of the raw drift, not more than ~its size
        assert rec["delta_max"] < 10 * max(rec["raw_rtr"], rec["raw_rrt"])
        # retraction actually closed the residual: orders of magnitude below raw drift
        assert audit[k] < 1e-10 < rec["raw_rrt"]


def test_qwen3_qk_norm_bitwise_untouched_by_learn():
    """Qwen3 per-head q_norm/k_norm (head-space, post-projection) are bitwise identical
    after learn_rotations — the arch spec excludes them from fusion and rotation."""
    model = _tiny_qwen3()
    before = {
        n: p.data.clone() for n, p in model.named_parameters() if "q_norm" in n or "k_norm" in n
    }
    assert len(before) == 2 * N_LAYERS
    assert all(not torch.all(p == 1) for p in before.values())  # gains were randomized
    learn_rotations(
        model,
        _calib_batches(),
        steps=3,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p.data, before[n]), f"{n} changed"


def test_int8_static_objective_smoke():
    """INT8_DEFAULT_CFG axes (per-out-channel W8 + per-tensor static A8): trains on the
    Llama tiny model in BOTH static-scope variants (batch = scale tracks current R,
    run = monotone running max), stays orthonormal, and records a positive static amax
    per target linear."""
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
            steps=3,
            lr=1.0,
            objective_cfg=obj,
            seed=0,
            log_every=0,
        )
        assert max(rs.ortho_audit().values()) < 1e-10
        amax = rs.meta.get("static_act_amax", {})
        assert len(amax) == 7 * N_LAYERS, f"expected {7 * N_LAYERS} static amax entries"
        assert all(v > 0 for v in amax.values())
        assert rs.meta["objective"]["a_mode"] == "per_tensor_static"
        assert rs.meta["objective"]["a_static_scope"] == scope
    assert INT8_DEFAULT_OBJECTIVE.a_static_scope == "batch"  # preset default


# --------------------------------------------------------------------------------------
# 3. RotationSet save/load round-trip
# --------------------------------------------------------------------------------------


def test_rotation_set_save_load_roundtrip():
    """save() -> load() reproduces every matrix bitwise (flat fp64 R.bin-format dict),
    and load() refuses a non-orthogonal file."""
    rs = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=4,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        rs.save(path)
        rs2 = RotationSet.load(path)
        assert set(rs2.rotations) == set(rs.rotations)
        for k in rs.rotations:
            assert torch.equal(rs2.rotations[k], rs.rotations[k]), f"{k} changed in transit"
        # Corrupt one matrix off the manifold: load must refuse.
        bad = dict(rs.rotations)
        bad["R1"] = bad["R1"] * 1.5
        torch.save(bad, path)
        with pytest.raises(ValueError, match="not orthogonal"):
            RotationSet.load(path)
        # Raw-drift-style file (small non-orthogonal perturbation, like a legacy R.bin
        # written without the final retraction): plain load refuses, orthogonalize=True
        # retracts and passes.
        torch.manual_seed(3)
        drifted = dict(rs.rotations)
        drifted["R1"] = drifted["R1"] + 3e-4 * torch.randn_like(drifted["R1"])
        torch.save(drifted, path)
        try:
            RotationSet.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError("load() accepted a drifted (non-retracted) R1")
        rs3 = RotationSet.load(path, orthogonalize=True)
        assert max(rs3.ortho_audit().values()) < 1e-10
        # the retraction stayed near the drifted matrix (order of the perturbation)
        assert (rs3.R1 - drifted["R1"]).abs().max().item() < 1e-2
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------------------
# 4. fold_rotations external-matrix path
# --------------------------------------------------------------------------------------


def test_fold_external_matches_seed_path():
    """Feeding the seed path's returned matrices back through R1=/R2= reproduces every
    parameter bitwise on both architectures — the external path is the same fold."""
    for build in (_tiny_llama, _tiny_qwen3):
        model_seed = build()
        rots = fold_rotations(model_seed, mode="hadamard", seed=3, use_r2=True)
        model_ext = build()  # identical construction seed -> identical weights
        returned = fold_rotations(
            model_ext, R1=rots["R1"], R2={k: v for k, v in rots.items() if k != "R1"}
        )
        params_seed = dict(model_seed.named_parameters())
        for n, p in model_ext.named_parameters():
            assert torch.equal(p.data, params_seed[n].data), f"{build.__name__}: {n} differs"
        assert set(returned) == set(rots)
        for k in rots:
            assert torch.equal(returned[k], rots[k])


def test_fold_external_accepts_sequence_r2():
    """R2 as a plain list ordered by layer is equivalent to the keyed-dict form."""
    rots = fold_rotations(_tiny_qwen3(), mode="hadamard", seed=4)
    r2_list = [rots[f"model.layers.{i}.self_attn.R2"] for i in range(N_LAYERS)]
    model_a, model_b = _tiny_qwen3(), _tiny_qwen3()
    fold_rotations(model_a, R1=rots["R1"], R2=r2_list)
    fold_rotations(model_b, R1=rots["R1"], R2={k: v for k, v in rots.items() if k != "R1"})
    pb = dict(model_b.named_parameters())
    for n, p in model_a.named_parameters():
        assert torch.equal(p.data, pb[n].data), f"{n} differs between R2 forms"


def test_fold_external_validation_errors():
    """Bad external inputs raise ValueError: R2 without R1; use_r2=True with only R1;
    non-orthogonal R1; wrong R2 count."""
    rots = fold_rotations(_tiny_qwen3(), mode="hadamard", seed=0)
    R1 = rots["R1"]
    R2 = {k: v for k, v in rots.items() if k != "R1"}

    def expect_value_error(fn, what):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {what}")

    expect_value_error(lambda: fold_rotations(_tiny_qwen3(), R2=R2), "R2 without R1")
    expect_value_error(lambda: fold_rotations(_tiny_qwen3(), R1=R1), "use_r2=True with only R1")
    expect_value_error(
        lambda: fold_rotations(_tiny_qwen3(), R1=R1 * 1.5, R2=R2), "non-orthogonal R1"
    )
    expect_value_error(
        lambda: fold_rotations(
            _tiny_qwen3(), R1=R1, R2=[next(iter(R2.values()))]
        ),  # 1 matrix for 2 layers
        "wrong R2 count",
    )


def test_fold_external_r1_only_use_r2_false():
    """R1-only external fold (use_r2=False) matches the seed path with use_r2=False."""
    model_seed = _tiny_llama()
    rots = fold_rotations(model_seed, mode="hadamard", seed=6, use_r2=False)
    assert set(rots) == {"R1"}
    model_ext = _tiny_llama()
    fold_rotations(model_ext, R1=rots["R1"], use_r2=False)
    ps = dict(model_seed.named_parameters())
    for n, p in model_ext.named_parameters():
        assert torch.equal(p.data, ps[n].data), f"{n} differs"


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
