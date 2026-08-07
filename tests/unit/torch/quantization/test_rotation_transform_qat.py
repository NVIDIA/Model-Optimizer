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

"""Transform-QAT tests: jointly learned rotations R1/R2 + per-input-channel seam
diagonals (OSTQuant-style) in modelopt.torch.quantization.rotation.

Covers:
  1. Default-off backward compatibility: ``learn_seam_diag=False`` (present or absent)
     is bitwise identical — matching rotations/histories, ``seam_diags is None``, and
     the no-diag assembly path reproduces the pre-change formula bitwise.
  2. learn_seam_diag=True: gradients reach R1, every R2 AND every seam-diag parameter
     (and nothing else); loss decreases on a repeated-batch overfit; the assembled
     reparametrized model is function-preserving at the zeros init AND for arbitrary
     nonzero log-scales (quant off) — the structural seam-identity check.
  3. fold_seam_diags round-trip: learn 5 steps -> bake diags + R into a fresh model
     (fold_seam_diags then fold_rotations) -> logits match the assembled reparametrized
     model, and the folded weights match the assembly entry-for-entry; either fold
     order preserves the function.
  4. save/load round-trip including seam_diags; old-format (flat R.bin) files still
     load with seam_diags=None; seam_diags=None saves the legacy flat format.

Plain test_* functions with asserts: collectable by pytest, and also runnable without
it via ``python test_rotation_transform_qat.py`` (the __main__ driver runs every test
function and exits nonzero on any failure). CPU-only, tiny models, seconds per test.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # HARD CONSTRAINT: CPU-only, never touch the GPU

import sys
import tempfile
import traceback

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import stateless
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modelopt.torch.quantization.rotation import (
    QuantObjective,
    RotationSet,
    fold_rotations,
    fold_seam_diags,
    learn_rotations,
)
from modelopt.torch.quantization.rotation.learn import (
    _ATTN_PROJS,
    _MLP_PROJS,
    _SEAM_DIAGS_KEY,
    _ActQuantHooks,
    _assemble_effective_weights,
    _fq_weight,
)

VOCAB = 128
# Standard tiny fixture (same as test_rotation_{fold,learn,ext_learner}.py): HEAD_DIM
# deliberately decoupled from HIDDEN // num_attention_heads (32 vs 64//4 = 16), like
# Qwen3-0.6B. GQA is real (4 q heads on 2 kv heads), so the o-seam group expansion is
# exercised, not degenerate.
HIDDEN = 64
HEAD_DIM = 32
N_LAYERS = 2
N_KV = 2
INTERMEDIATE = 2 * HIDDEN  # 128
O_SEAM_DIM = N_KV * HEAD_DIM  # 64

# Tiny-model W4A4 (g=16 divides every in_features {64, 128, 128}), as in the sibling
# test files — once plain, once with the transform-QAT flag.
TINY_W4A4 = QuantObjective(
    name="tiny_w4a4", w_bits=4, w_group=16, a_bits=4, a_mode="per_token_dynamic"
)
TINY_W4A4_DIAG = QuantObjective(
    name="tiny_w4a4_diag",
    w_bits=4,
    w_group=16,
    a_bits=4,
    a_mode="per_token_dynamic",
    learn_seam_diag=True,
)


def _randomize_rmsnorm_gains(model):
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)


def _build(cfg_cls, model_cls, tie=False):
    torch.manual_seed(1234)
    cfg = cfg_cls(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attn_implementation="eager",
    )
    model = model_cls(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _tiny_llama(tie=False):
    return _build(LlamaConfig, LlamaForCausalLM, tie)


def _tiny_qwen3(tie=False):
    return _build(Qwen3Config, Qwen3ForCausalLM, tie)


def _calib_batches(n_batches=2, bs=2, seq=16, seed=7):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (bs, seq)) for _ in range(n_batches)]


def _logits(model):
    torch.manual_seed(99)
    ids = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        return model(ids).logits


def _base_weights(model):
    sd = dict(model.named_parameters())
    base = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].data,
        "lm_head.weight": sd["lm_head.weight"].data,
    }
    for i in range(len(model.model.layers)):
        for proj in _ATTN_PROJS + _MLP_PROJS:
            name = f"model.layers.{i}.{proj}.weight"
            base[name] = sd[name].data
    return base


def _r_leaves(rotations, n_layers):
    R1 = nn.Parameter(rotations["R1"].to(torch.float32))
    R2s = [
        nn.Parameter(rotations[f"model.layers.{i}.self_attn.R2"].to(torch.float32))
        for i in range(n_layers)
    ]
    return R1, R2s


def _diag_leaves(n_layers, fill=None):
    """Per-layer {down, o} log-scale leaves: zeros (identity) or a provided filler fn."""
    out = []
    for i in range(n_layers):
        if fill is None:
            d = torch.zeros(INTERMEDIATE)
            o = torch.zeros(O_SEAM_DIM)
        else:
            d, o = fill(i)
        out.append({"down": nn.Parameter(d.float()), "o": nn.Parameter(o.float())})
    return out


def _assembled_logits(model, eff):
    torch.manual_seed(99)
    ids = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad(), stateless._reparametrize_module(model, eff):
        return model(ids).logits


# --------------------------------------------------------------------------------------
# 1. Default-off = bitwise identical behavior to before
# --------------------------------------------------------------------------------------


def _assemble_pre_change(base, R1, R2s, n_layers, head_dim, objective, out_dtype):
    """Rotation-only reference assembly (no seam diagonals) — the bitwise
    oracle for the learn_seam_diag=False path."""
    compute = R1.dtype
    d = head_dim

    def fin(w):
        if objective is not None and objective.w_bits is not None:
            w = _fq_weight(w, objective)
        return w.to(out_dtype)

    eff = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        eff[name] = (base[name].to(compute) @ R1).to(out_dtype)

    for i in range(n_layers):
        R2 = R2s[i]
        pre = f"model.layers.{i}."
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.up_proj"):
            n = pre + proj + ".weight"
            eff[n] = fin(base[n].to(compute) @ R1)

        n = pre + "mlp.down_proj.weight"
        eff[n] = fin(R1.t() @ base[n].to(compute))

        n = pre + "self_attn.v_proj.weight"
        a = base[n].to(compute) @ R1
        o_f, i_f = a.shape
        a = (a.t().reshape(i_f, o_f // d, d) @ R2).reshape(i_f, o_f).t().contiguous()
        eff[n] = fin(a)

        n = pre + "self_attn.o_proj.weight"
        w = R1.t() @ base[n].to(compute)
        o_f, i_f = w.shape
        eff[n] = fin((w.reshape(o_f, i_f // d, d) @ R2).reshape(o_f, i_f))
    return eff


def test_default_off_bitwise_identical():
    """learn_seam_diag defaults to False, and False (explicit or absent) is one code
    path: (i) steps=3 runs with the field absent vs. explicitly False give bitwise-equal
    rotations and identical loss histories, both with seam_diags=None; (ii) the no-diag
    assembly reproduces the pre-change formula bitwise for every effective weight."""
    assert QuantObjective(name="x").learn_seam_diag is False  # default off
    explicit_off = QuantObjective(
        name="tiny_w4a4",
        w_bits=4,
        w_group=16,
        a_bits=4,
        a_mode="per_token_dynamic",
        learn_seam_diag=False,
    )
    assert explicit_off == TINY_W4A4  # frozen dataclass equality: absent == False

    runs = []
    for obj in (TINY_W4A4, explicit_off):
        rs = learn_rotations(
            _tiny_qwen3(),
            _calib_batches(),
            steps=3,
            lr=1.0,
            objective_cfg=obj,
            seed=0,
            log_every=0,
        )
        assert rs.seam_diags is None, "learn_seam_diag=False must not produce seam_diags"
        assert "seam_diag" not in rs.meta
        runs.append(rs)
    a, b = runs
    assert set(a.rotations) == set(b.rotations)
    for k in a.rotations:
        assert torch.equal(a.rotations[k], b.rotations[k]), f"{k}: off-path not bitwise"
    assert [r["loss"] for r in a.history] == [r["loss"] for r in b.history]

    # Assembly-level bitwise oracle vs. the verbatim pre-change formula.
    model = _tiny_qwen3()
    init = learn_rotations(model, [], steps=0, objective_cfg=None, seed=0, log_every=0)
    R1, R2s = _r_leaves(init.rotations, N_LAYERS)
    base = _base_weights(model)
    for obj in (TINY_W4A4, None):
        new = _assemble_effective_weights(
            base,
            R1,
            R2s,
            N_LAYERS,
            HEAD_DIM,
            obj,
            torch.float32,
            seam_diag_params=None,
        )
        old = _assemble_pre_change(base, R1, R2s, N_LAYERS, HEAD_DIM, obj, torch.float32)
        assert set(new) == set(old)
        for k in old:
            assert torch.equal(new[k], old[k]), f"{k}: no-diag assembly != pre-change (obj={obj})"


def test_step0_loss_identical_diag_on_vs_off():
    """At the zeros init the diagonals are exact identity scales, so the step-0 loss of a
    learn_seam_diag=True run equals the rotation-only run's bitwise (trajectories may
    diverge from step 1 once Adam moves the diagonals)."""
    on = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=2,
        lr=1.0,
        objective_cfg=TINY_W4A4_DIAG,
        seed=0,
        log_every=0,
    )
    off = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=2,
        lr=1.0,
        objective_cfg=TINY_W4A4,
        seed=0,
        log_every=0,
    )
    assert on.history[0]["loss"] == off.history[0]["loss"], (
        f"step-0 loss differs: diag-on {on.history[0]['loss']} vs off {off.history[0]['loss']}"
    )


# --------------------------------------------------------------------------------------
# 2. learn_seam_diag=True: gradients, descent, structural function preservation
# --------------------------------------------------------------------------------------


def test_grads_reach_rotations_and_diags_and_nothing_else():
    """One step-0 forward/backward with diag leaves (zeros init) on both archs: finite
    loss; non-None, finite, nonzero grads on R1, every R2, and every down/o log-scale;
    no grad on any model parameter."""
    for build in (_tiny_llama, _tiny_qwen3):
        model = build()
        init = learn_rotations(model, [], steps=0, objective_cfg=None, seed=0, log_every=0)
        R1, R2s = _r_leaves(init.rotations, N_LAYERS)
        diag = _diag_leaves(N_LAYERS)
        base = _base_weights(model)
        hooks = _ActQuantHooks(TINY_W4A4_DIAG)
        n_hooked = hooks.attach(model)
        assert n_hooked == 7 * N_LAYERS
        torch.manual_seed(42)
        ids = torch.randint(0, VOCAB, (2, 16))
        try:
            eff = _assemble_effective_weights(
                base,
                R1,
                R2s,
                N_LAYERS,
                HEAD_DIM,
                TINY_W4A4_DIAG,
                torch.float32,
                seam_diag_params=diag,
            )
            with stateless._reparametrize_module(model, eff):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                loss.backward()
        finally:
            hooks.remove()
        tag = build.__name__
        assert torch.isfinite(loss), f"{tag}: loss not finite"
        assert R1.grad is not None and R1.grad.abs().max() > 0, f"{tag}: R1 grad missing/zero"
        for i, r2 in enumerate(R2s):
            assert r2.grad is not None and torch.isfinite(r2.grad).all(), f"{tag}: R2[{i}]"
            assert r2.grad.abs().max() > 0, f"{tag}: R2[{i}] grad identically zero"
        for i, sp in enumerate(diag):
            for key in ("down", "o"):
                g = sp[key].grad
                assert g is not None, f"{tag}: no grad reached log_s_{key}[{i}]"
                assert torch.isfinite(g).all(), f"{tag}: log_s_{key}[{i}] grad not finite"
                assert g.abs().max() > 0, f"{tag}: log_s_{key}[{i}] grad identically zero"
        for name, p in model.named_parameters():
            assert p.grad is None, f"{tag}: model param {name} received a gradient"


def test_learn_with_diag_trains_and_moves_scales():
    """learn_rotations with learn_seam_diag=True: seam_diags has every layer with the
    right shapes, strictly positive, visibly moved off the identity; the rotations stay
    orthonormal; meta records the diag group; loss decreases on a repeated batch."""
    batch = _calib_batches(n_batches=1, bs=2, seq=32, seed=11)
    rs = learn_rotations(
        _tiny_qwen3(),
        batch,
        steps=25,
        lr=1.0,
        objective_cfg=TINY_W4A4_DIAG,
        seed=0,
        log_every=0,
    )
    assert rs.seam_diags is not None and set(rs.seam_diags) == set(range(N_LAYERS))
    moved = 0.0
    for i in range(N_LAYERS):
        sd, so = rs.seam_diags[i]["down"], rs.seam_diags[i]["o"]
        assert sd.shape == (INTERMEDIATE,) and so.shape == (O_SEAM_DIM,)
        assert sd.dtype == torch.float64 and so.dtype == torch.float64
        assert sd.device.type == "cpu" and so.device.type == "cpu"
        assert (sd > 0).all() and (so > 0).all()
        moved = max(moved, (sd - 1).abs().max().item(), (so - 1).abs().max().item())
    assert moved > 1e-4, f"seam scales never moved off identity (max |s-1| = {moved:.3e})"
    assert max(rs.ortho_audit().values()) < 1e-10
    assert rs.meta["objective"]["learn_seam_diag"] is True
    assert rs.meta["seam_diag"]["lr"] == 1e-2
    assert 0 < rs.meta["seam_diag"]["s_min"] <= rs.meta["seam_diag"]["s_max"]
    losses = [r["loss"] for r in rs.history]
    assert min(losses[-5:]) < losses[0], (
        f"no decrease with diag learning: first {losses[0]:.6f}, last5 {losses[-5:]}"
    )


def test_steps0_diag_identity_and_rng_stream_unchanged():
    """steps=0 with learn_seam_diag=True returns identity scales (exact ones — zeros
    init consumes no RNG) and the SAME bitwise rotation draws as the seed path — the
    diag machinery must not shift the RNG stream."""
    init = learn_rotations(
        _tiny_qwen3(), [], steps=0, objective_cfg=TINY_W4A4_DIAG, seed=5, log_every=0
    )
    assert init.seam_diags is not None
    for i in range(N_LAYERS):
        for key, dim in (("down", INTERMEDIATE), ("o", O_SEAM_DIM)):
            s = init.seam_diags[i][key]
            assert torch.equal(s, torch.ones(dim, dtype=torch.float64)), (
                f"steps=0 seam scale [{i}][{key}] is not exact identity"
            )
    folded = fold_rotations(_tiny_qwen3(), mode="hadamard", seed=5, use_r2=True)
    for k in folded:
        assert torch.equal(init.rotations[k], folded[k]), f"{k}: RNG stream shifted"


def test_assembled_function_preserving_at_init_and_any_diag():
    """Quant off: the assembled reparametrized model equals the plain model function
    (logit-level) BOTH at the zeros init and — the real structural check — for random
    nonzero log-scales at every seam (any positive diagonal is an exact identity; a
    wrong axis, wrong GQA expansion, or wrong diag/R2 order would be O(1) off)."""
    for build in (_tiny_llama, _tiny_qwen3):
        plain = _logits(build())
        model = build()
        init = learn_rotations(model, [], steps=0, objective_cfg=None, seed=0, log_every=0)
        R1, R2s = _r_leaves(init.rotations, N_LAYERS)
        base = _base_weights(model)

        def rand_fill(i):
            g = torch.Generator().manual_seed(100 + i)
            return (
                0.4 * torch.randn(INTERMEDIATE, generator=g),
                0.4 * torch.randn(O_SEAM_DIM, generator=g),
            )

        for label, fill in (("zeros-init", None), ("random-diag", rand_fill)):
            diag = _diag_leaves(N_LAYERS, fill=fill)
            if label == "random-diag":  # make sure the case is not vacuous
                assert max(p.abs().max().item() for sp in diag for p in sp.values()) > 0.3
            eff = _assemble_effective_weights(
                base,
                R1,
                R2s,
                N_LAYERS,
                HEAD_DIM,
                None,
                torch.float32,
                seam_diag_params=diag,
            )
            out = _assembled_logits(model, eff)
            dmax = (out - plain).abs().max().item()
            assert torch.allclose(out, plain, rtol=0, atol=1e-3), (
                f"{build.__name__}/{label}: assembled model broke function "
                f"preservation (max |delta logit| = {dmax:.3e})"
            )


# --------------------------------------------------------------------------------------
# 3. fold_seam_diags round-trip against the assembled reparametrized model
# --------------------------------------------------------------------------------------


def test_fold_seam_diags_roundtrip_matches_assembly():
    """Learn 5 steps (diag on) -> bake into a fresh model with fold_seam_diags then
    fold_rotations -> (i) logits match the assembled reparametrized model within bf16-
    level tolerance, (ii) the folded weights match the assembly's effective weights
    entry-for-entry (the two folds compose to exactly the assembly's prefold-inside /
    rotation-outside convention), (iii) the reverse fold order preserves the function
    too, and (iv) the fp16-safety clamp never bit."""
    rs = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=5,
        lr=1.0,
        objective_cfg=TINY_W4A4_DIAG,
        seed=0,
        log_every=0,
    )
    moved = max(
        (rs.seam_diags[i][k] - 1).abs().max().item() for i in range(N_LAYERS) for k in ("down", "o")
    )
    assert moved > 1e-4, "trained scales are still identity — round-trip would be vacuous"

    # Assembled reference: prepped (untied+fused) model reparametrized with the learned
    # R and log-scales, quantization off.
    model_asm = _tiny_qwen3()
    learn_rotations(model_asm, [], steps=0, objective_cfg=None, seed=0, log_every=0)
    R1, R2s = _r_leaves(rs.rotations, N_LAYERS)
    diag = _diag_leaves(
        N_LAYERS,
        fill=lambda i: (
            torch.log(rs.seam_diags[i]["down"]).float(),
            torch.log(rs.seam_diags[i]["o"]).float(),
        ),
    )
    eff = _assemble_effective_weights(
        _base_weights(model_asm),
        R1,
        R2s,
        N_LAYERS,
        HEAD_DIM,
        None,
        torch.float32,
        seam_diag_params=diag,
    )
    ref = _assembled_logits(model_asm, eff)

    # Fold path on a fresh model: diags first, then rotations (assembly convention).
    model_fold = _tiny_qwen3()
    evidence = fold_seam_diags(model_fold, rs.seam_diags)
    assert set(evidence["layers"]) == set(range(N_LAYERS))
    assert not any(rec["clamped"] for rec in evidence["layers"].values())
    assert all(rec["down_s_spread"] > 1 for rec in evidence["layers"].values())
    fold_rotations(model_fold, R1=rs.R1, R2=rs.R2)
    out = _logits(model_fold)
    dmax = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=0, atol=1e-2), (
        f"folded model != assembled model (max |delta logit| = {dmax:.3e})"
    )

    # Weight-level: every effective weight is reproduced by the composed folds.
    params = dict(model_fold.named_parameters())
    for name, w in eff.items():
        d = (params[name].data - w).abs().max().item()
        assert d < 1e-4, f"{name}: folded weight != assembled effective weight ({d:.3e})"

    # Reverse order (rotations first, then diags) is also a functional identity.
    model_rev = _tiny_qwen3()
    fold_rotations(model_rev, R1=rs.R1, R2=rs.R2)
    fold_seam_diags(model_rev, rs.seam_diags)
    out_rev = _logits(model_rev)
    assert torch.allclose(out_rev, ref, rtol=0, atol=1e-2), (
        f"reverse fold order broke function (max delta {(out_rev - ref).abs().max().item():.3e})"
    )


def test_fold_seam_diags_validation_and_clamp():
    """fold_seam_diags refuses wrong-length or non-positive scales and unknown layers;
    the smax clamp bites (and is reported) for scales beyond the ceiling; a plain
    identity fold is a bf16-free exact no-op on an fp32 model up to fp64 round-trip."""
    ones_d, ones_o = torch.ones(INTERMEDIATE), torch.ones(O_SEAM_DIM)

    def expect_error(fn, what, exc=ValueError):
        try:
            fn()
        except exc:
            pass
        else:
            raise AssertionError(f"expected {exc.__name__}: {what}")

    expect_error(
        lambda: fold_seam_diags(_tiny_qwen3(), {0: {"down": torch.ones(7), "o": ones_o}}),
        "wrong down length",
    )
    expect_error(
        lambda: fold_seam_diags(_tiny_qwen3(), {0: {"down": -ones_d, "o": ones_o}}),
        "negative scales",
    )
    expect_error(
        lambda: fold_seam_diags(_tiny_qwen3(), {99: {"down": ones_d, "o": ones_o}}),
        "layer index out of range",
    )
    expect_error(
        lambda: fold_seam_diags(_tiny_qwen3(), {0: {"down": ones_d}}),
        "missing 'o' key",
    )

    # Identity scales: exact no-op (fp64 round-trip of unchanged values is bitwise).
    model = _tiny_qwen3()
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    ev = fold_seam_diags(model, {i: {"down": ones_d, "o": ones_o} for i in range(N_LAYERS)})
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n]), f"identity fold changed {n}"
    assert not any(rec["clamped"] for rec in ev["layers"].values())

    # Clamp: s=512 with smax=256 folds as 256 and reports clamped=True.
    model = _tiny_qwen3()
    up0 = model.model.layers[0].mlp.up_proj.weight.data.clone()
    ev = fold_seam_diags(model, {0: {"down": 512.0 * ones_d, "o": ones_o}}, smax=256.0)
    assert ev["layers"][0]["clamped"] is True
    assert ev["layers"][0]["down_s_max"] == 512.0  # telemetry reports the raw scale
    got = model.model.layers[0].mlp.up_proj.weight.data
    want = (up0.to(torch.float64) / 256.0).to(torch.float32)
    assert torch.equal(got, want), "clamped fold did not use the smax ceiling"


# --------------------------------------------------------------------------------------
# 4. save/load round-trip incl. seam_diags; old-format compatibility
# --------------------------------------------------------------------------------------


def test_save_load_roundtrip_with_seam_diags_and_old_format():
    """New format round-trips rotations AND seam_diags bitwise; seam_diags=None writes
    the legacy flat dict (no reserved key); an old-format (pure-rotation) file loads
    with seam_diags=None; the ortho gate and orthogonalize=True still work with the
    seam payload present."""
    rs = learn_rotations(
        _tiny_qwen3(),
        _calib_batches(),
        steps=4,
        lr=1.0,
        objective_cfg=TINY_W4A4_DIAG,
        seed=0,
        log_every=0,
    )
    assert rs.seam_diags is not None
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        # New-format round-trip: everything bitwise.
        rs.save(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert _SEAM_DIAGS_KEY in raw
        rs2 = RotationSet.load(path)
        assert set(rs2.rotations) == set(rs.rotations)
        for k in rs.rotations:
            assert torch.equal(rs2.rotations[k], rs.rotations[k]), f"{k} changed in transit"
        assert rs2.seam_diags is not None and set(rs2.seam_diags) == set(rs.seam_diags)
        for i in rs.seam_diags:
            for key in ("down", "o"):
                assert torch.equal(rs2.seam_diags[i][key], rs.seam_diags[i][key]), (
                    f"seam_diags[{i}][{key}] changed in transit"
                )

        # seam_diags=None saves the legacy flat format: only rotation keys on disk.
        rs_plain = RotationSet(rotations=dict(rs.rotations))
        rs_plain.save(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert set(raw) == set(rs.rotations), "legacy save format changed"

        # Old-format file (flat rotation dict, e.g. a legacy R.bin): loads fine,
        # seam_diags=None.
        torch.save(dict(rs.rotations), path)
        rs3 = RotationSet.load(path)
        assert rs3.seam_diags is None
        for k in rs.rotations:
            assert torch.equal(rs3.rotations[k], rs.rotations[k])

        # Ortho gate still guards new-format files.
        bad = dict(rs.rotations)
        bad["R1"] = bad["R1"] * 1.5
        bad[_SEAM_DIAGS_KEY] = {
            int(i): {k: v.clone() for k, v in pair.items()} for i, pair in rs.seam_diags.items()
        }
        torch.save(bad, path)
        with pytest.raises(ValueError, match="not orthogonal"):
            RotationSet.load(path)

        # orthogonalize=True retracts the rotations and PRESERVES the seam payload.
        torch.manual_seed(3)
        drifted = dict(rs.rotations)
        drifted["R1"] = drifted["R1"] + 3e-4 * torch.randn_like(drifted["R1"])
        drifted[_SEAM_DIAGS_KEY] = bad[_SEAM_DIAGS_KEY]
        torch.save(drifted, path)
        rs4 = RotationSet.load(path, orthogonalize=True)
        assert max(rs4.ortho_audit().values()) < 1e-10
        assert rs4.seam_diags is not None
        for i in rs.seam_diags:
            for key in ("down", "o"):
                assert torch.equal(rs4.seam_diags[i][key], rs.seam_diags[i][key])
    finally:
        os.unlink(path)


def test_rotation_set_rejects_bad_seam_diags():
    """The RotationSet constructor validates the seam payload: non-positive scales and
    wrong key sets are refused; int-like string layer keys are normalized to int."""
    rots = fold_rotations(_tiny_qwen3(), mode="hadamard", seed=0)
    good = {
        i: {"down": torch.ones(INTERMEDIATE), "o": torch.ones(O_SEAM_DIM)} for i in range(N_LAYERS)
    }
    rs = RotationSet(rotations=dict(rots), seam_diags={str(i): v for i, v in good.items()})
    assert set(rs.seam_diags) == set(range(N_LAYERS))  # str keys normalized
    assert rs.seam_diags[0]["down"].dtype == torch.float64

    def expect_value_error(seam, what):
        try:
            RotationSet(rotations=dict(rots), seam_diags=seam)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {what}")

    neg = {0: {"down": -torch.ones(INTERMEDIATE), "o": torch.ones(O_SEAM_DIM)}}
    expect_value_error(neg, "negative scales")
    zero = {0: {"down": torch.zeros(INTERMEDIATE), "o": torch.ones(O_SEAM_DIM)}}
    expect_value_error(zero, "zero scales")
    missing = {0: {"down": torch.ones(INTERMEDIATE)}}
    expect_value_error(missing, "missing 'o' key")
    extra = {0: {"down": torch.ones(INTERMEDIATE), "o": torch.ones(O_SEAM_DIM), "x": 1}}
    expect_value_error(extra, "extra key")


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
