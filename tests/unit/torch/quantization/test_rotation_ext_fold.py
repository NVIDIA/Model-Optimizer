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

"""Extensive fold property tests (T20.1) for
modelopt.torch.quantization.rotation.fold_rotations.

Coverage beyond test_rotation_fold.py:
  1. an architecture sweep over {Llama, Qwen3} x {GQA 1,2,4} x {head_dim ==/!= hidden/heads}
     x {tied, untied} x 2 fold seeds x {hadamard (incl. the Paley had-12 branch), random}
     with the full identity-invariant bundle checked per cell;
  2. external-path == seed-path bitwise on every sweep config;
  3. idempotent norm fusion (direct re-fusion AND a full identity-matrix refold);
  4. R.bin round-trip of fold-returned dicts through RotationSet save/load, including the
     off-manifold refusal and the orthogonalize=True polar retraction;
  5. use_r2=False on every config, and R2-as-list == R2-as-str-dict == R2-as-int-dict;
  6. an orientation oracle on a hand-built 1-layer model: reader W@R1 / writer R1^T W /
     v-rows R2 / o-cols R2 verified against independent fp64 manual matmuls of the tiny
     network (weight-level and seam-level), plus the RMSNorm rotation-invariance identity.

Plain test_* functions with asserts: collectable by pytest, and also runnable without it
via ``python test_rotation_ext_fold.py`` (the __main__ driver runs every test function and
exits nonzero on any failure). CPU-only, tiny models.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only: never touch a GPU (a GPU chain runs)

import sys
import tempfile
import traceback
from dataclasses import dataclass

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modelopt.torch.quantization.rotation import RotationSet, fold_rotations
from modelopt.torch.quantization.rotation.fold import _fuse_norm_into_linears

VOCAB = 96

# FP-equivalence tolerance: same budget as test_rotation_fold.py (fold math is float64;
# error enters only via the fp64 -> fp32 cast per rewritten weight; measured max |delta
# logit| ~3e-7 on these sizes, so 1e-4 keeps >100x headroom while still catching any real
# transform bug, which perturbs logits by O(1)).
ATOL_FP32 = 1e-4


# --------------------------------------------------------------------------------------
# Config sweep: deterministic specs spanning the requested axes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    name: str
    arch: str  # "llama" | "qwen3"
    hidden: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    tie: bool
    mode: str  # rotation mode for the seed path

    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.n_kv

    @property
    def head_dim_coincident(self) -> bool:
        return self.head_dim == self.hidden // self.n_heads


# Spanning set (~8 tiny configs). head_dim is DELIBERATELY decoupled from hidden/heads on
# most cells (like Qwen3-0.6B: 128 vs 1024/16 = 64) so a head_dim-resolution regression to
# the wrong fallback formula cannot pass silently. hidden 48 / head_dim 24 exercise the
# Paley had-12 branch of the hadamard generator; hidden 36 / head_dim 12 have NO 2^k*K
# Hadamard decomposition, forcing (and covering) the mode="random" path.
_CONFIGS = (
    _Spec("llama_g1_hdEQ", "llama", 64, 2, 4, 4, 16, tie=False, mode="hadamard"),
    _Spec("llama_g2_hdNE_tied", "llama", 64, 2, 4, 2, 32, tie=True, mode="hadamard"),
    _Spec("llama_g4_hdEQ_rand", "llama", 32, 3, 4, 1, 8, tie=False, mode="random"),
    _Spec("llama_g2_paley_tied", "llama", 48, 1, 4, 2, 24, tie=True, mode="hadamard"),
    _Spec("qwen3_g1_hdNE", "qwen3", 64, 2, 4, 4, 32, tie=False, mode="hadamard"),
    _Spec("qwen3_g4_hdEQ_tied", "qwen3", 64, 2, 8, 2, 8, tie=True, mode="hadamard"),
    _Spec("qwen3_g1_odd_rand_tied", "qwen3", 36, 1, 2, 2, 12, tie=True, mode="random"),
    _Spec("qwen3_g2_paley", "qwen3", 48, 2, 4, 2, 16, tie=False, mode="hadamard"),
)

_FOLD_SEEDS = (0, 1)


def _randomize_rmsnorm_gains(model):
    """Set every RMSNorm gain to a random non-one value: fresh HF models initialize norm
    weights to ones, which would make the norm-fusion math and the fused-to-ones /
    q_norm-untouched checks vacuous."""
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)


def _build(spec: _Spec):
    """Deterministic build: equal specs give bitwise-identical models (fixed init seed)."""
    torch.manual_seed(1234)
    kwargs = {
        "vocab_size": VOCAB,
        "hidden_size": spec.hidden,
        "intermediate_size": 2 * spec.hidden,
        "num_hidden_layers": spec.n_layers,
        "num_attention_heads": spec.n_heads,
        "num_key_value_heads": spec.n_kv,
        "head_dim": spec.head_dim,
        "max_position_embeddings": 128,
        "tie_word_embeddings": spec.tie,
        "attn_implementation": "eager",
    }
    if spec.arch == "llama":
        model = LlamaForCausalLM(LlamaConfig(**kwargs)).eval()
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**kwargs)).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _logits(model, vocab=VOCAB):
    torch.manual_seed(99)
    ids = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        return model(ids).logits


def _r2_keys(n_layers):
    return {f"model.layers.{i}.self_attn.R2" for i in range(n_layers)}


def test_spec_table_spans_axes():
    """Meta-test: the sweep table really spans every requested axis, so the sweep tests
    below cannot silently lose coverage if the table is edited."""
    assert len(_CONFIGS) == 8
    assert len({s.name for s in _CONFIGS}) == 8
    for arch in ("llama", "qwen3"):
        sub = [s for s in _CONFIGS if s.arch == arch]
        assert {s.gqa_ratio for s in sub} == {1, 2, 4}, f"{arch}: GQA ratios not spanned"
        assert {s.head_dim_coincident for s in sub} == {True, False}, (
            f"{arch}: need both head_dim == and != hidden/heads"
        )
        assert {s.tie for s in sub} == {True, False}, f"{arch}: need tied and untied"
    assert {s.mode for s in _CONFIGS} == {"hadamard", "random"}
    assert len(_FOLD_SEEDS) == 2
    # the Paley (48 = 2^2*12) and no-Hadamard-decomposition (36) cells are present
    assert any(s.hidden == 48 and s.mode == "hadamard" for s in _CONFIGS)
    assert any(s.hidden == 36 and s.mode == "random" for s in _CONFIGS)


# --------------------------------------------------------------------------------------
# 1. Randomized architecture sweep: identity invariants on every config x seed
# --------------------------------------------------------------------------------------


def test_arch_sweep_fold_identity():
    """For every spanning config x 2 fold seeds: fold_rotations (seed path) keeps fp32
    logits equal within ATOL_FP32; every pre-existing parameter shape is unchanged (the
    only allowed new parameter is the untied lm_head.weight, embed-shaped); Qwen3
    q/k_norm are bitwise untouched; fused norms are exactly ones; the returned dict has
    the R.bin key set with correctly-sized orthonormal fp64 CPU matrices (R2 sized
    head_dim x head_dim — catches head_dim-resolution regressions per config)."""
    for spec in _CONFIGS:
        for fold_seed in _FOLD_SEEDS:
            tag = f"{spec.name} seed={fold_seed}"
            model = _build(spec)
            shapes_before = {n: tuple(p.shape) for n, p in model.named_parameters()}
            embed_before = model.model.embed_tokens.weight.data.clone()
            qk_before = {
                n: p.data.clone()
                for n, p in model.named_parameters()
                if "q_norm" in n or "k_norm" in n
            }
            if spec.arch == "qwen3":
                assert len(qk_before) == 2 * spec.n_layers, tag
                assert all(not torch.all(v == 1) for v in qk_before.values()), (
                    f"{tag}: q/k_norm gains not randomized — untouched check would be vacuous"
                )
            else:
                assert not qk_before, tag

            before = _logits(model)
            rots = fold_rotations(model, mode=spec.mode, seed=fold_seed, use_r2=True)
            after = _logits(model)

            # (a) fp32 functional identity
            max_diff = (after - before).abs().max().item()
            assert torch.allclose(after, before, rtol=0, atol=ATOL_FP32), (
                f"{tag}: max |delta logit| = {max_diff:.3e} > {ATOL_FP32}"
            )
            # ... and the fold really rewrote weights (guards against a vacuous no-op)
            assert not torch.equal(model.model.embed_tokens.weight.data, embed_before), (
                f"{tag}: embed_tokens unchanged — fold was a no-op"
            )

            # (b) parameter shapes unchanged; only allowed new param: untied lm_head
            params_after = dict(model.named_parameters())
            for n, shp in shapes_before.items():
                assert n in params_after, f"{tag}: parameter {n} disappeared"
                assert tuple(params_after[n].shape) == shp, f"{tag}: shape of {n} changed"
            new = set(params_after) - set(shapes_before)
            if spec.tie:
                assert new == {"lm_head.weight"}, f"{tag}: unexpected new params {new}"
                assert tuple(params_after["lm_head.weight"].shape) == tuple(
                    model.model.embed_tokens.weight.shape
                ), f"{tag}: untied lm_head has wrong shape"
            else:
                assert not new, f"{tag}: unexpected new params {new}"

            # (c) Qwen3 q/k_norm bitwise untouched
            for n, v in qk_before.items():
                assert torch.equal(params_after[n].data, v), f"{tag}: {n} changed"

            # (d) fused norms exactly ones
            for li, layer in enumerate(model.model.layers):
                assert torch.all(layer.input_layernorm.weight == 1), (
                    f"{tag}: layer {li} input_layernorm not ones"
                )
                assert torch.all(layer.post_attention_layernorm.weight == 1), (
                    f"{tag}: layer {li} post_attention_layernorm not ones"
                )
            assert torch.all(model.model.norm.weight == 1), f"{tag}: final norm not ones"

            # (e) returned rotations: key set, dtype/device, per-config sizes, orthonormal
            assert set(rots) == {"R1"} | _r2_keys(spec.n_layers), f"{tag}: bad key set"
            for k, R in rots.items():
                size = spec.hidden if k == "R1" else spec.head_dim
                assert R.dtype == torch.float64 and R.device.type == "cpu", f"{tag}: {k}"
                assert R.shape == (size, size), (
                    f"{tag}: {k} shape {tuple(R.shape)} != {(size, size)}"
                )
                err = (R @ R.T - torch.eye(size, dtype=torch.float64)).abs().max().item()
                assert err < 1e-10, f"{tag}: {k} max |R R^T - I| = {err:.3e}"


# --------------------------------------------------------------------------------------
# 2. External path == seed path bitwise on every config
# --------------------------------------------------------------------------------------


def test_external_path_bitwise_matches_seed_path():
    """For every sweep config: feeding the seed path's returned matrices back through
    R1=/R2= on an identically-constructed model reproduces every parameter bitwise, and
    the external fold returns the same matrices bitwise."""
    for spec in _CONFIGS:
        m_seed = _build(spec)
        rots = fold_rotations(m_seed, mode=spec.mode, seed=0, use_r2=True)
        m_ext = _build(spec)  # identical construction seed -> identical weights
        returned = fold_rotations(
            m_ext, R1=rots["R1"], R2={k: v for k, v in rots.items() if k != "R1"}
        )
        ps = dict(m_seed.named_parameters())
        pe = dict(m_ext.named_parameters())
        assert set(ps) == set(pe), f"{spec.name}: parameter sets differ"
        for n, p in pe.items():
            assert torch.equal(p.data, ps[n].data), f"{spec.name}: {n} differs"
        assert set(returned) == set(rots), f"{spec.name}: returned key set differs"
        for k in rots:
            assert torch.equal(returned[k], rots[k]), f"{spec.name}: returned {k} differs"


# --------------------------------------------------------------------------------------
# 3. Idempotent norm fusion
# --------------------------------------------------------------------------------------


def test_idempotent_norm_fusion_refold():
    """On an already-folded model (all norm gains exactly ones): (a) re-running the norm
    fusion directly is a bitwise no-op on every weight; (b) a full identity refold
    (external R1 = I, R2 = I) — which re-runs untie + fusion + all rotation applications —
    is a bitwise no-op on every parameter (fp64 multiply by 1.0 / matmul by I is exact)."""
    for spec in (_CONFIGS[1], _CONFIGS[4]):  # llama tied + qwen3 untied
        model = _build(spec)
        fold_rotations(model, mode=spec.mode, seed=0, use_r2=True)
        snap = {n: p.data.clone() for n, p in model.named_parameters()}

        # (a) direct re-fusion of every already-fused edge
        for layer in model.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            _fuse_norm_into_linears(layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj])
            _fuse_norm_into_linears(layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj])
        _fuse_norm_into_linears(model.model.norm, [model.lm_head])
        for n, p in model.named_parameters():
            assert torch.equal(p.data, snap[n]), f"{spec.name}: re-fusing fused norms changed {n}"

        # (b) identity refold through the full public pipeline
        eye_h = torch.eye(spec.hidden, dtype=torch.float64)
        eye_d = torch.eye(spec.head_dim, dtype=torch.float64)
        rots = fold_rotations(model, R1=eye_h, R2=[eye_d] * spec.n_layers)
        for n, p in model.named_parameters():
            assert torch.equal(p.data, snap[n]), f"{spec.name}: identity refold changed {n}"
        assert torch.equal(rots["R1"], eye_h), f"{spec.name}: identity R1 not returned"


# --------------------------------------------------------------------------------------
# 4. R.bin round-trip of fold-returned dicts
# --------------------------------------------------------------------------------------


def test_rbin_roundtrip_and_load_gates():
    """The dict fold_rotations returns torch.saves as an R.bin and round-trips bitwise
    through RotationSet.load (and through RotationSet.save). load refuses off-manifold
    matrices (both gross and drift-sized); orthogonalize=True polar-retracts a drifted
    file back onto the manifold (audit < 1e-10, staying near the drifted matrix), and the
    retracted set passes fold_rotations' external gate."""
    spec = _CONFIGS[4]  # qwen3, 2 layers
    rots = fold_rotations(_build(spec), mode="hadamard", seed=2, use_r2=True)
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        # fold-returned dict -> R.bin -> RotationSet: bitwise
        torch.save(dict(rots), path)
        rs = RotationSet.load(path)
        assert set(rs.rotations) == set(rots)
        for k in rots:
            assert torch.equal(rs.rotations[k], rots[k]), f"{k} changed in transit"
        assert torch.equal(rs.R1, rots["R1"])
        assert set(rs.R2) == _r2_keys(spec.n_layers)

        # RotationSet.save -> load: bitwise again
        rs.save(path)
        rs2 = RotationSet.load(path)
        for k in rots:
            assert torch.equal(rs2.rotations[k], rots[k]), f"{k} changed via rs.save"

        # gross off-manifold: refuse
        bad = dict(rots)
        bad["R1"] = rots["R1"] * 1.01
        torch.save(bad, path)
        with pytest.raises(ValueError, match="not orthogonal"):
            RotationSet.load(path)

        # drift-sized off-manifold (legacy raw R.bin style): refuse without orthogonalize
        torch.manual_seed(3)
        drifted = dict(rots)
        drifted["R1"] = rots["R1"] + 5e-4 * torch.randn_like(rots["R1"])
        torch.save(drifted, path)
        with pytest.raises(ValueError, match="not orthogonal"):
            RotationSet.load(path)

        # orthogonalize=True: polar retraction back onto the manifold, near the input
        rs3 = RotationSet.load(path, orthogonalize=True)
        assert max(rs3.ortho_audit().values()) < 1e-10
        assert (rs3.R1 - drifted["R1"]).abs().max().item() < 1e-2
        # and the retracted set is fold-deployable (passes the external ortho gate)
        fold_rotations(_build(spec), R1=rs3.R1, R2=rs3.R2)
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------------------
# 5. use_r2=False on every config; R2 argument forms are equivalent
# --------------------------------------------------------------------------------------


def test_use_r2_false_all_configs():
    """R1-only fold on every sweep config: returns exactly {'R1'}, keeps fp32 logits
    within tolerance, fuses norms to ones, and changes no parameter shape."""
    for spec in _CONFIGS:
        tag = f"{spec.name} use_r2=False"
        model = _build(spec)
        shapes_before = {n: tuple(p.shape) for n, p in model.named_parameters()}
        before = _logits(model)
        rots = fold_rotations(model, mode=spec.mode, seed=1, use_r2=False)
        after = _logits(model)
        assert set(rots) == {"R1"}, f"{tag}: keys {set(rots)}"
        max_diff = (after - before).abs().max().item()
        assert torch.allclose(after, before, rtol=0, atol=ATOL_FP32), (
            f"{tag}: max |delta logit| = {max_diff:.3e} > {ATOL_FP32}"
        )
        params_after = dict(model.named_parameters())
        for n, shp in shapes_before.items():
            assert tuple(params_after[n].shape) == shp, f"{tag}: shape of {n} changed"
        for layer in model.model.layers:
            assert torch.all(layer.input_layernorm.weight == 1), tag
            assert torch.all(layer.post_attention_layernorm.weight == 1), tag
        assert torch.all(model.model.norm.weight == 1), tag


def test_r2_list_dict_int_forms_equivalent():
    """R2 as a layer-ordered list, as a str-keyed dict (R.bin convention) and as an
    int-keyed dict produce bitwise-identical folds."""
    for spec in (_CONFIGS[1], _CONFIGS[7]):  # llama + qwen3, both multi-form
        rots = fold_rotations(_build(spec), mode=spec.mode, seed=4, use_r2=True)
        R1 = rots["R1"]
        r2_str = {k: v for k, v in rots.items() if k != "R1"}
        r2_list = [rots[f"model.layers.{i}.self_attn.R2"] for i in range(spec.n_layers)]
        r2_int = {i: rots[f"model.layers.{i}.self_attn.R2"] for i in range(spec.n_layers)}

        m_str, m_list, m_int = _build(spec), _build(spec), _build(spec)
        fold_rotations(m_str, R1=R1, R2=r2_str)
        fold_rotations(m_list, R1=R1, R2=r2_list)
        fold_rotations(m_int, R1=R1, R2=r2_int)
        p_str = dict(m_str.named_parameters())
        for other, form in ((m_list, "list"), (m_int, "int-dict")):
            for n, p in other.named_parameters():
                assert torch.equal(p.data, p_str[n].data), (
                    f"{spec.name}: {n} differs between str-dict and {form} R2 forms"
                )


# --------------------------------------------------------------------------------------
# 6. Orientation oracle: hand-built 1-layer model vs independent manual matmuls
# --------------------------------------------------------------------------------------


def test_orientation_oracle_manual_matmul():
    """Hand-built 1-layer Llama (hidden 64, 4 Q heads, 2 KV heads — GQA 2 — head_dim 32
    != hidden/heads): fold a copy, then verify the orientation algebra directly against
    independent fp64 manual matmuls (computed WITHOUT reusing fold's helper functions):

    weight-level — q/k/v readers carry (W * gamma_in) @ R1 (v additionally per-KV-head
    rows R2^T W_h), o_proj carries R1^T W with per-Q-head columns @ R2, gate/up carry
    (W * gamma_post) @ R1, down_proj carries R1^T W, embed carries E @ R1 (NO gamma),
    lm_head carries (W * gamma_final) @ R1;

    seam-level — for random x: readers reproduce the original fused projections from the
    rotated stream x@R1; v heads come out R2-rotated; pushing head-mixed values through
    the rotated o_proj lands the original output rotated by R1 (same for down_proj and
    lm_head); RMSNorm with unit gain commutes with R1 (the identity that makes the whole
    fold work); and the two actual models produce equal logits."""
    n_q, n_kv, d, hidden, inter, vocab = 4, 2, 32, 64, 96, 64
    n_rep = n_q // n_kv

    def build():
        torch.manual_seed(4321)
        cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=inter,
            num_hidden_layers=1,
            num_attention_heads=n_q,
            num_key_value_heads=n_kv,
            head_dim=d,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )
        model = LlamaForCausalLM(cfg).eval()
        _randomize_rmsnorm_gains(model)
        return model

    model_ref, model_rot = build(), build()

    def f64(t):
        return t.data.detach().to(torch.float64).clone()

    L = model_ref.model.layers[0]
    A, M = L.self_attn, L.mlp
    Wq, Wk, Wv, Wo = (
        f64(A.q_proj.weight),
        f64(A.k_proj.weight),
        f64(A.v_proj.weight),
        f64(A.o_proj.weight),
    )
    Wg, Wu, Wd = f64(M.gate_proj.weight), f64(M.up_proj.weight), f64(M.down_proj.weight)
    g_in = f64(L.input_layernorm.weight)
    g_post = f64(L.post_attention_layernorm.weight)
    g_fin = f64(model_ref.model.norm.weight)
    E, Wlm = f64(model_ref.model.embed_tokens.weight), f64(model_ref.lm_head.weight)
    assert A.q_proj.bias is None and A.o_proj.bias is None and M.down_proj.bias is None

    rots = fold_rotations(model_rot, mode="random", seed=11, use_r2=True)
    R1, R2 = rots["R1"], rots["model.layers.0.self_attn.R2"]
    Lr = model_rot.model.layers[0]
    Ar, Mr = Lr.self_attn, Lr.mlp
    Wq_r, Wk_r, Wv_r, Wo_r = (
        f64(Ar.q_proj.weight),
        f64(Ar.k_proj.weight),
        f64(Ar.v_proj.weight),
        f64(Ar.o_proj.weight),
    )
    Wg_r, Wu_r, Wd_r = f64(Mr.gate_proj.weight), f64(Mr.up_proj.weight), f64(Mr.down_proj.weight)
    E_r, Wlm_r = f64(model_rot.model.embed_tokens.weight), f64(model_rot.lm_head.weight)

    # -------- weight-level oracle (independent per-head loops, pure fp64) --------
    # Stored weights went through fp64 -> fp32 casts, expected values are pure fp64:
    # entries are O(0.1), cast noise is O(1e-8), so 1e-6 is tight yet safe; a wrong
    # orientation (e.g. R1 @ W instead of W @ R1, or R2 vs R2^T) is an O(0.1) error.
    ATOL_W = 1e-6

    def rot_v_rows(W):  # per-KV-head row blocks: W_h <- R2^T @ W_h
        out = W.clone()
        for h in range(n_kv):
            out[h * d : (h + 1) * d, :] = R2.T @ W[h * d : (h + 1) * d, :]
        return out

    def rot_o_cols(W):  # per-Q-head column blocks: W[:, h*d:(h+1)*d] @ R2
        out = W.clone()
        for h in range(n_q):
            out[:, h * d : (h + 1) * d] = W[:, h * d : (h + 1) * d] @ R2
        return out

    expected = {
        "q_proj": (Wq * g_in) @ R1,  # reader (+ input_layernorm gamma)
        "k_proj": (Wk * g_in) @ R1,  # reader (+ gamma)
        "v_proj": rot_v_rows((Wv * g_in) @ R1),  # reader (+ gamma) + R2 rows per KV head
        "o_proj": rot_o_cols(R1.T @ Wo),  # writer + R2 cols per Q head; NO gamma
        "gate_proj": (Wg * g_post) @ R1,  # reader (+ post_attention gamma)
        "up_proj": (Wu * g_post) @ R1,  # reader (+ gamma)
        "down_proj": R1.T @ Wd,  # writer; NO gamma
        "embed": E @ R1,  # residual writer; NO gamma
        "lm_head": (Wlm * g_fin) @ R1,  # reader (+ final-norm gamma)
    }
    got = {
        "q_proj": Wq_r,
        "k_proj": Wk_r,
        "v_proj": Wv_r,
        "o_proj": Wo_r,
        "gate_proj": Wg_r,
        "up_proj": Wu_r,
        "down_proj": Wd_r,
        "embed": E_r,
        "lm_head": Wlm_r,
    }
    for name in expected:
        diff = (got[name] - expected[name]).abs().max().item()
        assert diff < ATOL_W, f"weight oracle: {name} max |delta| = {diff:.3e} >= {ATOL_W}"
    assert torch.all(Lr.input_layernorm.weight == 1)
    assert torch.all(Lr.post_attention_layernorm.weight == 1)
    assert torch.all(model_rot.model.norm.weight == 1)

    # -------- seam-level oracle: manual matmuls of the tiny network on random x --------
    ATOL_X = 1e-5
    torch.manual_seed(7)
    x = torch.randn(5, hidden, dtype=torch.float64)  # residual-stream rows (orig frame)
    x_rot = x @ R1  # the SAME stream in the rotated frame

    # readers: rotated weights on the rotated stream reproduce the original projections
    for nm, W_fused, W_rot in (
        ("q_proj", Wq * g_in, Wq_r),
        ("k_proj", Wk * g_in, Wk_r),
        ("gate_proj", Wg * g_post, Wg_r),
        ("up_proj", Wu * g_post, Wu_r),
        ("lm_head", Wlm * g_fin, Wlm_r),
    ):
        ref, rot = x @ W_fused.T, x_rot @ W_rot.T
        diff = (rot - ref).abs().max().item()
        assert diff < ATOL_X, f"seam oracle reader {nm}: max |delta| = {diff:.3e}"

    # v_proj: heads come out rotated by R2 (per KV head)
    v_ref = x @ (Wv * g_in).T  # [5, n_kv*d]
    v_rot = x_rot @ Wv_r.T
    for h in range(n_kv):
        blk = slice(h * d, (h + 1) * d)
        diff = (v_rot[:, blk] - v_ref[:, blk] @ R2).abs().max().item()
        assert diff < ATOL_X, f"seam oracle v head {h}: max |delta| = {diff:.3e}"

    # o_proj: attention-mix per Q head (probs are identical in both models because the
    # q/k seams above are identities), GQA repeat kv->q via h // n_rep, then the rotated
    # o_proj must land the original output rotated by R1.
    P = torch.softmax(torch.randn(5, 5, dtype=torch.float64), dim=-1)  # stand-in probs
    a_ref = torch.cat(
        [P @ v_ref[:, (h // n_rep) * d : (h // n_rep + 1) * d] for h in range(n_q)], dim=-1
    )
    a_rot = torch.cat(
        [P @ v_rot[:, (h // n_rep) * d : (h // n_rep + 1) * d] for h in range(n_q)], dim=-1
    )
    diff = (a_rot @ Wo_r.T - (a_ref @ Wo.T) @ R1).abs().max().item()
    assert diff < ATOL_X, f"seam oracle o_proj: max |delta| = {diff:.3e}"

    # down_proj writer: rotated-frame output = original output @ R1
    m = torch.randn(5, inter, dtype=torch.float64)
    diff = (m @ Wd_r.T - (m @ Wd.T) @ R1).abs().max().item()
    assert diff < ATOL_X, f"seam oracle down_proj: max |delta| = {diff:.3e}"

    # embed rows land in the rotated frame
    diff = (E_r - E @ R1).abs().max().item()
    assert diff < ATOL_W, f"seam oracle embed: max |delta| = {diff:.3e}"

    # RMSNorm(unit gain) commutes with the orthogonal R1 — the identity the fold rests on
    def rms(v, eps=1e-6):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)

    diff = (rms(x_rot) - rms(x) @ R1).abs().max().item()
    assert diff < 1e-12, f"RMSNorm rotation-invariance broken: max |delta| = {diff:.3e}"

    # -------- end-to-end: run both actual models on the same ids --------
    ref_logits, rot_logits = _logits(model_ref, vocab), _logits(model_rot, vocab)
    max_diff = (rot_logits - ref_logits).abs().max().item()
    assert torch.allclose(rot_logits, ref_logits, rtol=0, atol=ATOL_FP32), (
        f"oracle end-to-end: max |delta logit| = {max_diff:.3e} > {ATOL_FP32}"
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
