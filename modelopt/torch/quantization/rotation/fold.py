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

"""Offline SpinQuant/QuaRot rotation folding (R1 + R2) for HF RMSNorm decoder LMs.

:meth:`fold_rotations` rewrites the weights of a supported HuggingFace causal LM in place so
that a global orthogonal rotation R1 of the residual stream (and optionally a per-layer
head-space rotation R2 on the v_proj -> o_proj path) is folded into the checkpoint. The
transform is a functional identity up to one float64 -> original-dtype round-trip per weight,
and is applied *before* quantization: rotated activation/weight distributions are flatter
(fewer outliers) and therefore easier to quantize (SpinQuant, QuaRot).

Only the offline rotations are applied here. SpinQuant's online transforms (R3 post-RoPE QK
rotation, R4 down_proj activation Hadamard) require runtime kernel support and are out of
scope — see README.md in this directory.
"""

import math
from typing import Any

import torch
import torch.nn as nn

__all__ = ["fold_rotations", "fold_seam_diags"]

# --------------------------------------------------------------------------------------
# Random orthogonal matrix generation (random-sign Hadamard D @ H / sqrt(n), or Haar QR)
# --------------------------------------------------------------------------------------

# SpinQuant's priority order of hard-coded had-K block sizes; the subset with q = K - 1
# prime and q ≡ 3 (mod 4) is regenerated via Paley construction I. Any valid Hadamard
# decomposition yields an orthonormal matrix, which is all that matters for fresh random
# rotations (bitwise parity with SpinQuant's hard-coded had-K constants is irrelevant).
_HADK_PRIORITY = [172, 156, 140, 108, 60, 52, 36, 28, 44, 40, 20, 12]
_HADK_SUPPORTED = {12, 20, 44, 60, 108, 140}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _paley_hadamard(K: int) -> torch.Tensor:
    """Hadamard matrix of order K = q + 1 via Paley construction I (q prime, q ≡ 3 mod 4)."""
    q = K - 1
    residues = {(i * i) % q for i in range(1, q)}

    def chi(a: int) -> float:
        a %= q
        return 0.0 if a == 0 else (1.0 if a in residues else -1.0)

    S = torch.zeros(K, K, dtype=torch.float64)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    for i in range(q):
        for j in range(q):
            if i != j:
                S[1 + i, 1 + j] = chi(i - j)
    H = S + torch.eye(K, dtype=torch.float64)
    assert torch.allclose(H @ H.T, K * torch.eye(K, dtype=torch.float64)), (
        f"Paley construction failed for K={K}"
    )
    return H


def _get_hadK(n: int) -> tuple[torch.Tensor | None, int]:
    """Return ``(hadK matrix or None, K)`` such that ``n = 2^k * K``."""
    if _is_pow2(n):
        return None, 1
    for K in _HADK_PRIORITY:
        if n % K == 0 and _is_pow2(n // K) and K in _HADK_SUPPORTED:
            return _paley_hadamard(K), K
    raise ValueError(
        f"size {n} is not 2^k or 2^k*K for a regenerable K {sorted(_HADK_SUPPORTED)}; "
        'use mode="random" instead'
    )


def _matmul_hadU(X: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dim of X.

    Recurses down to block size K, then one ``hadK @ blocks`` matmul, then ``/ sqrt(n)``
    (the normalization is applied exactly once).
    """
    n = X.shape[-1]
    hadK, K = _get_hadK(n)
    inp = X.clone().view(-1, n, 1)
    out = inp.clone()
    while inp.shape[1] > K:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        out = out.view(inp.shape)
        out[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        out[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        out = out.view(inp.shape[0], inp.shape[1], -1)
        inp, out = out, inp
    del out
    if K > 1:
        assert hadK is not None
        inp = hadK.view(1, K, K).to(inp) @ inp
    return inp.view(X.shape) / math.sqrt(n)


def _random_hadamard_matrix(size: int) -> torch.Tensor:
    """``D @ H / sqrt(n)``: random-sign Hadamard.

    Signs come from the global CPU torch RNG (seeded once in :meth:`fold_rotations`; the draw
    order is load-bearing for seed reproducibility).
    """
    signs = torch.randint(0, 2, (size,))
    Q = torch.diag(signs.to(torch.float64) * 2 - 1)
    return _matmul_hadU(Q)


def _random_orthogonal_matrix(size: int) -> torch.Tensor:
    """QR of a float64 randn with sign fix -> Haar-uniform orthogonal Q."""
    m = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(m)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def _get_orthogonal_matrix(size: int, mode: str) -> torch.Tensor:
    if mode == "hadamard":
        R = _random_hadamard_matrix(size)
    elif mode == "random":
        R = _random_orthogonal_matrix(size)
    else:
        raise ValueError(f"unknown rotation mode: {mode!r} (expected 'hadamard' or 'random')")
    err = (R @ R.T - torch.eye(size, dtype=torch.float64)).abs().max().item()
    assert err < 1e-10, f"generated matrix not orthogonal (max |R R^T - I| = {err:.3e})"
    return R


# Externally supplied matrices are gated at 1e-4 (vs 1e-10 for fresh draws). The gate
# checks the ``R R^T`` form because that is what the fold orientation consumes (the
# reader/writer seams compose to ``x R1 R1^T W^T``) — and for near-orthogonal trained
# matrices the max-entry residual of ``R R^T - I`` is 10-20x larger than the ``R^T R``
# form the trainer's step audit reports (basis-dependent; measured raw 150-step R1:
# 5e-5 vs 1e-3). learn_rotations closes this with a final polar retraction (~1e-14 both
# forms), so learned RotationSets pass with huge headroom; raw legacy R.bins must be
# loaded with RotationSet.load(..., orthogonalize=True) first.
_EXTERNAL_ORTHO_TOL = 1e-4


def _as_external_rotation(mat, size: int, name: str) -> torch.Tensor:
    """Validate one externally supplied rotation: shape [size, size], orthonormal.

    Orthonormality is gated at :data:`_EXTERNAL_ORTHO_TOL`. Returns a float64 CPU copy (all
    fold math is float64).
    """
    R = torch.as_tensor(mat).detach().to(torch.float64).cpu()
    if R.shape != (size, size):
        raise ValueError(f"{name}: expected shape {(size, size)}, got {tuple(R.shape)}")
    err = (R @ R.T - torch.eye(size, dtype=torch.float64)).abs().max().item()
    if err >= _EXTERNAL_ORTHO_TOL:
        raise ValueError(
            f"{name}: not orthogonal (max |R R^T - I| = {err:.3e} >= "
            f"{_EXTERNAL_ORTHO_TOL}) — refusing to fold a non-orthogonal rotation"
        )
    return R


def _normalize_external_r2(R2, n_layers: int, head_dim: int) -> list[torch.Tensor]:
    """Normalize the accepted per-layer R2 formats into a validated list ordered by layer.

    Accepts a sequence ordered by layer, a dict keyed by layer index, or a dict in the R.bin
    key convention (``model.layers.{i}.self_attn.R2`` — the format fold_rotations returns and
    RotationSet.R2 provides).
    """
    if isinstance(R2, dict):
        by_idx = {}
        for k, v in R2.items():
            if isinstance(k, int):
                idx = k
            else:
                try:
                    idx = int(str(k).split("model.layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    raise ValueError(
                        f"R2 dict key {k!r} not understood (want an int layer index or "
                        "'model.layers.{i}.self_attn.R2')"
                    ) from None
            by_idx[idx] = v
        if sorted(by_idx) != list(range(n_layers)):
            raise ValueError(
                f"R2 must cover every layer 0..{n_layers - 1}; got indices {sorted(by_idx)}"
            )
        mats = [by_idx[i] for i in range(n_layers)]
    else:
        mats = list(R2)
        if len(mats) != n_layers:
            raise ValueError(f"R2 has {len(mats)} matrices for {n_layers} layers")
    return [_as_external_rotation(m, head_dim, f"R2[layer {i}]") for i, m in enumerate(mats)]


# --------------------------------------------------------------------------------------
# Architecture mapping registry
# --------------------------------------------------------------------------------------


def _llama_head_dim(cfg) -> int:
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def _qwen3_head_dim(cfg) -> int:
    head_dim = getattr(cfg, "head_dim", None)
    assert head_dim is not None, (
        "config.head_dim missing — refusing to fall back to hidden_size // num_attention_heads "
        "(wrong for Qwen3-0.6B: that gives 64 but the true head_dim is 128)"
    )
    return head_dim


# RMSNorm -> downstream-linear fusion edges, relative to one decoder layer. The final-norm ->
# lm_head edge is handled explicitly in fold_rotations. Per-head q_norm/k_norm (Qwen3) act in
# post-q/k_proj head space, not on the residual stream: they are NEVER fused, NEVER rotated.
_NORM_EDGES = (
    ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
)

# Keyed on the model class name; every entry must follow the standard HF decoder layout
# (model.model.{embed_tokens,layers,norm} + model.lm_head).
_ARCH_REGISTRY: dict[str, dict[str, Any]] = {
    "LlamaForCausalLM": {
        "has_qk_norm": False,
        "head_dim": _llama_head_dim,
        "norm_edges": _NORM_EDGES,
    },
    "Qwen3ForCausalLM": {
        "has_qk_norm": True,
        "head_dim": _qwen3_head_dim,
        "norm_edges": _NORM_EDGES,
    },
}


# --------------------------------------------------------------------------------------
# Norm fusion and rotation application (all math in float64, cast back to original dtype)
# --------------------------------------------------------------------------------------


def _fuse_norm_into_linears(norm: nn.Module, linears: list[nn.Module]) -> None:
    """Fold the RMSNorm gain into the input columns of each downstream linear.

    ``W <- (W.double() * gamma.double()).to(orig_dtype)``; the norm weight becomes ones.
    RMSNorm only — a norm bias (LayerNorm) is not supported.
    """
    assert getattr(norm, "bias", None) is None, (
        "unexpected bias on norm — LayerNorm fusion is not ported (RMSNorm only)"
    )
    gamma = norm.weight.data.to(torch.float64)
    for lin in linears:
        w = lin.weight
        assert w.shape[1] == gamma.numel(), (
            f"fuse mismatch: W in_dim {w.shape[1]} vs gamma {gamma.numel()}"
        )
        w.data = (w.data.to(torch.float64) * gamma).to(w.dtype)
    norm.weight.data = torch.ones_like(norm.weight.data)


def _rotate_input_cols(module: nn.Module, R1: torch.Tensor, row_chunk: int = 32768) -> None:
    """Reader/embedding rotation: ``W <- W @ R1``.

    Rows are chunked to bound the float64 temporary for vocab-sized matrices. Bias is never
    touched (input-side rotation).
    """
    w = module.weight
    R = R1.to(w.device)
    for s in range(0, w.shape[0], row_chunk):
        w.data[s : s + row_chunk] = (w.data[s : s + row_chunk].to(torch.float64) @ R).to(w.dtype)


def _rotate_output_rows(linear: nn.Module, R1: torch.Tensor) -> None:
    """Writer rotation (o_proj / down_proj): ``W <- R1^T @ W``; bias ``b <- R1^T b``."""
    w = linear.weight
    R = R1.to(w.device)
    w.data = (R.T @ w.data.to(torch.float64)).to(w.dtype)
    if linear.bias is not None:
        b = linear.bias
        b.data = (R.T @ b.data.to(torch.float64)).to(b.dtype)


def _rotate_v_proj_r2(linear: nn.Module, R2: torch.Tensor, head_dim: int) -> None:
    """Per-KV-head block of rows: ``W_h <- R2^T @ W_h``.

    Implemented as transpose, reshape to ``[in, out//d, d]``, right-multiply by R2, then
    reshape/transpose back.
    """
    assert linear.bias is None, (
        "v_proj has a bias — R2 would silently break equivalence (bias must be rotated per head)"
    )
    w = linear.weight
    out_f, in_f = w.shape
    assert out_f % head_dim == 0, f"v_proj out {out_f} not divisible by head_dim {head_dim}"
    R = R2.to(w.device)
    Wt = w.data.to(torch.float64).t()
    Wt = (Wt.reshape(in_f, out_f // head_dim, head_dim) @ R).reshape(in_f, out_f)
    w.data = Wt.t().contiguous().to(w.dtype)


def _rotate_o_proj_r2(linear: nn.Module, R2: torch.Tensor, head_dim: int) -> None:
    """Per-Q-head block of columns: ``W[:, h*d:(h+1)*d] <- W[:, h*d:(h+1)*d] @ R2``.

    The block count derives from the tensor shape (in_features = num_q_heads * head_dim), so
    GQA is handled automatically. Input-side w.r.t. v: bias untouched.
    """
    w = linear.weight
    out_f, in_f = w.shape
    assert in_f % head_dim == 0, f"o_proj in {in_f} not divisible by head_dim {head_dim}"
    R = R2.to(w.device)
    W = w.data.to(torch.float64).reshape(out_f, in_f // head_dim, head_dim) @ R
    w.data = W.reshape(out_f, in_f).to(w.dtype)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def fold_rotations(
    model: nn.Module,
    mode: str = "hadamard",
    seed: int = 0,
    use_r2: bool = True,
    R1: torch.Tensor | None = None,
    R2=None,
) -> dict[str, torch.Tensor]:
    """Fold offline SpinQuant/QuaRot rotations (R1 + per-layer R2) into ``model`` in place.

    Pipeline (order is load-bearing): untie tied embeddings with a real clone -> seed the
    global torch RNG -> fuse RMSNorm gains into downstream linears -> apply R1 (readers
    ``W @ R1``, writers ``R1^T @ W``; embed_tokens and lm_head included) -> apply one shared
    per-layer R2 on the v_proj -> o_proj head space. All rotation math runs in float64 and
    is cast back to the original weight dtype, so the model output is unchanged up to that
    round-trip.

    Rotation source — two mutually exclusive paths:

    * **Seed path (default)**: ``R1``/``R2`` omitted; matrices are drawn from the seeded
      global RNG per ``mode`` (byte-identical for equal seeds; unchanged legacy behavior).
    * **External path**: pass ``R1`` (and ``R2`` when ``use_r2=True``) explicitly — e.g.
      matrices learned by :meth:`~modelopt.torch.quantization.rotation.learn_rotations` —
      and they are folded through this same validated path. External matrices are gated at
      orthonormality :data:`_EXTERNAL_ORTHO_TOL` (1e-4, the trained-rotation deployability
      tolerance) instead of the 1e-10 fresh-draw gate; ``mode``/``seed`` are ignored.

    Args:
        model: HuggingFace causal LM whose class is registered (currently
            ``LlamaForCausalLM`` and ``Qwen3ForCausalLM``). Modified in place;
            ``config.tie_word_embeddings`` is set to False.
        mode: ``"hadamard"`` (random-sign Hadamard, ``D @ H / sqrt(n)``) or ``"random"``
            (Haar-uniform orthogonal via QR). Seed path only.
        seed: Seed for the global CPU torch RNG; all rotation matrices draw from it in a
            fixed order, so equal seeds give byte-identical rotations. Seed path only.
        use_r2: Also apply the per-layer head-space rotation R2 (one shared matrix per
            layer across all heads — required for GQA / repeat_kv correctness).
        R1: Optional external global rotation ``[hidden, hidden]``.
        R2: Optional external per-layer head-space rotations: a sequence ordered by layer,
            a dict keyed by layer index, or a dict in the returned key convention
            (``model.layers.{i}.self_attn.R2``). Requires ``R1`` and ``use_r2=True``.

    Returns:
        The applied rotations as float64 CPU tensors, keyed ``"R1"`` plus
        ``"model.layers.{i}.self_attn.R2"`` (SpinQuant optimized-checkpoint convention).

    Raises:
        NotImplementedError: If the model architecture is not in the registry.
        ValueError: If external matrices are inconsistent (shape/count/orthogonality, R2
            without R1, or a missing R2 with ``use_r2=True``).
    """
    arch = type(model).__name__
    if arch not in _ARCH_REGISTRY:
        raise NotImplementedError(
            f"fold_rotations: unsupported architecture {arch!r}; "
            f"supported: {sorted(_ARCH_REGISTRY)}"
        )
    spec = _ARCH_REGISTRY[arch]

    decoder = model.model
    layers = decoder.layers
    embed = decoder.embed_tokens
    head_dim = spec["head_dim"](model.config)

    # External-matrix path validation (learned rotations enter here).
    external = R1 is not None or R2 is not None
    r1_ext = r2_ext = None
    if external:
        if R1 is None:
            raise ValueError("external rotations: R1 is required when R2 is given")
        if use_r2 and R2 is None:
            raise ValueError(
                "external rotations: use_r2=True needs R2 matrices (or pass use_r2=False)"
            )
        if not use_r2 and R2 is not None:
            raise ValueError("external rotations: R2 given but use_r2=False")
        r1_ext = _as_external_rotation(R1, model.config.hidden_size, "R1")
        if use_r2:
            r2_ext = _normalize_external_r2(R2, len(layers), head_dim)

    # 1. Untie embeddings with a real clone: lm_head (reader) and embed_tokens (writer)
    # diverge below because only lm_head absorbs the final-norm gain.
    if model.lm_head.weight.data_ptr() == embed.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(
            embed.weight.data.clone(), requires_grad=embed.weight.requires_grad
        )
    model.config.tie_word_embeddings = False
    assert model.lm_head.weight.data_ptr() != embed.weight.data_ptr(), "untie failed"

    # Snapshots for post-condition checks (after untie: named_parameters() deduplicates a
    # tied lm_head.weight, so the snapshot would otherwise miss it).
    shapes_before = {n: tuple(p.shape) for n, p in model.named_parameters()}
    qk_norm_before = {}
    if spec["has_qk_norm"]:
        for idx, layer in enumerate(layers):
            qk_norm_before[f"{idx}.q_norm"] = layer.self_attn.q_norm.weight.data.clone()
            qk_norm_before[f"{idx}.k_norm"] = layer.self_attn.k_norm.weight.data.clone()

    # 2. Seed the global CPU RNG (every rotation matrix draws from it, in a fixed order).
    # External path: nothing is drawn, so the global RNG state is left untouched.
    if not external:
        torch.manual_seed(seed)

    # 3. Fuse RMSNorm gains into downstream linears (fused norms become exactly ones).
    for layer in layers:
        for norm_name, linear_names in spec["norm_edges"]:
            _fuse_norm_into_linears(
                layer.get_submodule(norm_name), [layer.get_submodule(n) for n in linear_names]
            )
    _fuse_norm_into_linears(decoder.norm, [model.lm_head])

    # 4. Rotate: R1 over the residual stream, then per-layer R2 on the v -> o head space.
    R1 = r1_ext if external else _get_orthogonal_matrix(model.config.hidden_size, mode)
    assert R1 is not None
    rotations = {"R1": R1}
    _rotate_input_cols(embed, R1)  # writer: rows e <- e @ R1
    _rotate_input_cols(model.lm_head, R1)  # reader: W <- W @ R1 (untied above)
    for idx, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp
        if use_r2:
            if external:
                assert r2_ext is not None
                R2 = r2_ext[idx]
            else:
                R2 = _get_orthogonal_matrix(head_dim, mode)
        else:
            R2 = None
        _rotate_input_cols(attn.q_proj, R1)
        _rotate_input_cols(attn.k_proj, R1)
        _rotate_input_cols(attn.v_proj, R1)
        _rotate_output_rows(attn.o_proj, R1)
        _rotate_input_cols(mlp.gate_proj, R1)
        _rotate_input_cols(mlp.up_proj, R1)
        _rotate_output_rows(mlp.down_proj, R1)
        # SpinQuant additionally folds the weight half of the ONLINE R4 activation Hadamard
        # into down_proj; without the matching runtime activation transform that destroys
        # the model, so it is deliberately skipped in this offline-only transform.
        if use_r2:
            _rotate_v_proj_r2(attn.v_proj, R2, head_dim)
            _rotate_o_proj_r2(attn.o_proj, R2, head_dim)
            rotations[f"model.layers.{idx}.self_attn.R2"] = R2

    # 5. Post-conditions: the invariants that make the transform an identity.
    for idx, layer in enumerate(layers):
        for norm_name, _ in spec["norm_edges"]:
            assert torch.all(layer.get_submodule(norm_name).weight.data == 1), (
                f"layer {idx} {norm_name} not fused to ones"
            )
        if spec["has_qk_norm"]:
            assert torch.equal(
                layer.self_attn.q_norm.weight.data, qk_norm_before[f"{idx}.q_norm"]
            ), f"q_norm[{idx}] changed"
            assert torch.equal(
                layer.self_attn.k_norm.weight.data, qk_norm_before[f"{idx}.k_norm"]
            ), f"k_norm[{idx}] changed"
    assert torch.all(decoder.norm.weight.data == 1), "final norm not fused to ones"
    for n, p in model.named_parameters():
        assert tuple(p.shape) == shapes_before[n], f"shape of {n} changed"

    return {k: v.cpu() for k, v in rotations.items()}


def fold_seam_diags(model: nn.Module, seam_diags, smax: float = 256.0) -> dict:
    """Bake learned per-input-channel seam scales into ``model`` in place (fp64 math).

    The transform-QAT counterpart of :meth:`fold_rotations` for the diagonal half of the
    learned reparametrization (``RotationSet.seam_diags``): exactly the two
    ROTATION-SURVIVING SmoothQuant seams of the T14 prefold, as exact per-seam
    functional identities —

    - **down seam** (``s_down [intermediate_size]``): ``up_proj`` rows ``/= s_down``
      (SwiGLU is elementwise: ``silu(g) * (u/s) == (silu(g)*u)/s``), ``down_proj``
      cols ``*= s_down``.
    - **o seam** (``s_o [n_kv_heads*head_dim]``, GQA-exact — a per-channel scale
      commutes with the per-head convex attention mix, shared across the q-heads of one
      kv head): ``v_proj`` rows ``/= s_o``, ``o_proj`` cols ``*= s_o`` expanded per
      q-head group.

    Row-scaled linears (up/v) have any bias divided too; col-scaled ones (down/o) are
    input-side, bias untouched. All math is float64, cast back to each weight's dtype —
    the model function is unchanged up to that round-trip. Composes with
    :meth:`fold_rotations` in EITHER order (each fold is an identity); applying
    ``fold_seam_diags`` FIRST reproduces the learner's effective-weight assembly
    (prefold-inside, rotation-outside) entry-for-entry.

    Args:
        model: Registered HF causal LM (``LlamaForCausalLM``/``Qwen3ForCausalLM``),
            modified in place.
        seam_diags: Mapping layer index -> ``{"down": s_down, "o": s_o}`` with strictly
            positive scale vectors (the :attr:`RotationSet.seam_diags` format; keys may
            be int or int-like str). May cover a subset of layers — each layer's seams
            are independent identities.
        smax: fp16-safety ceiling: scales are clamped to ``[1e-4, smax]`` before folding
            (T14 convention; default 256 — the fp16-endpoint-safe value from the T15
            activation-underflow finding, vs. 1e4 for bf16-only paths). A clamp that
            actually bites trades exactness for numeric safety and is reported in the
            returned evidence.

    Returns:
        Evidence dict: ``{"smax": ..., "layers": {i: {"down_s_max", "down_s_spread",
        "o_s_max", "o_s_spread", "clamped"}}}``.
    """
    arch = type(model).__name__
    if arch not in _ARCH_REGISTRY:
        raise NotImplementedError(
            f"fold_seam_diags: unsupported architecture {arch!r}; "
            f"supported: {sorted(_ARCH_REGISTRY)}"
        )
    head_dim = _ARCH_REGISTRY[arch]["head_dim"](model.config)
    layers = model.model.layers

    by_idx: dict[int, dict] = {}
    for k, pair in seam_diags.items():
        idx = int(k)
        if not 0 <= idx < len(layers):
            raise ValueError(f"seam_diags layer index {idx} out of range 0..{len(layers) - 1}")
        if set(pair) != {"down", "o"}:
            raise ValueError(f"seam_diags[{k!r}]: expected keys {{'down', 'o'}}, got {set(pair)}")
        by_idx[idx] = pair

    def _prep(vec, dim: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate one scale vector; return ``(clamped fp64 CPU scales, raw scales)``.

        Telemetry reports the RAW stats so a biting clamp is visible.
        """
        s = torch.as_tensor(vec).detach().to(torch.float64).cpu().flatten()
        if s.numel() != dim:
            raise ValueError(f"{name}: expected {dim} scales, got {s.numel()}")
        if not bool((s > 0).all()):
            raise ValueError(f"{name}: scales must be strictly positive")
        return s.clamp(1e-4, smax), s

    evidence: dict = {"smax": smax, "layers": {}}
    for idx in sorted(by_idx):
        layer = layers[idx]
        up, down = layer.mlp.up_proj.weight, layer.mlp.down_proj.weight
        v, o = layer.self_attn.v_proj.weight, layer.self_attn.o_proj.weight

        s_d, s_d_raw = _prep(by_idx[idx]["down"], down.shape[1], f"seam_diags[{idx}]['down']")
        assert up.shape[0] == s_d.numel(), (
            f"layer {idx}: up_proj out {up.shape[0]} != down_proj in {s_d.numel()}"
        )
        s_o, s_o_raw = _prep(by_idx[idx]["o"], v.shape[0], f"seam_diags[{idx}]['o']")
        assert s_o.numel() % head_dim == 0, (
            f"layer {idx}: o-seam scale dim {s_o.numel()} not divisible by head_dim {head_dim}"
        )
        assert o.shape[1] % s_o.numel() == 0, (
            f"layer {idx}: o_proj in {o.shape[1]} not a multiple of v_proj out {s_o.numel()}"
        )
        n_kv = s_o.numel() // head_dim
        group = o.shape[1] // s_o.numel()  # q-heads per kv head (GQA)

        # down seam: up rows / s_d (bias too — row scaling), down cols * s_d.
        s_dd = s_d.to(up.device)
        up.data = (up.data.to(torch.float64) / s_dd[:, None]).to(up.dtype)
        if layer.mlp.up_proj.bias is not None:
            b = layer.mlp.up_proj.bias
            b.data = (b.data.to(torch.float64) / s_dd).to(b.dtype)
        down.data = (down.data.to(torch.float64) * s_dd[None, :]).to(down.dtype)

        # o seam: v rows / s_o (bias too), o cols * s_o expanded per q-head group.
        s_oo = s_o.to(v.device)
        v.data = (v.data.to(torch.float64) / s_oo[:, None]).to(v.dtype)
        if layer.self_attn.v_proj.bias is not None:
            b = layer.self_attn.v_proj.bias
            b.data = (b.data.to(torch.float64) / s_oo).to(b.dtype)
        s_full = s_oo.reshape(n_kv, 1, head_dim).expand(n_kv, group, head_dim).reshape(-1)
        o.data = (o.data.to(torch.float64) * s_full[None, :].to(o.device)).to(o.dtype)

        evidence["layers"][idx] = {
            "down_s_max": s_d_raw.max().item(),
            "down_s_spread": (s_d_raw.max() / s_d_raw.min()).item(),
            "o_s_max": s_o_raw.max().item(),
            "o_s_spread": (s_o_raw.max() / s_o_raw.min()).item(),
            "clamped": bool((s_d != s_d_raw).any()) or bool((s_o != s_o_raw).any()),
        }
    return evidence
