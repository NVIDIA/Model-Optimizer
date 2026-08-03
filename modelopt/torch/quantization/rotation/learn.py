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

"""Learned SpinQuant rotations (R1 + per-layer R2) via Cayley SGD on the Stiefel manifold.

:meth:`learn_rotations` optimizes the same offline rotations that :meth:`fold_rotations`
folds — a global residual-stream rotation R1 ``[hidden, hidden]`` and one per-layer
head-space rotation R2 ``[head_dim, head_dim]`` on the v_proj -> o_proj path — by minimizing
the next-token cross-entropy of the *fake-quantized* rotated model on calibration text
(SpinQuant, https://arxiv.org/abs/2405.16406). The rotation parameters live on the Stiefel
manifold and are updated with a Cayley-transform SGD (the SGDG optimizer of Li et al.,
ported self-contained in :mod:`.sgdg`), so every iterate stays orthogonal to numerical tolerance and
the result folds through :meth:`fold_rotations`'s validated path via its ``R1=`` / ``R2=``
arguments.

Design notes, hyperparameter lineage (official SpinQuant vs. our internal reference trainer vs. this
module) and the objective-config table live in README.md next to this file. Architecture
knowledge (norm-fusion edges, reader/writer orientation, Qwen3 q/k_norm exclusion,
head_dim resolution, tied-embedding handling) is REUSED from ``fold.py`` — it is defined
exactly once, in the fold module's ``_ARCH_REGISTRY``.
"""

import math
import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .fold import _ARCH_REGISTRY, _fuse_norm_into_linears, _get_orthogonal_matrix
from .sgdg import SGDG

__all__ = [
    "INT8_DEFAULT_OBJECTIVE",
    "SEAM_DIAG_LR",
    "SGDG",
    "W4A4_G128_OBJECTIVE",
    "W16A4_ASYM_R4G_OBJECTIVE",
    "QuantObjective",
    "RotationSet",
    "learn_rotations",
]

# Deployability tolerance for rotation matrices entering a fold. Trained (fp32 Cayley)
# iterates drift off the manifold, and the max-entry residual is BASIS-DEPENDENT:
# ``R^T R - I`` and ``R R^T - I`` share eigenvalues but not entries — measured on real
# 150-step runs the R^T R form sits at ~5e-5 while the R R^T form (the one the fold
# orientation actually consumes: reader/writer seams compose to ``x R1 R1^T W^T``) is
# 10-20x larger (~1e-3). learn_rotations therefore applies a FINAL RETRACTION (polar
# projection to the nearest orthogonal matrix, ~1e-14 residual both forms) before
# returning, so its outputs pass this gate with orders-of-magnitude headroom; the 1e-4
# tolerance exists for hand-supplied matrices. Raw legacy R.bins (no retraction) need
# ``RotationSet.load(path, orthogonalize=True)``.
LEARNED_ORTHO_TOL = 1e-4

#: Peak Adam learning rate for the seam-diagonal parameters (``log s``) when
#: ``QuantObjective.learn_seam_diag`` is on. The diagonals live in a SEPARATE plain-Adam
#: group (never in the SGDG stiefel group — they are not on the manifold) and follow the
#: same cosine schedule as the rotation lr.
SEAM_DIAG_LR = 1e-2

#: Key under which :attr:`RotationSet.seam_diags` is stored inside a saved rotation
#: file. Absent from old-format (pure-R.bin) files, so those load with
#: ``seam_diags=None`` unchanged; a flat R.bin consumer must pop/ignore this key.
_SEAM_DIAGS_KEY = "__seam_diags__"

# The per-layer linears whose (rotated) weights the objective fake-quantizes and whose
# inputs the activation quantizer sees. Matches ModelOpt PTQ coverage for decoder LMs:
# embeddings and lm_head are never quantized (INT8_DEFAULT_CFG disables ``*lm_head*``).
_ATTN_PROJS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
_MLP_PROJS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


# --------------------------------------------------------------------------------------
# Fake-quant objective configuration (pluggable; STE numerics follow ModelOpt convention)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantObjective:
    """Fake-quantization spec for the training objective.

    Symmetric integer quant-dequant with ModelOpt max-calibration numerics: for ``b`` bits
    the scale is ``amax / (2**(b-1) - 1)`` (clamped to 1e-12), values are rounded
    half-to-even and clamped to ``[-2**(b-1), 2**(b-1) - 1]`` (e.g. int4 -> ``amax/7``,
    ``[-8, 7]``; int8 -> ``amax/127``, ``[-128, 127]``). Gradients pass straight through.

    Args:
        name: Tag recorded in :attr:`RotationSet.meta`.
        w_bits: Weight bit-width, or None to disable weight fake-quant.
        w_group: Weight quant group size along in_features (per-group), or None for
            per-output-channel scales.
        a_bits: Activation bit-width, or None to disable activation fake-quant.
        a_mode: ``"per_token_dynamic"`` (scale from each token's amax, recomputed every
            forward — SpinQuant/QuaRot W4A4 regime) or ``"per_tensor_static"`` (one scalar
            scale per linear — the ModelOpt INT8_DEFAULT_CFG ``algorithm="max"`` axis).
        a_static_scope: How the per-tensor-static scale surrogate is formed during
            training (deployment always recalibrates on the FINAL folded model):
            ``"batch"`` (default) — fresh amax per calibration batch, i.e. the scale
            tracks the current rotation exactly as post-hoc max calibration would; the
            stationary surrogate for the deployed endpoint. ``"run"`` — monotone running
            max over all batches seen (literal max-calibration semantics over the calib
            stream) — measured to be NON-STATIONARY under a moving rotation: past
            rotations' outliers keep the scale inflated, later steps train against grids
            deployment never uses, and the loss can drift UP (observed on Qwen3-1.7B:
            3.75 -> 4.87 over 150 steps). Kept for ablation.
        learn_seam_diag: OSTQuant-style transform-QAT — additionally learn per-input-
            channel diagonal scales at the two ROTATION-SURVIVING seams (down_proj input
            and o_proj input; at every norm-fed seam a folded diagonal is cancelled by
            norm fusion, so those seams have no surviving degree of freedom). The scales
            are parametrized as ``log s`` (init 0 = identity, positivity for free) and
            applied in the effective-weight assembly exactly like the T14 SmoothQuant
            prefold: ``up_proj`` rows ``/ s_down`` + ``down_proj`` cols ``* s_down``,
            and ``v_proj`` rows ``/ s_o`` (KV dim) + ``o_proj`` cols ``* s_o`` expanded
            per q-head group (GQA-exact) — a functional identity for ANY positive
            diagonal, trained jointly with R1/R2 through the same STE objective but in a
            separate plain-Adam group (:data:`SEAM_DIAG_LR`, same cosine schedule).
            Default False: bitwise-identical behavior to the rotation-only trainer.
        a_asym: Per-token dynamic ASYMMETRIC min-max affine activation fake-quant
            instead of the symmetric default — the official SpinQuant activation
            quantizer (paper A.4: "asymmetric quantization outperforms symmetric ...
            no clipping"). Numerics match the official ``ActQuantizer`` (``sym=False``,
            ``clip_ratio=1``) exactly: zero-inclusive token range, all-zero-token
            fallback to ``[-1, 1]``, ``scale = (max - min)/(2**b - 1)``,
            ``zp = round(-min/scale)``. Only implemented for
            ``a_mode="per_token_dynamic"``.
        r4_in_graph: Put the online R4 down_proj Hadamard into the TRAINING graph only
            (input hook ``x @ H`` before activation fake-quant + effective down_proj
            weight columns ``@ H`` — a functional-identity pair). The official trainer
            does this unconditionally, even for no-had deployment
            (``train_utils/main.py``); T12's arm4 measured that training WITHOUT it
            (deployment-faithful) makes the no-had result worse. The deployed model
            folds R1/R2 only — no online op survives in the returned
            :class:`RotationSet` or the fold. Power-of-2 seam dimension only.
    """

    name: str
    w_bits: int | None = 4
    w_group: int | None = 128
    a_bits: int | None = None
    a_mode: str = "per_token_dynamic"
    a_static_scope: str = "batch"
    learn_seam_diag: bool = False
    a_asym: bool = False
    r4_in_graph: bool = False

    def __post_init__(self):
        if self.a_bits is not None and self.a_mode not in (
            "per_token_dynamic",
            "per_tensor_static",
        ):
            raise ValueError(f"unknown a_mode: {self.a_mode!r}")
        if self.a_static_scope not in ("batch", "run"):
            raise ValueError(f"unknown a_static_scope: {self.a_static_scope!r}")
        if self.a_asym and self.a_bits is None:
            raise ValueError("a_asym=True requires a_bits")
        if self.a_asym and self.a_mode != "per_token_dynamic":
            raise ValueError("a_asym is only implemented for a_mode='per_token_dynamic'")


#: SpinQuant-paper-style W4A4 (per-group-128 sym weights + per-token dynamic sym int4
#: activations). Comparison point for the internal reference runs / the paper's W4A4 rows.
W4A4_G128_OBJECTIVE = QuantObjective(
    name="w4a4_g128", w_bits=4, w_group=128, a_bits=4, a_mode="per_token_dynamic"
)

#: The axes of ModelOpt's INT8_DEFAULT_CFG: per-output-channel sym int8 weights +
#: per-tensor STATIC sym int8 activations ("max" calibration), lm_head excluded. This is
#: the deployment target where random rotations barely help — the static per-tensor
#: activation scale is the collapse axis the learned rotation trains against.
#: ``a_static_scope="batch"``: the scale tracks the current rotation (see QuantObjective).
INT8_DEFAULT_OBJECTIVE = QuantObjective(
    name="int8_default", w_bits=8, w_group=None, a_bits=8, a_mode="per_tensor_static"
)

#: The official trainer's objective for GPTQ-deployed rows ("Cayley on 16-4-KV", paper
#: Table 3): weights stay 16-bit in the loss, A4 per-token dynamic ASYM min-max (paper
#: A.4), and the online R4 down_proj Hadamard lives in the TRAINING graph only — the
#: official code keeps it there even when deploying no-had, and T12's arm4 measured that
#: removing it (deployment-faithful training) makes the no-had result WORSE. The deployed
#: model still folds R1/R2 only.
W16A4_ASYM_R4G_OBJECTIVE = QuantObjective(
    name="w16a4_asym_r4g",
    w_bits=None,
    w_group=None,
    a_bits=4,
    a_mode="per_token_dynamic",
    a_asym=True,
    r4_in_graph=True,
)


class _FakeQuantSTE(torch.autograd.Function):
    """Symmetric quant-dequant with a precomputed scale; straight-through backward."""

    @staticmethod
    def forward(ctx, x, scale, qneg, qpos):
        return (torch.round(x / scale).clamp_(qneg, qpos)) * scale

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None, None


def _fq_weight(w: torch.Tensor, cfg: QuantObjective) -> torch.Tensor:
    """Fake-quantize a ``[out, in]`` weight per ``cfg`` (per-group or per-out-channel)."""
    qpos = float(2 ** (cfg.w_bits - 1) - 1)
    qneg = -qpos - 1.0
    if cfg.w_group is None:
        s = (w.abs().amax(dim=1, keepdim=True) / qpos).clamp_min(1e-12)
        return _FakeQuantSTE.apply(w, s, qneg, qpos)
    out_f, in_f = w.shape
    g = cfg.w_group
    wg = w.reshape(out_f, in_f // g, g)
    s = (wg.abs().amax(dim=-1, keepdim=True) / qpos).clamp_min(1e-12)
    return _FakeQuantSTE.apply(wg, s, qneg, qpos).reshape(out_f, in_f)


def _fq_act(x: torch.Tensor, s: torch.Tensor, bits: int) -> torch.Tensor:
    qpos = float(2 ** (bits - 1) - 1)
    q = (torch.round(x / s).clamp_(-qpos - 1.0, qpos)) * s
    return x + (q - x).detach()  # STE


def _fq_act_asym(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token dynamic ASYM min-max affine fake-quant (STE backward).

    Bit-for-bit the official ``ActQuantizer`` (``sym=False``, ``clip_ratio=1``) recipe:
    zero-inclusive token range, all-zero tokens fall back to ``[-1, 1]``,
    ``scale = (xmax - xmin)/(2**bits - 1)``, ``zp = round(-xmin/scale)``,
    ``q = clamp(round(x/scale) + zp, 0, 2**bits - 1)``, dequant ``scale * (q - zp)``.
    """
    qmax = float(2**bits - 1)
    xd = x.detach()
    xmin = xd.amin(dim=-1, keepdim=True).clamp_max_(0.0)
    xmax = xd.amax(dim=-1, keepdim=True).clamp_min_(0.0)
    degen = (xmin == 0) & (xmax == 0)
    xmin = torch.where(degen, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(degen, torch.full_like(xmax, 1.0), xmax)
    s = (xmax - xmin) / qmax
    zp = torch.round(-xmin / s)
    q = (torch.round(xd / s) + zp).clamp_(0.0, qmax)
    dq = (q - zp) * s
    return x + (dq - x).detach()  # STE


def _walsh_hadamard(n: int, device=None) -> torch.Tensor:
    """Normalized Sylvester Walsh-Hadamard matrix ``[n, n]`` (fp32): symmetric, ``H Hᵀ = I``.

    The fixed (not random-signed) Hadamard the official online-R4 op applies
    (``matmul_hadU``). Power-of-2 sizes only — the R4 seam dimension is
    ``config.intermediate_size`` (Llama-3.2-1B: 8192 = 2^13); non-power-of-2 seams
    (e.g. Qwen3's 6144 = 3·2048) need the had-K Kronecker composition, not implemented.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise NotImplementedError(
            f"r4_in_graph: seam dimension {n} is not a power of 2; the had-K Kronecker "
            "composition is not implemented"
        )
    H = torch.ones(1, 1, dtype=torch.float32, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


class _ActQuantHooks:
    """Forward-pre-hooks applying STE activation fake-quant on the target linears.

    ``per_token_dynamic``: scale = token amax / qpos, recomputed each forward.
    ``per_tensor_static``: one scalar scale per linear. Scope ``"batch"``: fresh amax per
    batch — the scale tracks the current rotation, the stationary surrogate for post-hoc
    max calibration of the folded model. Scope ``"run"``: monotone running max over every
    batch seen (literal calib-stream max semantics; non-stationary under a moving
    rotation — see QuantObjective). ``static_amax`` telemetry records the observed
    per-linear maximum either way.

    ``r4_had`` (``QuantObjective.r4_in_graph``): the mlp.down_proj hook additionally
    applies ``x @ H`` BEFORE its activation fake-quant — the online half of the
    training-graph R4 whose weight half lives in the assembly. Pass H pre-cast to the
    model dtype. With ``a_bits=None`` (W16-act-only variants keep a_bits set; this is
    the r4-only corner) the hooks attach to down_proj alone and just rotate.
    """

    def __init__(self, cfg: QuantObjective, r4_had: torch.Tensor | None = None):
        self.cfg = cfg
        self.r4_had = r4_had
        self.handles: list = []
        self.static_amax: dict[str, torch.Tensor] = {}

    def _hook(self, name: str):
        cfg = self.cfg
        r4h = self.r4_had if name.endswith("mlp.down_proj") else None
        qpos = None if cfg.a_bits is None else float(2 ** (cfg.a_bits - 1) - 1)

        def hook(module, inputs):
            x = inputs[0]
            if r4h is not None:
                x = x @ (r4h if r4h.dtype == x.dtype else r4h.to(x.dtype))
            if cfg.a_bits is None:
                return (x, *inputs[1:])
            if cfg.a_mode == "per_token_dynamic":
                if cfg.a_asym:
                    return (_fq_act_asym(x, cfg.a_bits), *inputs[1:])
                s = (x.detach().abs().amax(dim=-1, keepdim=True) / qpos).clamp_min(1e-12)
            else:  # per_tensor_static
                batch_amax = x.detach().abs().amax()
                prev = self.static_amax.get(name)
                run_max = batch_amax if prev is None else torch.maximum(prev, batch_amax)
                self.static_amax[name] = run_max  # telemetry: observed max either way
                amax = batch_amax if cfg.a_static_scope == "batch" else run_max
                s = (amax / qpos).clamp_min(1e-12)
            return (_fq_act(x, s, cfg.a_bits), *inputs[1:])

        return hook

    def attach(self, model: nn.Module) -> int:
        targets = (
            _ATTN_PROJS + _MLP_PROJS
            if self.cfg.a_bits is not None
            else ("mlp.down_proj",)  # r4-only: nothing to quantize elsewhere
        )
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ".layers." in name and name.endswith(targets):
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))
        return len(self.handles)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --------------------------------------------------------------------------------------
# Final retraction — nearest orthogonal matrix (the SGDG optimizer itself lives in .sgdg)
# --------------------------------------------------------------------------------------


def _polar_project(R: torch.Tensor) -> torch.Tensor:
    """Nearest orthogonal matrix in Frobenius norm: ``R = U S V^T -> U V^T`` (float64).

    Used as the FINAL retraction on trained matrices: fp32 Cayley iterates accumulate
    manifold drift whose max-entry residual is much larger in the ``R R^T`` form than the
    ``R^T R`` form the trainer audits (measured 10-20x on 150-step runs), and the fold
    orientation consumes ``R R^T``. The projection moves each entry by O(drift/2) — far
    below one bf16 ulp for the measured ~1e-3 drift — and restores exact orthogonality
    (~1e-14 both forms), making the subsequent checkpoint fold an exact identity again.
    This is the same retraction semantics SGDG itself applies stochastically
    (:meth:`_qr_retraction` with p = 1/101), applied deterministically once at the end.
    """
    U, _, Vh = torch.linalg.svd(R.to(torch.float64))
    return U @ Vh


# --------------------------------------------------------------------------------------
# RotationSet — the learned matrices, fold-ready
# --------------------------------------------------------------------------------------


@dataclass
class RotationSet:
    """Learned rotations in the fold/R.bin key convention, plus training telemetry.

    ``rotations`` maps ``"R1"`` and ``"model.layers.{i}.self_attn.R2"`` to float64 CPU
    matrices — exactly the dict :meth:`fold_rotations` returns and the format SpinQuant's
    optimized-rotation checkpoints (``R.bin``) use. Feed the matrices back through
    ``fold_rotations(model, R1=rs.R1, R2=rs.R2)``.

    ``seam_diags`` (transform-QAT, ``QuantObjective.learn_seam_diag``) maps layer index
    -> ``{"down": s_down [intermediate], "o": s_o [n_kv_heads*head_dim]}`` — the learned
    per-input-channel seam SCALES (``exp`` of the trained log-parameters: strictly
    positive; identity = ones) as float64 CPU vectors, or None when the diagonals were
    not learned. Bake them into a fresh model with
    :meth:`~modelopt.torch.quantization.rotation.fold_seam_diags` (plus
    ``fold_rotations`` for R). ``save``/``load`` round-trip them; old-format R.bin files
    load with ``seam_diags=None``.
    """

    rotations: dict[str, torch.Tensor]
    history: list[dict] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    seam_diags: dict[int, dict[str, torch.Tensor]] | None = None

    def __post_init__(self):
        if "R1" not in self.rotations:
            raise ValueError("RotationSet requires an 'R1' entry")
        self.rotations = {
            k: torch.as_tensor(v).detach().to(torch.float64).cpu()
            for k, v in self.rotations.items()
        }
        if self.seam_diags is not None:
            norm = {}
            for k, pair in self.seam_diags.items():
                if set(pair) != {"down", "o"}:
                    raise ValueError(
                        f"seam_diags[{k!r}]: expected keys {{'down', 'o'}}, got {set(pair)}"
                    )
                norm[int(k)] = {
                    kk: torch.as_tensor(vv).detach().to(torch.float64).cpu().flatten()
                    for kk, vv in pair.items()
                }
                for kk, vv in norm[int(k)].items():
                    if not bool((vv > 0).all()):
                        raise ValueError(
                            f"seam_diags[{k!r}][{kk!r}]: scales must be strictly positive"
                        )
            self.seam_diags = norm

    @property
    def R1(self) -> torch.Tensor:
        """The global residual-stream rotation (the ``"R1"`` entry)."""
        return self.rotations["R1"]

    @property
    def R2(self) -> dict[str, torch.Tensor]:
        """The per-layer R2 sub-dict (keys ``model.layers.{i}.self_attn.R2``)."""
        return {k: v for k, v in self.rotations.items() if k != "R1"}

    def ortho_audit(self) -> dict[str, float]:
        """Per rotation: ``max(max |R^T R - I|, max |R R^T - I|)`` in float64.

        Both forms are measured because their max-entry residuals differ for
        near-orthogonal matrices (basis-dependent; measured 10-20x on raw trained R1),
        and the fold orientation consumes the ``R R^T`` form.
        """
        out = {}
        for k, R in self.rotations.items():
            eye = torch.eye(R.shape[0], dtype=torch.float64)
            out[k] = max(
                (R.t() @ R - eye).abs().max().item(),
                (R @ R.t() - eye).abs().max().item(),
            )
        return out

    def save(self, path) -> None:
        """Write the flat float64 rotation dict (R.bin-compatible; telemetry not saved).

        With learned ``seam_diags`` present, they ride along under the reserved
        :data:`_SEAM_DIAGS_KEY` entry (new format); ``seam_diags=None`` writes exactly
        the legacy flat dict, byte-format-identical to before.
        """
        payload: dict[str, Any] = dict(self.rotations)
        if self.seam_diags is not None:
            payload[_SEAM_DIAGS_KEY] = {
                int(k): {kk: vv.clone() for kk, vv in pair.items()}
                for k, pair in self.seam_diags.items()
            }
        torch.save(payload, path)

    @classmethod
    def load(
        cls, path, ortho_tol: float = LEARNED_ORTHO_TOL, orthogonalize: bool = False
    ) -> "RotationSet":
        """Load a flat rotation dict; refuse matrices off the manifold beyond ortho_tol.

        ``orthogonalize=True`` applies the polar retraction to every matrix before the
        gate — for raw legacy R.bins written without the final retraction (their
        ``R R^T`` residual is typically ~1e-3 and would be refused otherwise).

        Backward compatible both ways: old-format files (pure rotation dict) load with
        ``seam_diags=None``; new-format files carry the learned seam diagonals under
        :data:`_SEAM_DIAGS_KEY` (never polar-projected — they are not rotations).
        """
        raw = torch.load(path, map_location="cpu", weights_only=True)
        seam_diags = raw.pop(_SEAM_DIAGS_KEY, None)
        if orthogonalize:
            raw = {k: _polar_project(torch.as_tensor(v)) for k, v in raw.items()}
        rs = cls(rotations=raw, seam_diags=seam_diags)
        for k, err in rs.ortho_audit().items():
            if err >= ortho_tol:
                raise ValueError(
                    f"{path}: rotation {k!r} is not orthogonal "
                    f"(max ortho residual = {err:.3e} >= {ortho_tol}) — refusing to load"
                    " (raw trained R.bin? pass orthogonalize=True for the polar"
                    " retraction)"
                )
        return rs


# --------------------------------------------------------------------------------------
# Effective-weight assembly (differentiable mirror of fold.py's orientation table)
# --------------------------------------------------------------------------------------


def _assemble_effective_weights(
    base: dict[str, torch.Tensor],
    R1: torch.Tensor,
    R2s: list[torch.Tensor],
    n_layers: int,
    head_dim: int,
    objective: QuantObjective | None,
    out_dtype: torch.dtype,
    seam_diag_params: list[dict[str, torch.Tensor]] | None = None,
    r4_had: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build every rotated (and weight-fake-quantized) effective weight for one forward.

    Same orientation table as fold.py — readers ``W @ R1``, writers ``R1^T @ W``,
    embed/lm_head included, v_proj per-KV-head rows ``R2^T @ W_h``, o_proj per-Q-head
    columns ``@ R2`` — but computed out-of-place in R's dtype (fp32) with R1/R2 as graph
    leaves, so ``loss.backward()`` reaches the rotation parameters. Because every
    residual-stream seam is consistently rotated, the assembled model computes the
    offline-rotated model's function for ANY orthogonal R1/R2 (see README.md).

    ``seam_diag_params`` (transform-QAT): optional per-layer ``{"down": log_s_down
    [intermediate], "o": log_s_o [n_kv_heads*head_dim]}`` graph leaves. Applied with the
    T14-prefold structure BEFORE weight fake-quant, prefold-inside / rotation-outside
    (i.e. exactly the composition ``t14_sq_prefold`` -> ``fold_rotations``): up_proj
    rows ``/ s_down``; down_proj cols ``* s_down``; v_proj rows ``/ s_o`` before the
    per-KV-head R2 row step; o_proj cols ``* s_o`` expanded per q-head group before the
    per-Q-head R2 column step. A functional identity for any positive diagonal (and a
    bitwise no-op path when None), so gradients on ``log s`` come only from the
    quantization error, like the rotations'.

    ``r4_had`` (``QuantObjective.r4_in_graph``): the normalized Walsh-Hadamard for the
    down_proj seam — the effective down_proj weight gets its columns rotated (``@ H``),
    pairing with the input-side ``x @ H`` hook into a functional identity that only the
    quantizers can see. Training-graph only: the fold consumes the returned R1/R2 and
    never sees H.
    """
    compute = R1.dtype
    d = head_dim

    def fin(w):
        if objective is not None and objective.w_bits is not None:
            w = _fq_weight(w, objective)
        return w.to(out_dtype)

    eff = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):  # never fake-quantized
        eff[name] = (base[name].to(compute) @ R1).to(out_dtype)

    for i in range(n_layers):
        R2 = R2s[i]
        pre = f"model.layers.{i}."
        sp = None if seam_diag_params is None else seam_diag_params[i]
        s_down = None if sp is None else torch.exp(sp["down"].to(compute))
        s_o = None if sp is None else torch.exp(sp["o"].to(compute))

        for proj in ("self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"):
            n = pre + proj + ".weight"
            eff[n] = fin(base[n].to(compute) @ R1)  # readers

        n = pre + "mlp.up_proj.weight"  # reader (R1); down-seam rows / s_down
        a = base[n].to(compute) @ R1
        if s_down is not None:
            a = a / s_down[:, None]
        eff[n] = fin(a)

        n = pre + "mlp.down_proj.weight"  # writer (R1); down-seam cols * s_down
        w = R1.t() @ base[n].to(compute)
        if s_down is not None:
            w = w * s_down[None, :]
        if r4_had is not None:  # training-graph R4: cols @ H, inverse of the x @ H hook
            w = w @ r4_had.to(compute)
        eff[n] = fin(w)

        n = pre + "self_attn.v_proj.weight"  # reader (R1) + o-seam rows / s_o (KV dim)
        a = base[n].to(compute) @ R1
        o_f, i_f = a.shape
        if s_o is not None:
            a = a / s_o[:, None]  # before R2: prefold-inside, rotation-outside
        a = (a.t().reshape(i_f, o_f // d, d) @ R2).reshape(i_f, o_f).t().contiguous()
        eff[n] = fin(a)

        n = pre + "self_attn.o_proj.weight"  # writer (R1) + o-seam cols * s_o expanded
        w = R1.t() @ base[n].to(compute)
        o_f, i_f = w.shape
        if s_o is not None:
            n_kv = s_o.numel() // d
            group = i_f // s_o.numel()  # q-heads per KV head (GQA sharing)
            s_full = s_o.reshape(n_kv, 1, d).expand(n_kv, group, d).reshape(-1)
            w = w * s_full[None, :]  # before R2: prefold-inside, rotation-outside
        eff[n] = fin((w.reshape(o_f, i_f // d, d) @ R2).reshape(o_f, i_f))
    return eff


def _iter_batches(calib_loader: Iterable, steps: int):
    """Yield ``steps`` batches, restarting the loader between epochs if re-iterable."""
    if steps <= 0:
        return
    n = 0
    it = iter(calib_loader)
    while n < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(calib_loader)
            try:
                batch = next(it)
            except StopIteration:
                raise ValueError(
                    "calib_loader yielded no batches (a one-shot generator that is already "
                    "exhausted? pass a re-iterable loader, e.g. a list or DataLoader)"
                ) from None
        yield batch
        n += 1


def _batch_input_ids(batch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        ids = batch
    elif isinstance(batch, Mapping) or hasattr(batch, "keys"):
        ids = batch["input_ids"]
    else:
        raise TypeError(
            f"unsupported calib batch type {type(batch).__name__}: pass input_ids tensors "
            "[bs, seq] or dicts with an 'input_ids' key"
        )
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    assert ids.dim() == 2, f"input_ids must be [bs, seq], got shape {tuple(ids.shape)}"
    return ids


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def learn_rotations(
    model: nn.Module,
    calib_loader: Iterable,
    steps: int = 150,
    lr: float = 1.5,
    mode: str = "hadamard",
    objective_cfg: QuantObjective | None = W4A4_G128_OBJECTIVE,
    seed: int = 0,
    init_rotations: Mapping[str, torch.Tensor] | None = None,
    log_every: int = 10,
    teacher: nn.Module | None = None,
    kd_alpha: float = 0.5,
    kd_temp: float = 2.0,
) -> RotationSet:
    """Learn SpinQuant rotations R1 + per-layer R2 for ``model`` by Cayley SGD.

    The model's weights are FROZEN; only the rotations train. Each step assembles the
    rotated effective weights out-of-place (same orientation table as
    :meth:`fold_rotations`), applies the objective's weight fake-quant (STE), runs the
    frozen model reparametrized with those weights on a calibration batch (activation
    fake-quant applied by pre-hooks per the objective), and takes one SGDG step on the
    next-token cross-entropy with cosine lr decay — the SpinQuant recipe.

    Side effects on ``model`` (identical to fold_rotations' pre-rotation steps, so a
    subsequent ``fold_rotations(model, R1=..., R2=...)`` on the SAME object — or on a
    fresh copy — is equivalent): tied embeddings are untied with a real clone
    (``config.tie_word_embeddings`` set False), RMSNorm gains are fused into downstream
    linears, and all parameters get ``requires_grad=False``. Weights are otherwise
    unchanged (the rotated weights live only in the per-step reparametrization);
    Qwen3 q/k_norm are bitwise untouched (asserted). Seeds the global torch CPU RNG and
    Python's ``random`` (for SGDG's stochastic QR retraction).

    Args:
        model: HuggingFace causal LM whose class is registered in fold.py's
            ``_ARCH_REGISTRY`` (``LlamaForCausalLM``, ``Qwen3ForCausalLM``). Run on the
            model's current device/dtype (CPU fp32 works; GPU bf16 recommended for real
            models).
        calib_loader: Re-iterable of calibration batches — ``input_ids`` tensors
            ``[bs, seq]`` or dicts with an ``input_ids`` key. Cycled for ``steps`` steps.
        steps: Number of Cayley-SGD steps (reference budget: 150). ``steps=0`` returns
            the untouched init (== ``fold_rotations(mode=mode, seed=seed)`` draws).
        lr: Peak learning rate, cosine-decayed to 0 (official SpinQuant default 1.5).
        mode: Rotation init family, ``"hadamard"`` (random-sign Hadamard) or ``"random"``
            (Haar QR) — same draw order as fold_rotations, so equal seeds give the same
            init matrices.
        objective_cfg: :class:`QuantObjective` for the fake-quant loss, or None to train
            against the unquantized CE (near-zero gradient — the rotated model is exactly
            equivalent; useful only as a null check).
        seed: Seed for the init draws and the retraction RNG.
        init_rotations: Optional warm start — a dict in the R.bin key convention
            (overrides mode/seed draws; gated at :data:`LEARNED_ORTHO_TOL`).
        log_every: Print a progress line every N steps (0 = silent).
        teacher: Optional frozen reference model for a KD objective (T25): loss becomes
            ``(1-kd_alpha)*CE + kd_alpha*kd_temp^2*KL(student || teacher)`` with the
            teacher's logits computed under ``no_grad`` on the same batch. The teacher
            is NEVER reparametrized/fused/modified — pass a separate (typically bf16)
            copy, not the model being trained. ``teacher=None`` (default) is the plain
            CE objective, bitwise-identical to the pre-T25 trainer.
        kd_alpha: KD mixing weight (only with ``teacher``; T22's measured setting 0.5).
        kd_temp: KD softmax temperature (only with ``teacher``; T22 setting 2.0).

    Returns:
        :class:`RotationSet` with float64 CPU matrices (audited orthonormal to
        :data:`LEARNED_ORTHO_TOL`), per-step ``history`` and hyperparameter ``meta``.
        With ``objective_cfg.learn_seam_diag=True`` (transform-QAT) it additionally
        carries the jointly learned per-layer seam scales in
        :attr:`RotationSet.seam_diags` — trained by a separate plain-Adam group
        (:data:`SEAM_DIAG_LR`, same cosine schedule; never the SGDG stiefel group) and
        bakeable into a fresh model via
        :meth:`~modelopt.torch.quantization.rotation.fold_seam_diags`.
    """
    from torch.nn.utils import stateless as _stateless

    arch = type(model).__name__
    if arch not in _ARCH_REGISTRY:
        raise NotImplementedError(
            f"learn_rotations: unsupported architecture {arch!r}; "
            f"supported: {sorted(_ARCH_REGISTRY)}"
        )
    spec: dict[str, Any] = _ARCH_REGISTRY[arch]

    decoder = model.model
    layers = decoder.layers
    embed = decoder.embed_tokens
    n_layers = len(layers)
    hidden = model.config.hidden_size
    head_dim = spec["head_dim"](model.config)
    device = embed.weight.device
    model_dtype = embed.weight.dtype
    t_all = time.time()

    # 1. Untie embeddings (real clone) — same as fold_rotations step 1.
    if model.lm_head.weight.data_ptr() == embed.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(
            embed.weight.data.clone(), requires_grad=embed.weight.requires_grad
        )
    model.config.tie_word_embeddings = False

    # Snapshots for post-conditions.
    qk_norm_before = {}
    if spec["has_qk_norm"]:
        for idx, layer in enumerate(layers):
            qk_norm_before[f"{idx}.q_norm"] = layer.self_attn.q_norm.weight.data.clone()
            qk_norm_before[f"{idx}.k_norm"] = layer.self_attn.k_norm.weight.data.clone()
    shapes_before = {n: tuple(p.shape) for n, p in model.named_parameters()}

    # 2. Fuse RMSNorm gains into downstream linears (fold_rotations step 3; fused norms
    # become exactly ones, which is what lets R1 commute through RMSNorm). Fusing an
    # already-fused model (all-ones gains) is a bitwise no-op, so this is idempotent.
    for layer in layers:
        for norm_name, linear_names in spec["norm_edges"]:
            _fuse_norm_into_linears(
                layer.get_submodule(norm_name),
                [layer.get_submodule(n) for n in linear_names],
            )
    _fuse_norm_into_linears(decoder.norm, [model.lm_head])

    # 3. Freeze the model; rotations are the only trainable parameters.
    model.eval()
    model.requires_grad_(False)

    # 4. Rotation parameters: fp32 on-device, init = the SAME seeded draw order as
    # fold_rotations (R1 first, then R2 by ascending layer), or a warm start.
    torch.manual_seed(seed)
    random.seed(seed)  # SGDG's stochastic QR-retraction draw — reproducible trajectories
    if init_rotations is None:
        draws = {"R1": _get_orthogonal_matrix(hidden, mode)}
        for i in range(n_layers):
            draws[f"model.layers.{i}.self_attn.R2"] = _get_orthogonal_matrix(head_dim, mode)
    else:
        draws = {
            k: torch.as_tensor(v).detach().to(torch.float64).cpu()
            for k, v in init_rotations.items()
        }
        assert "R1" in draws and draws["R1"].shape == (hidden, hidden), (
            f"init_rotations: R1 missing or wrong shape (want {(hidden, hidden)})"
        )
        for i in range(n_layers):
            k = f"model.layers.{i}.self_attn.R2"
            assert k in draws and draws[k].shape == (head_dim, head_dim), (
                f"init_rotations: missing/misshaped {k}"
            )
        for k, R64 in draws.items():
            eye = torch.eye(R64.shape[0], dtype=torch.float64)
            err = (R64 @ R64.T - eye).abs().max().item()
            assert err < LEARNED_ORTHO_TOL, (
                f"init_rotations[{k!r}]: not orthogonal (max |R R^T - I| = {err:.3e})"
            )
    R1 = nn.Parameter(draws["R1"].to(device=device, dtype=torch.float32))
    R2s = [
        nn.Parameter(draws[f"model.layers.{i}.self_attn.R2"].to(device=device, dtype=torch.float32))
        for i in range(n_layers)
    ]

    # 4b. Transform-QAT seam diagonals (QuantObjective.learn_seam_diag): per-layer
    # log-scale vectors for the two rotation-surviving seams, init zeros (= identity
    # scales, so the assembled model is function-preserving at init and NO extra RNG is
    # consumed — the R draw/trajectory stream is unchanged either way).
    learn_seam_diag = objective_cfg is not None and objective_cfg.learn_seam_diag
    seam_diag_params: list[dict[str, nn.Parameter]] | None = None
    if learn_seam_diag:
        intermediate = model.config.intermediate_size
        n_kv = model.config.num_key_value_heads
        n_q = model.config.num_attention_heads
        assert n_q % n_kv == 0, f"GQA group not integral: {n_q} q heads / {n_kv} kv heads"
        seam_diag_params = [
            {
                "down": nn.Parameter(torch.zeros(intermediate, dtype=torch.float32, device=device)),
                "o": nn.Parameter(torch.zeros(n_kv * head_dim, dtype=torch.float32, device=device)),
            }
            for _ in range(n_layers)
        ]

    # 5. Base-weight references (frozen, post-fusion) + objective preconditions.
    sd = dict(model.named_parameters())
    base = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].data,
        "lm_head.weight": sd["lm_head.weight"].data,
    }
    for i in range(n_layers):
        for proj in _ATTN_PROJS + _MLP_PROJS:
            name = f"model.layers.{i}.{proj}.weight"
            base[name] = sd[name].data
            bias = f"model.layers.{i}.{proj}.bias"
            assert bias not in sd, (
                f"{bias} exists — bias handling is not implemented in the assembly"
            )
            if (
                objective_cfg is not None
                and objective_cfg.w_bits is not None
                and objective_cfg.w_group is not None
            ):
                assert base[name].shape[1] % objective_cfg.w_group == 0, (
                    f"{name}: in_features {base[name].shape[1]} not divisible by "
                    f"w_group {objective_cfg.w_group}"
                )

    r4_had = None
    if objective_cfg is not None and objective_cfg.r4_in_graph:
        r4_had = _walsh_hadamard(model.config.intermediate_size, device=device)
        assert base["model.layers.0.mlp.down_proj.weight"].shape[1] == r4_had.shape[0], (
            "down_proj in_features != config.intermediate_size"
        )

    hooks = None
    if objective_cfg is not None and (objective_cfg.a_bits is not None or r4_had is not None):
        hooks = _ActQuantHooks(
            objective_cfg,
            r4_had=None if r4_had is None else r4_had.to(model_dtype),
        )
        n_hooked = hooks.attach(model)
        expected_hooks = (
            n_layers * len(_ATTN_PROJS + _MLP_PROJS)
            if objective_cfg.a_bits is not None
            else n_layers
        )
        assert n_hooked == expected_hooks, (
            f"activation hooks attached to {n_hooked} linears, expected {expected_hooks}"
        )

    if learn_seam_diag:
        assert base["model.layers.0.mlp.up_proj.weight"].shape[0] == intermediate, (
            "up_proj out_features != config.intermediate_size"
        )
        assert base["model.layers.0.self_attn.v_proj.weight"].shape[0] == n_kv * head_dim, (
            "v_proj out_features != num_key_value_heads * head_dim"
        )
        assert base["model.layers.0.self_attn.o_proj.weight"].shape[1] == n_q * head_dim, (
            "o_proj in_features != num_attention_heads * head_dim"
        )

    # The rotations train on the Stiefel manifold (SGDG); the seam diagonals are
    # UNCONSTRAINED log-parameters and get their own plain-Adam optimizer (never the
    # stiefel group) at SEAM_DIAG_LR, sharing the cosine schedule below.
    opt = SGDG([R1, *R2s], lr=lr, momentum=0.0, stiefel=True)
    opt_diag = None
    if learn_seam_diag:
        assert seam_diag_params is not None
        opt_diag = torch.optim.Adam(
            [p for sp in seam_diag_params for p in (sp["down"], sp["o"])],
            lr=SEAM_DIAG_LR,
        )
    history: list[dict] = []

    def _r1_ortho() -> float:
        Rd = R1.detach().to(torch.float64)
        eye = torch.eye(Rd.shape[0], dtype=torch.float64, device=Rd.device)
        return (Rd.t() @ Rd - eye).abs().max().item()

    # 6. Training loop.
    try:
        for step, batch in enumerate(_iter_batches(calib_loader, steps)):
            cos_t = 0.5 * (1.0 + math.cos(math.pi * step / max(steps, 1)))
            lr_t = lr * cos_t
            for gp in opt.param_groups:
                gp["lr"] = lr_t
            if opt_diag is not None:  # same cosine schedule, SEAM_DIAG_LR peak
                for gp in opt_diag.param_groups:
                    gp["lr"] = SEAM_DIAG_LR * cos_t
            ids = _batch_input_ids(batch).to(device)
            t0 = time.time()
            eff = _assemble_effective_weights(
                base,
                R1,
                R2s,
                n_layers,
                head_dim,
                objective_cfg,
                model_dtype,
                seam_diag_params=seam_diag_params,
                r4_had=r4_had,
            )
            # Reparametrize for forward AND backward: with activation checkpointing the
            # recompute happens during backward and must still see the effective weights
            # (torch.func.functional_call would restore the originals first).
            with _stateless._reparametrize_module(model, eff):
                out = model(input_ids=ids, labels=ids, use_cache=False)
                loss = out.loss
                if teacher is not None:  # KD objective (T25); teacher never touched
                    with torch.no_grad():
                        tlogits = teacher(input_ids=ids).logits
                    T = kd_temp
                    kd = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(out.logits / T, dim=-1),
                        torch.nn.functional.softmax(tlogits / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                    loss = (1.0 - kd_alpha) * loss + kd_alpha * kd
                opt.zero_grad(set_to_none=True)
                if opt_diag is not None:
                    opt_diag.zero_grad(set_to_none=True)
                loss.backward()
            del eff
            if step == 0:  # gradients must reach every R (and every diag) and nothing else
                assert R1.grad is not None and all(r.grad is not None for r in R2s), (
                    "no gradient reached the rotation parameters"
                )
                if seam_diag_params is not None:
                    assert all(
                        sp[k].grad is not None for sp in seam_diag_params for k in ("down", "o")
                    ), "no gradient reached the seam-diagonal parameters"
                assert all(p.grad is None for p in model.parameters()), (
                    "a frozen model weight received a gradient"
                )
            opt.step()
            if opt_diag is not None:
                opt_diag.step()
            rec = {
                "step": step,
                "lr": round(lr_t, 6),
                "loss": round(loss.item(), 6),
                "r1_ortho": _r1_ortho(),
                "dt_s": round(time.time() - t0, 3),
            }
            history.append(rec)
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(
                    f"[learn_rotations] step {step:4d}/{steps}  lr={lr_t:.4f}  "
                    f"loss={rec['loss']:.4f}  r1_ortho={rec['r1_ortho']:.2e}",
                    flush=True,
                )
    finally:
        if hooks is not None:
            hooks.remove()

    # 7. Post-conditions: the model is functionally the original (untied + fused only).
    if spec["has_qk_norm"]:
        for idx, layer in enumerate(layers):
            assert torch.equal(
                layer.self_attn.q_norm.weight.data, qk_norm_before[f"{idx}.q_norm"]
            ), f"q_norm[{idx}] changed"
            assert torch.equal(
                layer.self_attn.k_norm.weight.data, qk_norm_before[f"{idx}.k_norm"]
            ), f"k_norm[{idx}] changed"
    for n, p in model.named_parameters():
        assert tuple(p.shape) == shapes_before[n], f"shape of {n} changed"

    retraction_log: dict[str, dict[str, float]] = {}
    if history:
        # Final retraction: project each trained matrix to the nearest orthogonal one
        # (see _polar_project). Raw fp32 Cayley drift is recorded per matrix (both
        # residual forms + max entry moved) before being closed to ~1e-14.
        rotations = {}
        raw64 = {"R1": R1.detach().to(torch.float64).cpu()}
        for i, r2 in enumerate(R2s):
            raw64[f"model.layers.{i}.self_attn.R2"] = r2.detach().to(torch.float64).cpu()
        for k, R in raw64.items():
            eye = torch.eye(R.shape[0], dtype=torch.float64)
            proj = _polar_project(R)
            retraction_log[k] = {
                "raw_rtr": (R.t() @ R - eye).abs().max().item(),
                "raw_rrt": (R @ R.t() - eye).abs().max().item(),
                "delta_max": (proj - R).abs().max().item(),
            }
            rotations[k] = proj
    else:
        # steps=0: hand back the exact float64 init draws — bitwise identical to
        # fold_rotations(mode=mode, seed=seed)'s matrices (same RNG, same draw order).
        # No retraction: fresh draws are orthonormal to ~1e-15 by construction.
        rotations = dict(draws)

    # Transform-QAT: export the learned seam scales s = exp(log_s) (fp64 CPU; ones when
    # steps=0 — the identity init). Positivity is structural (exp), so these always pass
    # the RotationSet gate.
    seam_diags = None
    if seam_diag_params is not None:
        seam_diags = {
            i: {
                "down": torch.exp(sp["down"].detach().to(torch.float64)).cpu(),
                "o": torch.exp(sp["o"].detach().to(torch.float64)).cpu(),
            }
            for i, sp in enumerate(seam_diag_params)
        }

    meta = {
        "arch": arch,
        "hidden": hidden,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "steps": steps,
        "lr": lr,
        "mode": mode,
        "seed": seed,
        "objective": None if objective_cfg is None else objective_cfg.__dict__.copy(),
        "kd": None if teacher is None else {"alpha": kd_alpha, "temp": kd_temp},
        "warm_start": init_rotations is not None,
        "final_loss": history[-1]["loss"] if history else None,
        "final_retraction": retraction_log,
        "elapsed_s": round(time.time() - t_all, 1),
    }
    if hooks is not None and hooks.static_amax:
        meta["static_act_amax"] = {k: float(v) for k, v in hooks.static_amax.items()}
    if seam_diags is not None:
        all_s = torch.cat([v for pair in seam_diags.values() for v in pair.values()])
        meta["seam_diag"] = {
            "lr": SEAM_DIAG_LR,
            "s_min": all_s.min().item(),
            "s_max": all_s.max().item(),
        }

    rs = RotationSet(rotations=rotations, history=history, meta=meta, seam_diags=seam_diags)
    audit = rs.ortho_audit()
    worst = max(audit, key=lambda k: audit[k])
    assert audit[worst] < LEARNED_ORTHO_TOL, (
        f"ORTHO AUDIT FAILED after training: max |R^T R - I| = {audit[worst]:.3e} "
        f">= {LEARNED_ORTHO_TOL} ({worst}) — rotations are not deployable"
    )
    return rs
