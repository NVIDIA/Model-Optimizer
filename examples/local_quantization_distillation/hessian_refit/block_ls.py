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

"""Block least-squares refit of native GGML scale fields for a unit's last linear.

Given frozen integer levels ``L`` (rows x in), the student Gram ``H_ss = Z_s^T Z_s``,
the cross Gram ``H_st = Z_s^T Z_t`` over the unit's valid tokens and the teacher weight
``W``, choose per-group steps ``s`` (rows x K) minimizing

    || Z_s (L * expand(s))^T - Z_t W^T ||_F^2 = sum_r  s_r^T A_r s_r - 2 s_r^T c_r + const

with ``A_r = V_r^T H_ss V_r`` and ``c_r = V_r^T H_st w_r``, where ``V_r`` places the
levels of each group in its columns. The steps are then projected onto the format's
hierarchy, an FP16 block scale ``g_b`` times the legal local multiplier ``(2 l + 1) / 8``
with ``l`` in ``[0, local_max]``, by alternating an exact block solve for ``g`` and an
exact per-group enumeration for ``l``.

Experimental; the Gram accumulation and model wiring live in the drivers.
"""

import torch

__all__ = [
    "expand_steps",
    "format_geometry",
    "group_quadratics",
    "objective",
    "refine_local",
    "refit_hierarchy",
    "solve_block_scales",
    "solve_continuous",
]

GGML_BLOCK = 256


def _chunk_rows(rows, k, group, budget_bytes=2 << 30):
    per_row = k * k * group * 4
    return max(1, min(rows, budget_bytes // max(1, per_row)))


@torch.no_grad()
def group_quadratics(levels, h_ss, u, group):
    """Return ``A`` (rows, K, K) and ``c`` (rows, K) in float64 for group size ``group``.

    ``levels`` are signed integer levels (rows x in), ``h_ss`` the student Gram (in x in)
    and ``u = W @ H_st^T`` (rows x in), so ``c[r, k] = sum_{p in k} L[r, p] u[r, p]``.
    """
    rows, n = levels.shape
    if n % group:
        raise ValueError("Group size must divide the input dimension.")
    k = n // group
    levels = levels.to(torch.float32).reshape(rows, k, group)
    h4 = h_ss.to(torch.float32).reshape(k, group, k, group)
    a = torch.empty(rows, k, k, dtype=torch.float64, device=levels.device)
    for start in range(0, rows, _chunk_rows(rows, k, group)):
        chunk = levels[start : start + _chunk_rows(rows, k, group)]
        # T[r, k, j, q] = sum_p L[r, k, p] H[k, p, j, q]
        partial = torch.einsum("rkp,kpjq->rkjq", chunk, h4)
        a[start : start + chunk.shape[0]] = torch.einsum("rkjq,rjq->rkj", partial, chunk).double()
    c = (levels * u.to(torch.float32).reshape(rows, k, group)).sum(-1).double()
    return a, c


@torch.no_grad()
def objective(a, c, s):
    """Per-row ``s^T A s - 2 s^T c`` (without the constant teacher term)."""
    s = s.double()
    return torch.einsum("rk,rkj,rj->r", s, a, s) - 2 * (s * c).sum(-1)


@torch.no_grad()
def solve_continuous(a, c, damp=0.01):
    """Ridge least squares per row; ``damp`` scales the mean diagonal.

    Rows whose system is not positive definite (rank-deficient statistics, rounding) get a
    progressively heavier ridge instead of failing the whole batch.
    """
    k = a.shape[-1]
    diag = a.diagonal(dim1=1, dim2=2)
    scale = diag.abs().mean(-1) + 1e-8 * diag.abs().mean() + 1e-30
    ridge = damp * diag.mean(-1).clamp_min(0) + 1e-8 * diag.abs().mean() + 1e-30
    eye = torch.eye(k, dtype=a.dtype, device=a.device)
    system = a + ridge[:, None, None] * eye
    factor, info = torch.linalg.cholesky_ex(system)
    extra = 1e-6
    while int(info.max()) != 0:
        bad = info != 0
        system[bad] = a[bad] + (ridge[bad] + extra * scale[bad])[:, None, None] * eye
        factor_bad, info_bad = torch.linalg.cholesky_ex(system[bad])
        factor[bad] = factor_bad
        info = info.clone()
        info[bad] = info_bad
        extra *= 100
        if extra > 1e6:
            raise RuntimeError("Scale solve stayed indefinite under a dominant ridge.")
    return torch.cholesky_solve(c.unsqueeze(-1), factor).squeeze(-1)


def _block_index(k, groups_per_block, device):
    return torch.arange(k, device=device) // groups_per_block


@torch.no_grad()
def solve_block_scales(a, c, multipliers, groups_per_block, damp=0.01):
    """Exact per-row solve for FP16 block scales given fixed local multipliers.

    ``multipliers`` is (rows, K) with entries ``(2 l + 1) / 8``. Returns ``g`` (rows, B),
    clamped to be non-negative and rounded to FP16.
    """
    rows, k = multipliers.shape
    b = k // groups_per_block
    index = _block_index(k, groups_per_block, a.device)
    m = torch.zeros(rows, k, b, dtype=torch.float64, device=a.device)
    m[:, torch.arange(k, device=a.device), index] = multipliers.double()
    a_g = torch.einsum("rkb,rkj,rjd->rbd", m, a, m)
    c_g = torch.einsum("rkb,rk->rb", m, c)
    g = solve_continuous(a_g, c_g, damp).clamp_min(0)
    return g.to(torch.float16).double()


@torch.no_grad()
def refine_local(a, c, g, local, groups_per_block, local_max, sweeps=2):
    """Exact coordinate descent over legal local scales, one group at a time.

    ``g`` is (rows, B) block scales, ``local`` (rows, K) integer multipliers. Each step
    evaluates every legal ``l`` for one group with the other groups held fixed, so the
    objective never increases. Returns the updated ``local`` and final steps ``s``.
    """
    rows, k = local.shape
    index = _block_index(k, groups_per_block, a.device)
    g_k = g.gather(1, index.expand(rows, -1))  # (rows, K)
    candidates = (2 * torch.arange(local_max + 1, device=a.device, dtype=torch.float64) + 1) / 8
    s = g_k * (2 * local.double() + 1) / 8
    a_s = torch.einsum("rkj,rj->rk", a, s)
    diag = a.diagonal(dim1=1, dim2=2)
    for _ in range(sweeps):
        for j in range(k):
            residual = c[:, j] - a_s[:, j] + diag[:, j] * s[:, j]
            trial = g_k[:, j : j + 1] * candidates[None]  # (rows, L)
            values = diag[:, j : j + 1] * trial.square() - 2 * trial * residual[:, None]
            best = values.argmin(-1)
            new = trial.gather(1, best[:, None]).squeeze(1)
            delta = new - s[:, j]
            a_s += a[:, :, j] * delta[:, None]
            s[:, j] = new
            local[:, j] = best
    return local, s


@torch.no_grad()
def refit_hierarchy(a, c, g0, local0, groups_per_block, local_max, *, damp=0.01, rounds=3):
    """Alternate exact block-scale solves and local enumeration from the PTQ fields.

    Returns ``(g, local, s, history)`` where ``history`` lists the summed objective after
    the initial state, the continuous solve (reference only) and each round.
    """
    rows, k = local0.shape
    index = _block_index(k, groups_per_block, a.device)
    local = local0.clone().long()
    g = g0.double().clone()
    s = g.gather(1, index.expand(rows, -1)) * (2 * local.double() + 1) / 8
    history = {"ptq": float(objective(a, c, s).sum())}
    history["continuous"] = float(objective(a, c, solve_continuous(a, c, damp)).sum())
    for round_index in range(rounds):
        g = solve_block_scales(a, c, (2 * local.double() + 1) / 8, groups_per_block, damp)
        local, s = refine_local(a, c, g, local, groups_per_block, local_max)
        history[f"round{round_index}"] = float(objective(a, c, s).sum())
    return g, local, s, history


def expand_steps(levels, s, group):
    """Dequantize ``levels`` with per-group steps ``s`` (rows, K) to float32."""
    rows, n = levels.shape
    return (levels.float().reshape(rows, -1, group) * s.float().unsqueeze(-1)).reshape(rows, n)


def format_geometry(fmt):
    """Return ``(group, groups_per_block, local_max)`` for a GGML format object."""
    group = fmt.int8_group_size
    local_max = 7 if fmt.name.startswith("iq1") else 15
    return group, GGML_BLOCK // group, local_max
