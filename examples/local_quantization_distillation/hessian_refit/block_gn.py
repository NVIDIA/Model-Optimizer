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

"""Scale-space Gauss-Newton statistics for the interior linears of a gated MLP unit.

For ``o = (silu(x Wg^T) * (x Wu^T)) Wd^T`` with frozen integer levels, the unit output is
nonlinear in the gate/up scale steps ``s_g, s_u`` (rows x K). Per output row ``i`` of the
interior projections the Jacobian of the residual ``r_t = o_t - o_t^teacher`` factors as

    d r_t / d s_{g,i} = wd_i (alpha_{t,i} V_{g,t,i}),   alpha = silu'(g) * u
    d r_t / d s_{u,i} = wd_i (beta_{t,i}  V_{u,t,i}),   beta  = silu(g)

where ``wd_i`` is column ``i`` of the dequantized down weight and ``V_{t,i,k}`` the
group-``k`` sum of ``levels_i * x_t``. Rows are coupled only through ``wd_i^T wd_j``,
which this block-diagonal approximation drops, so the step is damped (Levenberg-Marquardt)
and the result is projected onto the legal FP16-times-nibble hierarchy in the same
Gauss-Newton metric with :func:`block_ls.refit_hierarchy`.
"""

import torch

from . import block_ls

__all__ = ["GaussNewtonStats", "LineSearch", "project", "silu_and_grad"]


def silu_and_grad(g):
    sig = torch.sigmoid(g)
    return g * sig, sig * (1 + g * (1 - sig))


class GaussNewtonStats:
    """Per-row joint (gate, up) Gauss-Newton blocks and gradients for one unit."""

    def __init__(self, rows, k, device):
        shape = (rows, k, k)
        self.m_gg = torch.zeros(shape, dtype=torch.float32, device=device)
        self.m_gu = torch.zeros(shape, dtype=torch.float32, device=device)
        self.m_uu = torch.zeros(shape, dtype=torch.float32, device=device)
        self.b_g = torch.zeros(rows, k, dtype=torch.float64, device=device)
        self.b_u = torch.zeros(rows, k, dtype=torch.float64, device=device)
        self.sq_err = 0.0
        self.tokens = 0
        self.col_norm_sq = torch.zeros(rows, dtype=torch.float64, device=device)
        self.k = k

    @torch.no_grad()
    def add(self, x, g, u, r, w_down, lev_g, lev_u, group, chunk=256):
        """Accumulate one batch of valid tokens.

        ``x`` (T, n) unit input, ``g``/``u`` (T, rows) gate/up pre-activations, ``r``
        (T, d_model) residual, ``w_down`` (d_model, rows) dequantized down weight,
        ``lev_*`` (rows, n) signed integer levels as float.
        """
        rows, n = lev_g.shape
        k = n // group
        if self.tokens == 0:
            self.col_norm_sq = w_down.double().square().sum(0)
        beta, dsilu = silu_and_grad(g.float())
        alpha = dsilu * u.float()
        p = r.float() @ w_down.float()  # (T, rows): wd_i^T r_t
        self.sq_err += float(r.float().square().sum())
        self.tokens += x.shape[0]
        lg = lev_g.reshape(rows, k, group).float()
        lu = lev_u.reshape(rows, k, group).float()
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            for start in range(0, x.shape[0], chunk):
                xs = x[start : start + chunk].float().reshape(-1, k, group)
                vg = torch.einsum("tkp,ikp->itk", xs, lg)  # (rows, T, K)
                vu = torch.einsum("tkp,ikp->itk", xs, lu)
                a = alpha[start : start + chunk].T.unsqueeze(-1)  # (rows, T, 1)
                b = beta[start : start + chunk].T.unsqueeze(-1)
                ps = p[start : start + chunk].T.unsqueeze(-1)
                vg.mul_(a)  # scaled Jacobian factors, shared by all blocks
                vu.mul_(b)
                self.m_gg.baddbmm_(vg.transpose(1, 2), vg)
                self.m_gu.baddbmm_(vg.transpose(1, 2), vu)
                self.m_uu.baddbmm_(vu.transpose(1, 2), vu)
                self.b_g += (vg * ps).sum(1).double()
                self.b_u += (vu * ps).sum(1).double()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    @torch.no_grad()
    def step(self, s_g, s_u, damp, row_chunk=512):
        """Damped Gauss-Newton step; returns continuous targets and the per-row metrics."""
        rows, k = s_g.shape
        c = self.col_norm_sq
        target_g = torch.empty(rows, k, dtype=torch.float64, device=s_g.device)
        target_u = torch.empty_like(target_g)
        for start in range(0, rows, row_chunk):
            stop = min(rows, start + row_chunk)
            cc = c[start:stop, None, None]
            top = torch.cat([self.m_gg[start:stop].double(), self.m_gu[start:stop].double()], dim=2)
            bottom = torch.cat(
                [self.m_gu[start:stop].double().transpose(1, 2), self.m_uu[start:stop].double()],
                dim=2,
            )
            a = cc * torch.cat([top, bottom], dim=1)  # (R, 2K, 2K)
            b = torch.cat([self.b_g[start:stop], self.b_u[start:stop]], dim=1)
            diag = a.diagonal(dim1=1, dim2=2)
            # Absolute floor keeps rows whose gate never activates (all-zero blocks) solvable.
            ridge = damp * diag + 1e-6 * diag.mean() + 1e-30
            system = a + torch.diag_embed(ridge)
            factor, info = torch.linalg.cholesky_ex(system)
            if int(info.max()) != 0:
                system = a + torch.diag_embed(10 * ridge)
                factor = torch.linalg.cholesky(system)
            delta = -torch.cholesky_solve(b.unsqueeze(-1), factor).squeeze(-1)
            target_g[start:stop] = delta[:, :k]
            target_u[start:stop] = delta[:, k:]
        return target_g, target_u

    def metric(self, which):
        """Per-row Gauss-Newton metric for one projection, built on demand (fp64, large)."""
        block = self.m_gg if which == "gate" else self.m_uu
        return self.col_norm_sq[:, None, None] * block.double()


class LineSearch:
    """Exact step length along a direction under the linearized (Gauss-Newton) model.

    Rows are coupled through the shared down projection, which the per-row step ignores,
    so the collective step overshoots. With ``jv_t = J_t delta`` computed analytically, the
    model ``|| r_t + eta jv_t ||^2`` is minimized at ``eta* = -<jv, r> / <jv, jv>``.
    """

    def __init__(self, delta_g, delta_u, lev_g, lev_u, w_down, group):
        self.delta_g, self.delta_u = delta_g.float(), delta_u.float()
        rows, n = lev_g.shape
        k = n // group
        self.lg = lev_g.reshape(rows, k, group).float()
        self.lu = lev_u.reshape(rows, k, group).float()
        self.w_down = w_down.float()
        self.group, self.k = group, k
        self.num = 0.0
        self.den = 0.0
        self.base = 0.0

    @torch.no_grad()
    def add(self, x, g, u, r, chunk=256):
        beta, dsilu = silu_and_grad(g.float())
        alpha = dsilu * u.float()
        r = r.float()
        self.base += float(r.square().sum())
        for start in range(0, x.shape[0], chunk):
            xs = x[start : start + chunk].float().reshape(-1, self.k, self.group)
            q_g = torch.einsum("tkp,ikp,ik->ti", xs, self.lg, self.delta_g)
            q_u = torch.einsum("tkp,ikp,ik->ti", xs, self.lu, self.delta_u)
            inner = alpha[start : start + chunk] * q_g + beta[start : start + chunk] * q_u
            jv = inner @ self.w_down.T  # (T, d_model)
            rs = r[start : start + chunk]
            self.num += float((jv * rs).sum())
            self.den += float(jv.square().sum())

    def eta(self, maximum=1.0):
        if self.den <= 0:
            return 0.0, {"num": self.num, "den": self.den}
        eta = -self.num / self.den
        predicted = self.base + 2 * eta * self.num + eta * eta * self.den
        return min(max(eta, 0.0), maximum), {
            "num": self.num,
            "den": self.den,
            "eta_raw": eta,
            "model_mse_ratio": predicted / max(self.base, 1e-30),
        }


@torch.no_grad()
def project(metric, target, g0, local0, groups_per_block, local_max, damp=0.01, rounds=2):
    """Project continuous steps onto the native hierarchy in the given quadratic metric."""
    c = torch.einsum("rkj,rj->rk", metric, target)
    return block_ls.refit_hierarchy(
        metric, c, g0, local0, groups_per_block, local_max, damp=damp, rounds=rounds
    )
