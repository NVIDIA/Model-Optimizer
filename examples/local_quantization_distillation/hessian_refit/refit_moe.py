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

"""Offline per-expert Hessian refit of native GGML scale fields for routed-expert MoE layers.

For one routed-expert MoE unit, with frozen codes,

    out_t = sum_{e in top-k(t)} w_{t,e} * D_e (silu(clamp(G_e x_t)) * clamp(U_e x_t)) + shared(x_t)

the router and the shared expert are identical in teacher and student, so the unit error is
``sum_e w_{t,e} (y^s_{t,e} - y^t_{t,e})``. Each expert is refit on the tokens routed to it,
weighted by the squared routing weight (the cross-expert terms are dropped):

* ``down``: the expert output is linear in its scale steps, so a per-row least squares with
  the student Gram and the student/teacher cross Gram is exact (``block_ls``), kept close to
  the PTQ scales by a proximal ridge.
* ``gate_up``, three step kinds that can be scheduled per iteration, each with an exact line
  search along the step for the row coupling through ``down``:
  ``block`` is a per-row joint Gauss-Newton step on the FP16 scale of every 256-weight block
  (relative coordinates, exact curvature); ``kfac`` moves every group step, using the exact
  gradient and a Kronecker-factored curvature (input Gram times the per-row second moment of
  the activation derivative); ``group`` moves every group step with the exact per-row-pair
  curvature from a token subsample. The last two project onto the FP16-times-nibble hierarchy.
* optional gain steps that rescale block scales with codes frozen, and gain diagnostics.

Inputs (formats in the README): per-pack captures of every unit's valid-token inputs and
routing, the exported PTQ payloads, and the source checkpoint for the teacher expert weights
(GLM-5.3-Flash layout). Output: per layer, payloads whose codes are untouched and whose scale
fields are refit. Experimental.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from modelopt.torch.quantization.ggml.registry import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields

from . import block_ls

BLOCK = block_ls.GGML_BLOCK


@contextlib.contextmanager
def tf32():
    """TF32 only where a damped system absorbs its rounding; Grams and forwards stay exact fp32."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


# --------------------------------------------------------------------------------------
# Native payload view
# --------------------------------------------------------------------------------------
class Projection:
    """One packed weight: integer levels plus the FP16 block scales and local multipliers."""

    def __init__(self, packed, shape, fmt, device):
        self.fmt, self.shape = fmt, tuple(shape)
        rows, n = self.shape
        self.blocks = n // BLOCK
        self.group = fmt.int8_group_size
        self.groups_per_block = BLOCK // self.group
        self.packed = packed.to(device).reshape(rows, self.blocks, fmt.block_bytes).contiguous()
        self.refresh()

    def refresh(self):
        rows, n = self.shape
        levels, steps = self.fmt.decode_int8(
            self.packed, torch.tensor(self.shape, dtype=torch.int64)
        )
        self.levels, self.steps = levels, steps  # int8 [rows, n], float32 [rows, n / group]
        scale, local, self.local_max = decode_scale_fields(self.packed, self.fmt)
        self.scale = scale.reshape(rows, self.blocks)  # signed, exactly FP16
        self.local = local.reshape(rows, -1).long()

    def weight(self):
        """Dequantized float32 weight."""
        return self.levels.float() * self.steps.repeat_interleave(self.group, dim=1)

    def set_fields(self, scale, local=None):
        local = self.local if local is None else local
        self.packed = replace_scale_fields(
            self.packed,
            self.fmt,
            scale.reshape(-1).float(),
            local.reshape(-1, self.groups_per_block),
        ).reshape_as(self.packed)
        self.refresh()

    def check_layout(self):
        expected = (
            self.scale.repeat_interleave(self.groups_per_block, dim=1) * (2 * self.local + 1) / 8
        )
        torch.testing.assert_close(self.steps, expected.to(self.steps.dtype), rtol=1e-5, atol=1e-12)


# --------------------------------------------------------------------------------------
# Functional expert math
# --------------------------------------------------------------------------------------
def activation(gate, up, limit):
    """SwiGLU with clamps: value and the partial derivatives with respect to both pre-activations."""
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(-limit, limit)
    sig = torch.sigmoid(gate_c)
    silu = gate_c * sig
    z = silu * up_c
    alpha = sig * (1 + gate_c * (1 - sig)) * up_c * (gate <= limit)
    beta = silu * (up.abs() <= limit)
    return z, alpha, beta


def block_partials(x, weight):
    """``v[t, i, b] = sum_{p in block b} weight[i, p] x[t, p]`` for x [m, n], weight [rows, n]."""
    m, n = x.shape
    rows = weight.shape[0]
    blocks = n // BLOCK
    xb = x.reshape(m, blocks, BLOCK).permute(1, 0, 2)  # [blocks, m, 256]
    wb = weight.reshape(rows, blocks, BLOCK).permute(1, 2, 0)  # [blocks, 256, rows]
    return torch.bmm(xb, wb).permute(1, 2, 0)  # [m, rows, blocks]


def group_partials(x, weight, group):
    """``v[t, i, g] = sum_{p in group g} weight[i, p] x[t, p]`` for x [m, n], weight [rows, n]."""
    m, n = x.shape
    rows = weight.shape[0]
    xb = x.reshape(m, n // group, group).permute(1, 0, 2)  # [groups, m, group]
    wb = weight.reshape(rows, n // group, group).permute(1, 2, 0)  # [groups, group, rows]
    return torch.bmm(xb, wb).permute(1, 2, 0)  # [m, rows, groups]


@torch.no_grad()
def teacher_forward(x, w_gu, w_d, limit, chunk):
    n = x.shape[0]
    inter = w_gu.shape[0] // 2
    z = torch.empty(n, inter, device=x.device)
    y = torch.empty(n, w_d.shape[0], device=x.device)
    for s in range(0, n, chunk):
        pre = x[s : s + chunk].float() @ w_gu.T
        z[s : s + chunk] = activation(pre[:, :inter], pre[:, inter:], limit)[0]
        y[s : s + chunk] = z[s : s + chunk] @ w_d.T
    return z, y


@torch.no_grad()
def student_output(x, w_gu, w_d, limit, chunk):
    n = x.shape[0]
    inter = w_gu.shape[0] // 2
    y = torch.empty(n, w_d.shape[0], device=x.device)
    for s in range(0, n, chunk):
        pre = x[s : s + chunk].float() @ w_gu.T
        y[s : s + chunk] = activation(pre[:, :inter], pre[:, inter:], limit)[0] @ w_d.T
    return y


# --------------------------------------------------------------------------------------
# Solves
# --------------------------------------------------------------------------------------
@torch.no_grad()
def solve_down(proj, h_ss, h_st, w_teacher, prox, rounds, sweeps):
    """Exact per-row least squares for the down scale fields, proximal to the current ones."""
    rows, n = proj.shape
    sign = torch.where(proj.scale < 0, -1.0, 1.0)
    signed = (proj.levels.float().reshape(rows, proj.blocks, BLOCK) * sign.unsqueeze(-1)).reshape(
        rows, n
    )
    g0 = proj.scale.abs().double()
    local = proj.local.clone()
    s0 = g0.repeat_interleave(proj.groups_per_block, dim=1) * (2 * local.double() + 1) / 8
    u = w_teacher @ h_st.T
    a, c = block_ls.group_quadratics(signed, h_ss, u, proj.group)
    diag = a.diagonal(dim1=1, dim2=2)
    lam = prox * diag.mean(-1) + 1e-9 * diag.mean() + 1e-30
    diag.add_(lam[:, None])
    c.add_(lam[:, None] * s0)
    g = g0
    for _ in range(rounds):
        g = block_ls.solve_block_scales(
            a, c, (2 * local.double() + 1) / 8, proj.groups_per_block, damp=1e-9
        )
        local, _ = block_ls.refine_local(
            a, c, g, local, proj.groups_per_block, proj.local_max, sweeps=sweeps
        )
    changed = float((local != proj.local).double().mean())
    proj.set_fields(sign * g.float(), local)
    return {
        "local_changed": changed,
        "global_rel_change": float(((g - g0).abs() / g0.clamp_min(1e-12)).mean()),
    }


@torch.no_grad()
def damped_solve(a, b, damp):
    """Solve ``(a + damp * diag(a)) x = b`` per row, escalating the ridge on indefinite rows."""
    diag = a.diagonal(dim1=1, dim2=2)
    floor = diag.abs().mean(-1, keepdim=True) + diag.abs().mean() + 1e-30
    ridge = damp * diag.clamp_min(0) + 1e-6 * diag.abs().mean() + 1e-30
    system = a + torch.diag_embed(ridge)
    factor, info = torch.linalg.cholesky_ex(system)
    extra = 1e-4
    while int(info.max()) != 0:
        bad = info != 0
        system[bad] = a[bad] + torch.diag_embed(ridge[bad] + extra * floor[bad])
        factor_bad, info_bad = torch.linalg.cholesky_ex(system[bad])
        factor[bad] = factor_bad
        info = info.clone()
        info[bad] = info_bad
        extra *= 100
        if extra > 1e9:
            raise RuntimeError("Scale system stayed indefinite under a dominant ridge.")
    return torch.cholesky_solve(b.unsqueeze(-1), factor).squeeze(-1)


@torch.no_grad()
def project_steps(proj, metric, target, sign, rounds, sweeps):
    """Nearest legal FP16-times-nibble fields to ``target`` steps in the given per-row metric."""
    diag = metric.diagonal(dim1=1, dim2=2)
    # A weak identity term keeps rows the data never excites exactly where they are.
    diag.add_((1e-3 * diag.mean(-1) + 1e-6 * diag.mean() + 1e-30)[:, None])
    c = torch.einsum("rkj,rj->rk", metric, target)
    local = proj.local.clone()
    g = proj.scale.abs().double()
    for _ in range(rounds):
        g = block_ls.solve_block_scales(
            metric, c, (2 * local.double() + 1) / 8, proj.groups_per_block, damp=1e-9
        )
        local, _ = block_ls.refine_local(
            metric, c, g, local, proj.groups_per_block, proj.local_max, sweeps=sweeps
        )
    changed = float((local != proj.local).double().mean())
    proj.set_fields(sign * g.float(), local)
    return changed


@torch.no_grad()
def gauss_newton_direction(m, b, col_norm_sq, damp):
    """Per-row damped step in relative block-scale coordinates: ``[inter, 2 * blocks]``."""
    a = col_norm_sq[:, None, None].double() * m.double()
    diag = a.diagonal(dim1=1, dim2=2)
    ridge = damp * diag + 1e-6 * diag.mean() + 1e-30
    system = a + torch.diag_embed(ridge)
    factor, info = torch.linalg.cholesky_ex(system)
    extra = 1e-3
    while int(info.max()) != 0:
        bad = info != 0
        system[bad] = a[bad] + torch.diag_embed(
            ridge[bad] + extra * (diag[bad].abs().mean(-1, keepdim=True) + diag.abs().mean())
        )
        factor_bad, info_bad = torch.linalg.cholesky_ex(system[bad])
        factor[bad] = factor_bad
        info = info.clone()
        info[bad] = info_bad
        extra *= 100
        if extra > 1e9:
            raise RuntimeError("Gauss-Newton system stayed indefinite under a dominant ridge.")
    return -torch.cholesky_solve(b.unsqueeze(-1), factor).squeeze(-1)


@torch.no_grad()
def refine_expert(x, w, gu, dn, w_gu_t, w_d_t, args):
    """Alternate the exact down solve and line-searched gate_up steps for one expert."""
    device = x.device
    n = x.shape[0]
    inter = gu.shape[0] // 2
    hidden = dn.shape[0]
    blocks, group = gu.blocks, gu.group
    limit, chunk = args.swiglu_limit, args.token_chunk
    z_t, y_t = teacher_forward(x, w_gu_t, w_d_t, limit, chunk)
    w2 = w * w
    report = {"tokens": n, "eta": [], "steps": []}
    z_s = torch.empty(n, inter, device=device)
    schedule = [kind for kind in args.interior.split(",") if kind and kind != "none"]
    # Activation derivatives are kept only when a gate_up step will line search with them.
    alpha_all = torch.empty(n if schedule else 0, inter, device=device)
    beta_all = torch.empty(n if schedule else 0, inter, device=device)
    # Level geometry of gate_up is fixed: magnitudes move, signs and codes do not.
    sign = torch.where(gu.scale < 0, -1.0, 1.0)
    signed = None
    gram_x = levels_metric = None
    for iteration in range(args.iters + 1):
        kind = schedule[iteration % len(schedule)] if schedule and iteration < args.iters else None
        w_gu_s, w_d_s = gu.weight(), dn.weight()
        h_ss = torch.zeros(inter, inter, device=device)
        h_st = torch.zeros(inter, inter, device=device)
        if kind == "block":
            m = torch.zeros(inter, 2 * blocks, 2 * blocks, device=device)
            b = torch.zeros(inter, 2 * blocks, dtype=torch.float64, device=device)
        elif kind == "group":
            groups = gu.shape[1] // group
            grad = torch.zeros(2 * inter, hidden, device=device)
        elif kind == "kfac":
            if signed is None:
                signed = (
                    gu.levels.float().reshape(2 * inter, blocks, BLOCK) * sign.unsqueeze(-1)
                ).reshape(2 * inter, -1)
                gram_x = torch.zeros(hidden, hidden, device=device)
                for s in range(0, n, chunk):
                    xw = x[s : s + chunk].float() * w[s : s + chunk, None]
                    gram_x.addmm_(xw.T, xw)
                levels_metric = block_ls.group_quadratics(
                    signed, gram_x, torch.zeros_like(signed), group
                )[0]
                del gram_x
            grad = torch.zeros(2 * inter, hidden, device=device)
            moment = torch.zeros(2 * inter, dtype=torch.float64, device=device)
        squared = 0.0
        for s in range(0, n, chunk):
            xc = x[s : s + chunk].float()
            wc = w[s : s + chunk]
            if kind == "block":
                v = block_partials(xc, w_gu_s)
                pre = v.sum(-1)
            else:
                pre = xc @ w_gu_s.T
            z, alpha, beta = activation(pre[:, :inter], pre[:, inter:], limit)
            z_s[s : s + chunk] = z
            wz = z * wc[:, None]
            h_ss.addmm_(wz.T, wz)
            h_st.addmm_(wz.T, z_t[s : s + chunk] * wc[:, None])
            rho = z @ w_d_s.T - y_t[s : s + chunk]
            squared += float((rho.square().sum(-1) * w2[s : s + chunk]).sum())
            if kind is None:
                continue
            alpha_all[s : s + chunk], beta_all[s : s + chunk] = alpha, beta
            p = (rho * wc[:, None]) @ w_d_s  # [m, inter], carries one factor of w
            if kind == "block":
                jw = torch.cat(
                    (
                        v[:, :inter] * (alpha * wc[:, None]).unsqueeze(-1),
                        v[:, inter:] * (beta * wc[:, None]).unsqueeze(-1),
                    ),
                    dim=-1,
                )  # [m, inter, 2 * blocks]
                with tf32():
                    m.baddbmm_(jw.permute(1, 2, 0), jw.permute(1, 0, 2))
                b += torch.einsum("ti,tia->ia", p, jw).double()
                del v, jw
            else:
                xw = xc * wc[:, None]
                grad[:inter].addmm_((p * alpha).T, xw)
                grad[inter:].addmm_((p * beta).T, xw)
                if kind == "kfac":
                    wc2 = w2[s : s + chunk, None]
                    moment[:inter] += (alpha.square() * wc2).sum(0).double()
                    moment[inter:] += (beta.square() * wc2).sum(0).double()
        if iteration == 0:
            report["mse_before"] = squared / (n * hidden)
        col_norm_sq = w_d_s.square().sum(0)
        if args.down == "exact":
            report["down"] = solve_down(
                dn, h_ss, h_st, w_d_t, args.prox, args.down_rounds, args.down_sweeps
            )
            w_d_s = dn.weight()
        if kind is None:
            break
        if kind == "block":
            delta = gauss_newton_direction(
                m, b, col_norm_sq, args.gn_damp
            ).float()  # relative, [inter, 2 * blocks]
            delta_rows = torch.cat(
                (delta[:, :blocks], delta[:, blocks:]), dim=0
            )  # [2 * inter, blocks]
            dw = w_gu_s * delta_rows.repeat_interleave(BLOCK, dim=1)
        elif kind == "group":
            # Exact per-row-pair curvature over every scale step of the pair, from an evenly
            # spaced token subsample; the gradient above uses all tokens.
            m = torch.zeros(inter, 2 * groups, 2 * groups, device=device)
            pick = torch.linspace(0, n - 1, min(n, args.gn_tokens), device=device).long().unique()
            mass = 0.0
            for s in range(0, pick.numel(), args.gn_chunk):
                rows_t = pick[s : s + args.gn_chunk]
                wc = w[rows_t]
                v = group_partials(x[rows_t].float(), w_gu_s, group)
                pre = v.sum(-1)
                _, alpha, beta = activation(pre[:, :inter], pre[:, inter:], limit)
                jw = torch.cat(
                    (
                        v[:, :inter] * (alpha * wc[:, None]).unsqueeze(-1),
                        v[:, inter:] * (beta * wc[:, None]).unsqueeze(-1),
                    ),
                    dim=-1,
                )  # [m, inter, 2 * groups]
                del v
                with tf32():
                    m.baddbmm_(jw.permute(1, 2, 0), jw.permute(1, 0, 2))
                mass += float(wc.square().sum())
                del jw
            m *= float(w2.sum()) / max(mass, 1e-30)
            b = (w_gu_s * grad).reshape(2 * inter, groups, group).sum(-1).double()
            delta = gauss_newton_direction(
                m, torch.cat((b[:inter], b[inter:]), dim=1), col_norm_sq, args.gn_damp
            ).float()
            delta_rows = torch.cat(
                (delta[:, :groups], delta[:, groups:]), dim=0
            )  # relative, [2 * inter, groups]
            dw = w_gu_s * delta_rows.repeat_interleave(group, dim=1)
        else:
            row_scale = torch.cat((col_norm_sq, col_norm_sq)).double() * moment / float(w2.sum())
            metric = levels_metric * row_scale[:, None, None]
            gradient = (signed * grad).reshape(2 * inter, -1, group).sum(-1).double()
            delta_steps = -damped_solve(
                metric, gradient, args.gn_damp
            )  # absolute, [2 * inter, groups]
            dw = signed * delta_steps.float().repeat_interleave(group, dim=1)
        num = den = 0.0
        for s in range(0, n, chunk):
            dpre = x[s : s + chunk].float() @ dw.T
            dz = (
                alpha_all[s : s + chunk] * dpre[:, :inter]
                + beta_all[s : s + chunk] * dpre[:, inter:]
            )
            jd = dz @ w_d_s.T
            rho = z_s[s : s + chunk] @ w_d_s.T - y_t[s : s + chunk]
            wc2 = w2[s : s + chunk]
            num += float(((jd * rho).sum(-1) * wc2).sum())
            den += float((jd.square().sum(-1) * wc2).sum())
        eta = min(max(-num / den, 0.0), args.eta_max) if den > 0 else 0.0
        report["eta"].append(eta)
        report["steps"].append(kind)
        if kind == "block":
            factor = (1 + eta * delta_rows).clamp(1 - args.step_clip, 1 + args.step_clip)
            gu.set_fields(gu.scale * factor)
        elif kind == "group":
            current = gu.steps.abs().double()
            target = current * (1 + eta * delta_rows.double()).clamp(
                1 - args.step_clip, 1 + args.step_clip
            )
            # Project in each row's own curvature (the pair's diagonal blocks), in absolute steps.
            inverse = torch.where(
                current > 0, 1 / current.clamp_min(1e-30), torch.zeros_like(current)
            )
            metric = torch.cat((m[:, :groups, :groups], m[:, groups:, groups:]), dim=0).double()
            del m
            metric *= inverse[:, :, None]
            metric *= inverse[:, None, :]
            project_steps(gu, metric, target, sign, args.project_rounds, args.down_sweeps)
            del metric
        else:
            target = (gu.steps.abs().double() + eta * delta_steps).clamp_min(0)
            project_steps(gu, metric, target, sign, args.project_rounds, args.down_sweeps)
            del metric
    rho = student_output(x, gu.weight(), dn.weight(), limit, chunk) - y_t
    report["mse_after"] = float((rho.square().sum(-1) * w2).sum()) / (n * hidden)
    return report


def _apply_factor(proj, factor, clip):
    """Multiply the FP16 block scales by ``factor`` ([rows] or [rows, blocks]); codes frozen."""
    factor = torch.where(torch.isfinite(factor) & (factor > 0), factor, torch.ones_like(factor))
    factor = factor.clamp(1 / clip, clip).float()
    if factor.dim() == 1:
        factor = factor[:, None]
    proj.set_fields(proj.scale * factor)
    return float(factor.mean())


def _linear_factor(target, decoded, gram, mode):
    """Derived-d gain step of one linear under its input Gram: per row or per 256-column block.

    ``row``/``block`` put the output gain at 1 (``t'Gt / q'Gt``); ``row_energy``/``block_energy``
    match the output energy (``sqrt(t'Gt / q'Gq)``, the geometric mean of the least-squares and
    the gain-1 scale). Block modes use the block-diagonal part of the Gram.
    """
    rows, n = target.shape
    if mode.startswith("row"):
        gt = target @ gram
        tt = (gt * target).sum(-1)
        if mode == "row":
            return tt / (gt * decoded).sum(-1).clamp_min(1e-30)
        return (tt / ((decoded @ gram) * decoded).sum(-1).clamp_min(1e-30)).sqrt()
    blocks = n // BLOCK
    index = torch.arange(blocks, device=gram.device)
    gb = gram.reshape(blocks, BLOCK, blocks, BLOCK)[index, :, index]  # [blocks, 256, 256]
    tb, qb = target.reshape(rows, blocks, BLOCK), decoded.reshape(rows, blocks, BLOCK)
    gt = torch.einsum("rbi,bij->rbj", tb, gb)
    tt = (gt * tb).sum(-1)
    if mode == "block":
        return tt / (gt * qb).sum(-1).clamp_min(1e-30)
    return (tt / (torch.einsum("rbi,bij->rbj", qb, gb) * qb).sum(-1).clamp_min(1e-30)).sqrt()


@torch.no_grad()
def gain_step(x, w, gu, dn, w_gu_t, w_d_t, args):
    """Diagnose the output gain of one expert and, if asked, apply a derived-d gain step last.

    Gate_up is corrected as a linear under its input Gram. Down is corrected either the same way
    under the student's intermediate Gram (teacher weight as target, what GPTQ sees) or against
    the unit target per output row (``unit_row``: gain 1, ``unit_energy``: energy match).
    """
    limit, chunk = args.swiglu_limit, args.token_chunk
    n = x.shape[0]
    inter, hidden = gu.shape[0] // 2, dn.shape[0]
    weights = w if args.gain_weighted else torch.ones_like(w)
    report = {}

    def stats():
        """Unit sums per output row and the two input Grams at the current student state."""
        w_gu_s, w_d_s = gu.weight(), dn.weight()
        gram_x = torch.zeros(hidden, hidden, device=x.device)
        gram_z = torch.zeros(inter, inter, device=x.device)
        dot = torch.zeros(hidden, dtype=torch.float64, device=x.device)
        teacher = torch.zeros_like(dot)
        student = torch.zeros_like(dot)
        pre_num = pre_den = 0.0
        for s in range(0, n, chunk):
            xc = x[s : s + chunk].float()
            wc, wu = w[s : s + chunk, None], weights[s : s + chunk, None]
            xw = xc * wu
            gram_x.addmm_(xw.T, xw)
            pre_s, pre_t = xc @ w_gu_s.T, xc @ w_gu_t.T
            pre_num += float((pre_s * pre_t * wu.square()).sum())
            pre_den += float((pre_t.square() * wu.square()).sum())
            z = activation(pre_s[:, :inter], pre_s[:, inter:], limit)[0]
            zw = z * wu
            gram_z.addmm_(zw.T, zw)
            y_s = (z @ w_d_s.T) * wc
            y_t = (activation(pre_t[:, :inter], pre_t[:, inter:], limit)[0] @ w_d_t.T) * wc
            dot += (y_s * y_t).sum(0).double()
            teacher += y_t.square().sum(0).double()
            student += y_s.square().sum(0).double()
        return gram_x, gram_z, dot, teacher, student, pre_num / max(pre_den, 1e-30)

    gram_x, gram_z, dot, teacher, student, pre_gain = stats()
    report.update(
        gate_up_gain=pre_gain,
        unit_gain=float(dot.sum() / teacher.sum().clamp_min(1e-30)),
        unit_energy=float(student.sum() / teacher.sum().clamp_min(1e-30)),
        unit_mse=float((student.sum() + teacher.sum() - 2 * dot.sum()) / (n * hidden)),
    )
    changed = False
    if args.gain_gate_up != "none":
        gram_x.diagonal().add_(args.gain_damp * gram_x.diagonal().mean())
        factor = _linear_factor(w_gu_t, gu.weight(), gram_x, args.gain_gate_up)
        report["gate_up_factor"] = _apply_factor(gu, factor, args.gain_clip)
        changed = True
    del gram_x
    if args.gain_down != "none":
        if changed:  # the intermediate moved with gate_up
            _, gram_z, dot, teacher, student, _ = stats()
        if args.gain_down == "unit_row":
            factor = teacher / dot.clamp_min(1e-30)
        elif args.gain_down == "unit_energy":
            factor = (teacher / student.clamp_min(1e-30)).sqrt()
        else:
            gram_z.diagonal().add_(args.gain_damp * gram_z.diagonal().mean())
            factor = _linear_factor(w_d_t, dn.weight(), gram_z, args.gain_down)
        report["down_factor"] = _apply_factor(dn, factor.float(), args.gain_clip)
        changed = True
    if changed:
        _, _, dot, teacher, student, pre_gain = stats()
        report.update(
            gate_up_gain_after=pre_gain,
            unit_gain_after=float(dot.sum() / teacher.sum().clamp_min(1e-30)),
            unit_energy_after=float(student.sum() / teacher.sum().clamp_min(1e-30)),
            unit_mse_after=float((student.sum() + teacher.sum() - 2 * dot.sum()) / (n * hidden)),
        )
    return report


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def dequantize_fp8(quantized, scales, device):
    q = quantized.to(torch.float32).to(device)
    rows, cols = q.shape[-2:]
    s = (
        (scales.to(torch.float32) - 127.0).exp2()
        if scales.dtype == torch.uint8
        else scales.to(torch.float32)
    )
    s = s.to(device)
    scale_rows, scale_cols = s.shape[-2:]
    q = q.reshape(scale_rows, rows // scale_rows, scale_cols, cols // scale_cols)
    return (q * s.reshape(scale_rows, 1, scale_cols, 1)).reshape(rows, cols).to(torch.bfloat16)


def load_teacher_layer(source, layer, device, num_experts):
    """Teacher routed and shared expert weights of one layer, dequantized as the model loads them."""
    source = Path(source)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp"
    with contextlib.ExitStack() as stack:
        handles = {}

        def read(key):
            name = index[key]
            if name not in handles:
                handles[name] = stack.enter_context(
                    safe_open(str(source / name), framework="pt", device="cpu")
                )
            return handles[name].get_tensor(key)

        def weight(key):
            value = read(key)
            scale_key = key + "_scale_inv"
            return (
                dequantize_fp8(value, read(scale_key), device)
                if scale_key in index
                else value.to(device, torch.bfloat16)
            )

        first = weight(f"{prefix}.experts.0.gate_proj.weight")
        inter, hidden = first.shape
        gate_up = torch.empty(num_experts, 2 * inter, hidden, dtype=torch.bfloat16, device=device)
        down = torch.empty(num_experts, hidden, inter, dtype=torch.bfloat16, device=device)
        for e in range(num_experts):
            gate_up[e, :inter] = (
                first if e == 0 else weight(f"{prefix}.experts.{e}.gate_proj.weight")
            )
            gate_up[e, inter:] = weight(f"{prefix}.experts.{e}.up_proj.weight")
            down[e] = weight(f"{prefix}.experts.{e}.down_proj.weight")
        shared = {
            part: weight(f"{prefix}.shared_experts.{part}_proj.weight")
            for part in ("gate", "up", "down")
        }
    return gate_up, down, shared


def load_payloads(directory, layer):
    records = sorted(Path(directory).glob(f"layer-{layer:02d}-rank-*.pt"))
    if not records:
        raise FileNotFoundError(f"No payload export for layer {layer} in {directory}.")
    result, covered = {}, 0
    for path in records:
        record = torch.load(path, map_location="cpu", weights_only=True)
        total = record["total"]
        for projection in ("gate_up", "down"):
            if projection not in result:
                result[projection] = torch.empty(
                    total, record[projection].shape[1], dtype=torch.uint8
                )
                result[projection + "_shape"] = record[projection + "_shape"]
            result[projection][record["start"] : record["start"] + record["count"]] = record[
                projection
            ]
        covered += record["count"]
        result["format"] = record["format"]
    if covered != result["gate_up"].shape[0]:
        raise ValueError(
            f"Layer {layer}: payload export covers {covered} of {result['gate_up'].shape[0]} experts."
        )
    return result


def load_capture(directory, split, layer, device, max_packs):
    files = sorted((Path(directory) / split / f"layer-{layer:02d}").glob("pack-*.pt"))
    if max_packs:
        files = files[:max_packs]
    if not files:
        raise FileNotFoundError(f"No {split} capture for layer {layer} in {directory}.")
    xs, indices, weights, check = [], [], [], None
    count = 0
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if "y" in record and check is None:
            check = (count, record["y"])
        xs.append(record["x"])
        indices.append(record["idx"])
        weights.append(record["w"])
        count += record["x"].shape[0]
    return {
        "x": torch.cat(xs).to(device),
        "idx": torch.cat(indices).to(device).long(),
        "w": torch.cat(weights).to(device),
        "check": check,
        "packs": len(files),
    }


def routing_table(indices, weights, num_experts):
    """Token ids and routing weights sorted by expert, with per-expert offsets."""
    n, k = indices.shape
    flat = indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tokens = torch.arange(n, device=indices.device).repeat_interleave(k)[order]
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=indices.device)
    offsets[1:] = counts.cumsum(0)
    return tokens, weights.reshape(-1)[order], offsets


# --------------------------------------------------------------------------------------
# Layer driver
# --------------------------------------------------------------------------------------
@torch.no_grad()
def solve_layer(layer, args, fmt, device):
    started = time.perf_counter()
    payload = load_payloads(args.payloads, layer)
    num_experts = payload["gate_up"].shape[0]
    gate_up_t, down_t, shared = load_teacher_layer(args.source, layer, device, num_experts)
    hidden = gate_up_t.shape[2]
    fit = load_capture(args.capture, "fit", layer, device, args.fit_packs)
    val = load_capture(args.capture, "validation", layer, device, args.validation_packs)
    fit_tokens, fit_weights, fit_offsets = routing_table(fit["idx"], fit["w"], num_experts)
    val_tokens, val_weights, val_offsets = routing_table(val["idx"], val["w"], num_experts)
    loaded = time.perf_counter()
    n_val = val["x"].shape[0]
    residual_ptq = torch.zeros(n_val, hidden, device=device)
    residual_new = torch.zeros(n_val, hidden, device=device)
    check = None
    if fit["check"] is not None:
        start, target = fit["check"]
        check = {
            "start": start,
            "stop": start + target.shape[0],
            "target": target.to(device).float(),
            "sum": torch.zeros(target.shape[0], hidden, device=device),
        }
    if args.init_dir:
        # Start from an earlier solver output (same codes, other scale fields) instead of PTQ.
        initial = torch.load(
            Path(args.init_dir) / f"layer-{layer:02d}.pt", map_location="cpu", weights_only=True
        )
        for projection in ("gate_up", "down"):
            if initial[projection].shape != payload[projection].shape:
                raise ValueError(
                    f"Layer {layer}: {args.init_dir} does not match the payload layout."
                )
            payload[projection] = initial[projection]
        del initial
    out = {"gate_up": payload["gate_up"].clone(), "down": payload["down"].clone()}
    use_gain = args.gain_report or args.gain_gate_up != "none" or args.gain_down != "none"
    unit_student = torch.zeros(n_val if use_gain else 0, hidden, device=device)
    unit_teacher = torch.zeros(n_val if use_gain else 0, hidden, device=device)
    experts: list[dict] = []
    skipped, mapping_error = 0, 0.0
    first, last = (int(v) for v in args.experts.split(":")) if args.experts else (0, num_experts)
    val_before = val_after = 0.0
    generator = torch.Generator(device="cpu").manual_seed(args.seed + layer)
    for e in range(first, last):
        gu = Projection(payload["gate_up"][e], payload["gate_up_shape"], fmt, device)
        dn = Projection(payload["down"][e], payload["down_shape"], fmt, device)
        w_gu_t, w_d_t = gate_up_t[e].float(), down_t[e].float()
        if e == 0:
            gu.check_layout()
            dn.check_layout()
        relative = float((gu.weight() - w_gu_t).norm() / w_gu_t.norm().clamp_min(1e-30))
        mapping_error = max(mapping_error, relative)
        if relative > args.mapping_tolerance:
            raise ValueError(
                f"Layer {layer} expert {e}: payload does not match the teacher expert ({relative:.3f})."
            )
        # Validation tokens: the PTQ contribution, later the refit one.
        v_tok = val_tokens[val_offsets[e] : val_offsets[e + 1]]
        v_w = val_weights[val_offsets[e] : val_offsets[e + 1]]
        if v_tok.numel():
            x_val = val["x"][v_tok]
            y_val_t = teacher_forward(x_val, w_gu_t, w_d_t, args.swiglu_limit, args.token_chunk)[1]
            rho = (
                student_output(x_val, gu.weight(), dn.weight(), args.swiglu_limit, args.token_chunk)
                - y_val_t
            )
            residual_ptq.index_add_(0, v_tok, rho * v_w[:, None])
            val_before += float((rho.square().sum(-1) * v_w.square()).sum())
        tok = fit_tokens[fit_offsets[e] : fit_offsets[e + 1]]
        w = fit_weights[fit_offsets[e] : fit_offsets[e + 1]]
        if check is not None and tok.numel():
            inside = (tok >= check["start"]) & (tok < check["stop"])
            if bool(inside.any()):
                y_check = teacher_forward(
                    fit["x"][tok[inside]], w_gu_t, w_d_t, args.swiglu_limit, args.token_chunk
                )[1]
                check["sum"].index_add_(
                    0, tok[inside] - check["start"], y_check * w[inside][:, None]
                )
        if tok.numel() > args.max_tokens:
            keep = torch.randperm(tok.numel(), generator=generator)[: args.max_tokens].to(device)
            tok, w = tok[keep], w[keep]
        if tok.numel() < args.min_tokens:
            skipped += 1
            experts.append({"expert": e, "tokens": int(tok.numel()), "skipped": True})
        else:
            solving = args.down != "none" or any(
                k and k != "none" for k in args.interior.split(",")
            )
            report = (
                refine_expert(fit["x"][tok], w, gu, dn, w_gu_t, w_d_t, args)
                if solving
                else {"tokens": int(tok.numel())}
            )
            if use_gain:
                report["gain"] = gain_step(fit["x"][tok], w, gu, dn, w_gu_t, w_d_t, args)
            report["expert"] = e
            experts.append(report)
            out["gate_up"][e] = gu.packed.reshape(-1).cpu()
            out["down"][e] = dn.packed.reshape(-1).cpu()
        if v_tok.numel():
            y_val_s = student_output(
                x_val, gu.weight(), dn.weight(), args.swiglu_limit, args.token_chunk
            )
            rho = y_val_s - y_val_t
            residual_new.index_add_(0, v_tok, rho * v_w[:, None])
            val_after += float((rho.square().sum(-1) * v_w.square()).sum())
            if use_gain:
                unit_student.index_add_(0, v_tok, y_val_s * v_w[:, None])
                unit_teacher.index_add_(0, v_tok, y_val_t * v_w[:, None])
        if args.progress and (e + 1) % args.progress == 0:
            done = [r for r in experts if not r.get("skipped")]
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "experts": e + 1,
                        "elapsed": time.perf_counter() - started,
                        "mean_mse_before": sum(r.get("mse_before", 0.0) for r in done)
                        / max(1, len(done)),
                        "mean_mse_after": sum(r.get("mse_after", 0.0) for r in done)
                        / max(1, len(done)),
                    }
                ),
                flush=True,
            )
    summary = {
        "layer": layer,
        "format": payload["format"],
        "experts": num_experts,
        "skipped_experts": skipped,
        "fit_tokens": int(fit["x"].shape[0]),
        "fit_packs": fit["packs"],
        "validation_tokens": int(n_val),
        "validation_unit_mse_ptq": float(residual_ptq.square().sum()) / max(1, n_val * hidden),
        "validation_unit_mse_refit": float(residual_new.square().sum()) / max(1, n_val * hidden),
        "validation_expert_sum_ptq": val_before / max(1, n_val * hidden),
        "validation_expert_sum_refit": val_after / max(1, n_val * hidden),
        "expert_range": [first, last],
        "max_payload_teacher_relative_error": mapping_error,
        "load_seconds": loaded - started,
        "solve_seconds": time.perf_counter() - loaded,
    }
    done = [r for r in experts if not r.get("skipped")]
    if use_gain:
        energy = float(unit_teacher.square().sum().clamp_min(1e-30))
        summary["validation_unit_gain"] = float((unit_student * unit_teacher).sum()) / energy
        summary["validation_unit_energy"] = float(unit_student.square().sum()) / energy
        gains = [r["gain"] for r in done if "gain" in r]
        tokens = [r["tokens"] for r in done if "gain" in r]
        for key in (
            "gate_up_gain",
            "unit_gain",
            "unit_energy",
            "gate_up_gain_after",
            "unit_gain_after",
            "unit_energy_after",
            "gate_up_factor",
            "down_factor",
        ):
            values = [(g[key], t) for g, t in zip(gains, tokens) if key in g]
            if values:
                summary["fit_" + key] = sum(v * t for v, t in values) / max(
                    1, sum(t for _, t in values)
                )
    if done and "mse_before" in done[0]:
        summary["expert_mse_before"] = sum(r["mse_before"] for r in done) / len(done)
        summary["expert_mse_after"] = sum(r["mse_after"] for r in done) / len(done)
        etas = [eta for r in done for eta in r["eta"][:1]]
        summary["first_eta_mean"] = sum(etas) / len(etas) if etas else None
    if check is not None and (first, last) == (0, num_experts):
        x_check = fit["x"][check["start"] : check["stop"]].float()
        gate = x_check @ shared["gate"].float().T
        up = x_check @ shared["up"].float().T
        shared_out = activation(gate, up, args.swiglu_limit)[0] @ shared["down"].float().T
        error = (check["sum"] + shared_out - check["target"]).norm() / check[
            "target"
        ].norm().clamp_min(1e-30)
        summary["teacher_unit_relative_error"] = float(error)
        if float(error) > args.check_tolerance:
            raise ValueError(
                f"Layer {layer}: functional teacher unit differs from the captured output ({float(error):.4f})."
            )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f".layer-{layer:02d}.pt.pending"
    torch.save(out, temporary)
    temporary.replace(out_dir / f"layer-{layer:02d}.pt")
    (out_dir / f"layer-{layer:02d}.json").write_text(
        json.dumps({"summary": summary, "experts": experts}, indent=1) + "\n"
    )
    print(json.dumps({"summary": summary}), flush=True)
    return summary


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True)
    p.add_argument("--payloads", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", required=True, help="comma-separated layer indices")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--format", default="iq2_s")
    p.add_argument(
        "--interior",
        default="block",
        help="comma-separated per-iteration schedule of none|block|kfac|group",
    )
    p.add_argument(
        "--init-dir", default="", help="start from this solver output instead of the PTQ payloads"
    )
    p.add_argument(
        "--gain-report", action="store_true", help="measure output gains without changing anything"
    )
    p.add_argument(
        "--gain-gate-up",
        default="none",
        choices=["none", "row", "row_energy", "block", "block_energy"],
    )
    p.add_argument(
        "--gain-down",
        default="none",
        choices=["none", "row", "row_energy", "block", "block_energy", "unit_row", "unit_energy"],
    )
    p.add_argument("--gain-damp", type=float, default=0.01)
    p.add_argument("--gain-clip", type=float, default=2.0)
    p.add_argument(
        "--gain-weighted", action="store_true", help="weight the gain statistics by routing weight"
    )
    p.add_argument("--gn-tokens", type=int, default=16384, help="tokens for the group curvature")
    p.add_argument("--gn-chunk", type=int, default=1024)
    p.add_argument("--project-rounds", type=int, default=1)
    p.add_argument(
        "--experts", default="", help="first:last subset of experts to refit (others keep PTQ)"
    )
    p.add_argument("--down", default="exact", choices=["none", "exact"])
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--gn-damp", type=float, default=0.1)
    p.add_argument("--eta-max", type=float, default=1.0)
    p.add_argument("--step-clip", type=float, default=0.5)
    p.add_argument("--prox", type=float, default=0.01)
    p.add_argument("--down-rounds", type=int, default=2)
    p.add_argument("--down-sweeps", type=int, default=1)
    p.add_argument("--min-tokens", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=131072)
    p.add_argument("--token-chunk", type=int, default=4096)
    p.add_argument("--fit-packs", type=int, default=0)
    p.add_argument("--validation-packs", type=int, default=0)
    p.add_argument("--swiglu-limit", type=float, default=10.0)
    p.add_argument("--mapping-tolerance", type=float, default=0.9)
    p.add_argument("--check-tolerance", type=float, default=0.02)
    p.add_argument("--progress", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fmt = GGML_FORMAT_REGISTRY[args.format]
    device = torch.device(args.device)
    for layer in [int(v) for v in args.layers.split(",") if v]:
        if (Path(args.out) / f"layer-{layer:02d}.json").is_file():
            print(json.dumps({"layer": layer, "status": "already solved"}), flush=True)
            continue
        solve_layer(layer, args, fmt, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
