# Adapted from https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/c5ab4e8/stiefel_optimizer.py
# (the SGDG "Cayley SGD" Stiefel-manifold optimizer of Li et al., ICLR 2020), with the
# repository's helper functions (gutils.py / utils.py) inlined and minor API and
# robustness modifications documented in the port note below. The same code is vendored
# as train_utils/optimizer.py in Meta's SpinQuant repository; this port was verified
# bitwise against that copy on the stiefel branch at momentum 0.0 and 0.9.
# Copyright (c) 2020 Jun Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""SGDG — Cayley SGD on the Stiefel manifold (self-contained port, see header for lineage).

The Stiefel-branch MATH is unchanged from the original — every rotation parameter used by
:mod:`.learn` is square fp32 with ``stiefel=True``, so that branch always runs. Textual
deltas vs. the original:

- stiefel branch: three unused local reads (``weight_decay``/``dampening``/``nesterov``)
  dropped;
- non-stiefel fallback: deprecated ``Tensor.add_(Number, Tensor)`` overloads use the
  modern ``add_(t, alpha=...)`` signature, and the original's bare try/except around the
  weight-decay add (which silently skipped weight decay when stale locals leaked from a
  prior stiefel iteration) is replaced by fresh group reads.

KNOWN ORIGINAL QUIRK, faithfully reproduced: momentum is INERT in the stiefel branch.
``V = momentum * V - g.t()`` rebinds ``V`` to a temp and ``V.copy_(V_new)`` writes into
that temp — ``state["momentum_buffer"]`` stays zeros forever (dead store; proven: momentum
0.9 vs 0.0 give bitwise-identical trajectories). Kept for trajectory parity with the
original code and our reference runs.

NOTE: the stiefel branch draws ``random.randint`` from Python's GLOBAL random module
(original behavior, kept): with p = 1/101 per parameter per step the iterate is
re-projected by QR retraction. Seed ``random`` for reproducible trajectories
(:meth:`.learn.learn_rotations` does this from its ``seed`` argument).
"""

import random

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["SGDG"]


def _unit(v, dim: int = 1, eps: float = 1e-8):
    vnorm = _norm(v, dim)
    return v / vnorm.add(eps), vnorm


def _norm(v, dim: int = 1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def _matrix_norm_one(W):
    return torch.abs(W).sum(dim=0).max()


def _cayley_loop(X, W, tan_vec, t):
    """Fixed-point iteration for the Cayley transform: Y = X + t*W*(X+Y)/2, 5 iterations."""
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()


def _qr_retraction(tan_vec):  # tan_vec: p-by-n, p <= n
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()
    return q


_EPSILON = 1e-8


class SGDG(Optimizer):
    """SGD-G: SGD on the Stiefel manifold via the Cayley transform (see module docstring).

    With ``stiefel=True`` and a square parameter, each step (i) row-normalizes the iterate,
    (ii) builds the skew-symmetric tangent ``W = W_hat - W_hat^T`` from the Riemannian
    gradient projection, (iii) moves along the Cayley curve with step ``min(lr, 1 /
    ||W||_1)`` via 5 fixed-point iterations — an orthogonality-preserving update to
    first order. With p = 1/101 per step the iterate is exactly re-orthogonalized by QR.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        stiefel: bool = False,
        omega: float = 0,
        grad_clip=None,
    ) -> None:
        """Set up parameter groups with the original SGDG defaults (see class docstring)."""
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "stiefel": stiefel,
            "omega": 0,
            "grad_clip": grad_clip,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        """Restore optimizer state, defaulting ``nesterov`` for groups pickled without it."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step (Cayley-curve update on the stiefel branch)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                unity, _ = _unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = _qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)

                    lr = group["lr"]

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(g.t().size(), device=p.device)

                    V = param_state["momentum_buffer"]
                    V = momentum * V - g.t()
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (_matrix_norm_one(W) + _EPSILON)
                    alpha = min(t, lr)

                    p_new = _cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t())  # n-by-p
                    p.data.copy_(p_new.view(p.size()))
                    # Original dead store, reproduced verbatim: V was rebound to a temp
                    # above, so this never reaches state["momentum_buffer"] — momentum is
                    # inert in the stiefel branch (see module docstring).
                    V.copy_(V_new)

                else:
                    weight_decay = group["weight_decay"]
                    dampening = group["dampening"]
                    nesterov = group["nesterov"]
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(p.data, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        d_p = d_p.add(buf, alpha=momentum) if nesterov else buf
                    p.data.add_(d_p, alpha=-group["lr"])

        return loss
