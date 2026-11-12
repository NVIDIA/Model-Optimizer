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

"""Extended SGDG numerics tests: manifold invariance, step cap, quirk contracts, retractions.

Drives the SGDG port (:class:`modelopt.torch.quantization.rotation.SGDG`) through fully
deterministic seeded runs (the stiefel branch draws its stochastic-QR-retraction trigger
from Python's global ``random`` — one ``randint`` per parameter per step, so seeding
``random`` fixes the trajectory) and asserts orthogonality invariance under adversarial
gradients, the ``alpha = min(t, lr)`` step cap, the documented inert-momentum dead-store
quirk, both retraction branches, and the polar-projection nearest-orthogonal property.

Plain test_* functions with asserts: collectable by pytest, and also runnable without it
via ``python test_rotation_ext_sgdg.py`` (the __main__ driver runs every test function
and exits nonzero on any failure). CPU-only, tiny matrices, seconds per test.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only unit tests: never claim a GPU

import random
import sys
import traceback

import torch

from modelopt.torch.quantization.rotation import SGDG
from modelopt.torch.quantization.rotation.learn import _polar_project
from modelopt.torch.quantization.rotation.sgdg import _qr_retraction


def _ortho_err(R):
    """max |R^T R - I| in float64."""
    Rd = R.detach().to(torch.float64)
    eye = torch.eye(Rd.shape[0], dtype=torch.float64)
    return (Rd.t() @ Rd - eye).abs().max().item()


def _seeded_orthogonal_fp32(n, seed):
    """Seeded fp32 orthogonal init (fp64 Haar QR cast down — same recipe as the trainer)."""
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    return q.to(torch.float32)


def _run_trajectory(
    sgdg_cls,
    momentum,
    steps=20,
    n=16,
    lr=0.4,
    grad_scale=1.0,
    param_seed=123,
    grad_seed=456,
    py_seed=789,
):
    """Drive one SGDG (ours or official) through a fully deterministic run.

    Identical seeds -> identical parameter init, identical gradient sequence, identical
    global-``random`` state (the stiefel branch consumes exactly one ``randint(1, 101)``
    per parameter per step, so equal ``py_seed`` gives equal retraction triggers).
    Returns (list of per-step parameter snapshots, final state momentum_buffer).
    """
    P = torch.nn.Parameter(_seeded_orthogonal_fp32(n, param_seed))
    opt = sgdg_cls([P], lr=lr, momentum=momentum, stiefel=True)
    gen = torch.Generator().manual_seed(grad_seed)
    random.seed(py_seed)
    traj = []
    for _ in range(steps):
        P.grad = grad_scale * torch.randn(n, n, generator=gen)
        opt.step()
        traj.append(P.detach().clone())
    buf = opt.state[P]["momentum_buffer"].detach().clone()
    return traj, buf


# --------------------------------------------------------------------------------------
# 1. 500-step orthogonality invariance under adversarial gradients
# --------------------------------------------------------------------------------------


def test_500_step_orthogonality_adversarial_grads():
    """500 SGDG steps with adversarial unit-scale randn gradients (far harsher than real
    CE gradients): |R^T R - I| < 1e-4 at EVERY step (fp32 Cayley drift + occasional
    stochastic QR resets), and the parameter moves far from its init."""
    n, steps = 32, 500
    P = torch.nn.Parameter(_seeded_orthogonal_fp32(n, 0))
    P0 = P.detach().clone()
    opt = SGDG([P], lr=1.5, momentum=0.0, stiefel=True)
    gen = torch.Generator().manual_seed(1)
    random.seed(0)
    worst = 0.0
    for step in range(steps):
        P.grad = torch.randn(n, n, generator=gen)
        opt.step()
        err = _ortho_err(P)
        worst = max(worst, err)
        assert err < 1e-4, f"step {step}: |R^T R - I| = {err:.3e} >= 1e-4"
    assert (P.detach() - P0).abs().max().item() > 1e-2, "parameter never moved"
    assert worst > 0.0  # fp32 iterates are never exactly on the manifold


# --------------------------------------------------------------------------------------
# 2. Step-cap edge: lr = 100 (alpha = min(t, lr) engages the t cap)
# --------------------------------------------------------------------------------------


def test_step_cap_lr100_no_blowup():
    """lr = 100 with unit-scale gradients: the Cayley step size is capped at
    t = 1/(||W||_1 + eps) << lr (verified by replicating the tangent construction for the
    first step), so 50 steps neither blow up nor leave the manifold."""
    n = 24
    P = torch.nn.Parameter(_seeded_orthogonal_fp32(n, 2))
    P0 = P.detach().clone()
    opt = SGDG([P], lr=100.0, momentum=0.0, stiefel=True)
    gen = torch.Generator().manual_seed(3)
    random.seed(1)
    for step in range(50):
        g = torch.randn(n, n, generator=gen)
        if step == 0:
            # Replicate the step's tangent construction: the cap must engage (t << lr).
            unity = P.detach() / P.detach().norm(p=2, dim=1, keepdim=True).add(1e-8)
            V = -g.t()
            MX = V @ unity
            XMX = unity @ MX
            XXMX = unity.t() @ XMX
            W_hat = MX - 0.5 * XXMX
            W = W_hat - W_hat.t()
            t = (0.5 * 2 / (W.abs().sum(dim=0).max() + 1e-8)).item()
            assert t < 100.0, f"cap never engages: t = {t:.3e} >= lr = 100"
            assert t < 1.0, f"expected a tight cap for unit grads, got t = {t:.3e}"
        P.grad = g
        opt.step()
        assert torch.isfinite(P).all(), f"step {step}: non-finite entries at lr=100"
        err = _ortho_err(P)
        assert err < 1e-4, f"step {step}: |R^T R - I| = {err:.3e} at lr=100"
    assert (P.detach() - P0).abs().max().item() > 1e-3, "parameter never moved at lr=100"


# --------------------------------------------------------------------------------------
# 3. Momentum is inert in the stiefel branch (our port, faithfully reproduced quirk)
# --------------------------------------------------------------------------------------


def test_momentum_inert_in_stiefel_branch():
    """momentum=0.0 and momentum=0.9 give BITWISE-identical trajectories in our port:
    the official dead store (V rebound to a temp before V.copy_(V_new)) keeps the state
    momentum_buffer at exact zeros forever, so the momentum hyperparameter cannot affect
    the stiefel update."""
    traj0, buf0 = _run_trajectory(SGDG, 0.0, steps=15, py_seed=11)
    traj9, buf9 = _run_trajectory(SGDG, 0.9, steps=15, py_seed=11)
    for step, (a, b) in enumerate(zip(traj0, traj9)):
        assert torch.equal(a, b), f"step {step}: momentum changed the stiefel trajectory"
    assert torch.all(buf0 == 0) and torch.all(buf9 == 0), "momentum_buffer was written"


# --------------------------------------------------------------------------------------
# 4. Forced QR-retraction branch (monkeypatched random.randint)
# --------------------------------------------------------------------------------------


def test_forced_qr_retraction_branch():
    """Force rand_num == 1 (QR retraction) vs rand_num != 1 (plain Cayley) on the same
    drifted init and gradient: the forced step re-orthogonalizes the iterate (the control
    keeps the drift — the Cayley step preserves, not restores, orthogonality) and the
    trajectory measurably changes."""
    n = 24
    torch.manual_seed(5)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    drifted = (q + 5e-3 * torch.randn(n, n, dtype=torch.float64)).to(torch.float32)
    g = torch.randn(n, n)
    results = {}
    orig_randint = random.randint
    for label, forced in (("control", 2), ("forced", 1)):
        P = torch.nn.Parameter(drifted.clone())
        opt = SGDG([P], lr=0.4, momentum=0.0, stiefel=True)
        random.randint = lambda a, b, _v=forced: _v  # deterministic branch selection
        try:
            P.grad = g.clone()
            opt.step()
        finally:
            random.randint = orig_randint
        results[label] = P.detach().clone()
    assert random.randint is orig_randint  # the monkeypatch really was restored
    err_forced = _ortho_err(results["forced"])
    err_control = _ortho_err(results["control"])
    assert err_forced < 1e-4, f"forced retraction left the iterate off-manifold: {err_forced:.3e}"
    assert err_control > 1e-4, (
        f"contrast broken: control step already orthogonal ({err_control:.3e}) — "
        "the drifted init did not drift"
    )
    diff = (results["forced"] - results["control"]).abs().max().item()
    assert not torch.equal(results["forced"], results["control"])
    assert diff > 1e-5, f"forcing the retraction barely changed the step ({diff:.3e})"


# --------------------------------------------------------------------------------------
# 5. Polar retraction: nearest-orthogonal sanity
# --------------------------------------------------------------------------------------


def test_polar_project_nearest_orthogonal():
    """For 20 random near-orthogonal drifted matrices, _polar_project returns a matrix
    that is (a) orthogonal to 1e-12 in BOTH residual forms, and (b) strictly closer to
    the input in Frobenius norm than any of 50 random orthogonal probes AND than the QR
    retraction of the same input (the polar factor is THE nearest orthogonal matrix)."""
    n = 24
    gen = torch.Generator().manual_seed(6)
    eye = torch.eye(n, dtype=torch.float64)
    for trial in range(20):
        q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
        A = q + 1e-2 * torch.randn(n, n, generator=gen, dtype=torch.float64)
        P = _polar_project(A)
        # (a) orthogonality at fp64-SVD level, both basis-dependent residual forms
        rtr = (P.t() @ P - eye).abs().max().item()
        rrt = (P @ P.t() - eye).abs().max().item()
        assert rtr < 1e-12, f"trial {trial}: |P^T P - I| = {rtr:.3e}"
        assert rrt < 1e-12, f"trial {trial}: |P P^T - I| = {rrt:.3e}"
        d_polar = (A - P).norm().item()
        assert d_polar > 0.0  # the input really was off the manifold
        # (b) vs the QR retraction of the same input (clone: _qr_retraction mutates
        # its argument in place via t_())
        Q_qr = _qr_retraction(A.clone())
        qr_ortho = (Q_qr.t() @ Q_qr - eye).abs().max().item()
        assert qr_ortho < 1e-10, f"trial {trial}: QR retraction not orthogonal"
        d_qr = (A - Q_qr).norm().item()
        assert d_polar < d_qr, (
            f"trial {trial}: polar ({d_polar:.6e}) not closer than QR ({d_qr:.6e})"
        )
        # (b) vs 50 random orthogonal probes
        for probe_i in range(50):
            probe, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
            d_probe = (A - probe).norm().item()
            assert d_polar < d_probe, (
                f"trial {trial} probe {probe_i}: polar ({d_polar:.6e}) not closer "
                f"than a random probe ({d_probe:.6e})"
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
