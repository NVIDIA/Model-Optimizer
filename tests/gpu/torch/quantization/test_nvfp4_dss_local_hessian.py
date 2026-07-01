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

"""Decoupled Scale Search (DSS) for NVFP4 static weights, targeted at the local-Hessian objective.

DSS decouples the FP4 quantization scale ``s_q`` (high precision) from the stored FP8
dequantization scale ``s_d``. For a *separable* (diagonal) metric this is a no-op — per-element
coupled rounding is already optimal — but for a *non-separable* (off-diagonal Hessian) metric it
can strictly reduce the block error. These tests pin both, plus the two-scale fake-quant primitive
and the decoupled export round-trip.
"""

import pytest
import torch
from conftest import requires_triton

from modelopt.torch.kernels.quantization.gemm.fp4_kernel import (
    compute_fp4_scales,
    static_blockwise_fp4_fake_quant,
)
from modelopt.torch.quantization.calib import NVFP4DSSCalibrator, NVFP4MSECalibrator
from modelopt.torch.quantization.calib.mse import default_dss_betas

BLOCK_SIZE = 16


def test_default_dss_betas_grid_and_guards():
    """The beta grid is centered on an exact 1.0, and absurd/None steps fail fast (not hang)."""
    betas = default_dss_betas(0.05)
    assert int((betas == 1.0).sum()) == 1  # exactly one coupled reference
    assert float(betas.min()) == pytest.approx(0.5) and float(betas.max()) == pytest.approx(1.5)
    # Out-of-range / None steps raise instead of building a giant grid or crashing on round(0.5/None).
    for bad in (None, 1e-6, 0.6, 0.0, -0.05):
        with pytest.raises(ValueError, match="dss_beta_step"):
            default_dss_betas(bad)


FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_fp4(a):
    lv = FP4_LEVELS.to(a.device)
    return lv[(a.unsqueeze(-1) - lv).abs().argmin(-1)]


# --------------------------------------------------------------------------------------------
# Two-scale fake-quant primitive
# --------------------------------------------------------------------------------------------
@requires_triton
def test_two_scale_fake_quant_matches_manual_reference():
    """``quant_amax`` rounds codes with ``s_q`` but reconstructs with the FP8 ``s_d``."""
    torch.manual_seed(0)
    x = torch.randn(128, BLOCK_SIZE, device="cuda")
    amax = x.abs().amax(-1)
    g = amax.max()
    quant_amax = amax * 1.3

    y = static_blockwise_fp4_fake_quant(x, amax, g, quant_amax=quant_amax)

    s_q = (quant_amax / 6.0).unsqueeze(-1)
    s_d = compute_fp4_scales(amax, g, True).unsqueeze(-1)  # FP8(amax/6)
    ref = torch.sign(x) * _round_fp4(x.abs() / s_q) * s_d
    assert torch.allclose(y, ref, atol=1e-5), f"maxdiff {(y - ref).abs().max().item():.3e}"


@requires_triton
def test_two_scale_fake_quant_beta_one_matches_coupled_for_fp8_exact_sd():
    """With an FP8-exact ``s_d`` and ``quant_amax == amax``, DSS reproduces the coupled result."""
    torch.manual_seed(1)
    x = torch.randn(64, BLOCK_SIZE, device="cuda")
    amax = x.abs().amax(-1)
    g = amax.max()
    # Snap amax so amax/6 is exactly FP8-representable (what the calibrated s_d candidates are).
    amax_exact = compute_fp4_scales(amax, g, True) * 6.0

    coupled = static_blockwise_fp4_fake_quant(x, amax_exact, g)
    dss = static_blockwise_fp4_fake_quant(x, amax_exact, g, quant_amax=amax_exact)
    assert torch.equal(coupled, dss)


# --------------------------------------------------------------------------------------------
# DSS calibrator: helps the non-separable metric, ties the separable one
# --------------------------------------------------------------------------------------------
def _dss_quant_func(global_amax):
    def quant_func(x, amax, quant_amax=None):
        return static_blockwise_fp4_fake_quant(
            x, amax, global_amax, True, x.dtype, quant_amax=quant_amax
        )

    return quant_func


def _hessian_error_func(hessian):
    """Per-block ``dwᵀ H dw`` expanded to element shape (row-sum == the quadratic form * bs)."""

    def error_func(x, xq):
        dw = (x - xq).to(torch.float32)  # [n_blocks, bs]
        qf = torch.einsum("nb,bd,nd->n", dw, hessian, dw)  # [n_blocks]
        return qf.unsqueeze(-1).expand(-1, dw.shape[-1])

    return error_func


def _achieved_loss(x, quant_func, amax, error_func, quant_amax=None):
    xq = quant_func(x, amax, quant_amax=quant_amax)
    return float(error_func(x, xq)[:, 0].sum())  # column 0 == per-block qf; sum over blocks


@requires_triton
@pytest.mark.parametrize("off_diagonal", [False, True])
def test_dss_calibrator_vs_coupled(off_diagonal):
    """DSS never loses to coupled; it strictly wins only for a non-separable (off-diagonal) H."""
    torch.manual_seed(7)
    device = "cuda"
    n_blocks = 512
    x = torch.randn(n_blocks, BLOCK_SIZE, device=device)
    per_block_amax = x.abs().amax(-1)
    global_amax = per_block_amax.max()
    quant_func = _dss_quant_func(global_amax)

    if off_diagonal:
        a = torch.randn(BLOCK_SIZE, BLOCK_SIZE, device=device)
        hessian = a @ a.t() + BLOCK_SIZE * torch.eye(BLOCK_SIZE, device=device)  # SPD, off-diagonal
    else:
        hessian = torch.diag(torch.rand(BLOCK_SIZE, device=device) + 0.5)  # SPD, diagonal
    error_func = _hessian_error_func(hessian)

    # Coupled baseline.
    coupled_cal = NVFP4MSECalibrator(
        amax=per_block_amax.clone(),
        axis=0,
        global_amax=global_amax,
        quant_func=quant_func,
        error_func=error_func,
    )
    coupled_cal.collect(x)
    coupled_amax = coupled_cal.compute_amax()
    coupled_loss = _achieved_loss(x, quant_func, coupled_amax, error_func)

    # DSS.
    dss_cal = NVFP4DSSCalibrator(
        amax=per_block_amax.clone(),
        axis=0,
        global_amax=global_amax,
        quant_func=quant_func,
        error_func=error_func,
        betas=default_dss_betas(0.05),
    )
    dss_cal.collect(x)
    dss_amax = dss_cal.compute_amax()
    dss_quant_amax = dss_cal.compute_quant_amax()
    dss_loss = _achieved_loss(x, quant_func, dss_amax, error_func, quant_amax=dss_quant_amax)

    assert dss_quant_amax is not None and dss_quant_amax.shape == dss_amax.shape
    # DSS is never worse than coupled (beta == 1 at the coupled optimum is always searched).
    assert dss_loss <= coupled_loss * (1 + 1e-5)

    if off_diagonal:
        # Non-separable metric: decoupling should strictly help, and pick s_q != s_d somewhere.
        assert dss_loss < coupled_loss, f"dss {dss_loss:.6f} !< coupled {coupled_loss:.6f}"
        assert not torch.allclose(dss_quant_amax, dss_amax)
    else:
        # Separable metric: coupled rounding is per-element optimal -> DSS ties it.
        assert dss_loss == pytest.approx(coupled_loss, rel=1e-5)


# --------------------------------------------------------------------------------------------
# Decoupled export round-trip (deploy parity)
# --------------------------------------------------------------------------------------------
@requires_triton
def test_decoupled_export_matches_fake_quant():
    """NVFP4 export with a quant scale reproduces the two-scale fake-quant reconstruction.

    The stored codes come from ``s_q`` and the stored scale is ``s_d``; dequantizing
    ``codes * s_d`` must equal ``static_blockwise_fp4_fake_quant(..., quant_amax=s_q*6)`` — so a
    standard NVFP4 runtime (which computes ``code * scale``) sees the DSS result.
    """
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    torch.manual_seed(3)
    cout, cin = 32, 128
    weight = torch.randn(cout, cin, device="cuda")
    n_blocks = cout * (cin // BLOCK_SIZE)
    per_block_amax = weight.reshape(n_blocks, BLOCK_SIZE).abs().amax(-1)
    global_amax = per_block_amax.max()
    # FP8-exact s_d (calibrated candidates are), high-precision s_q = 1.2 * s_d.
    amax_d = compute_fp4_scales(per_block_amax, global_amax, True).reshape(n_blocks) * 6.0
    quant_amax = amax_d * 1.2

    wsf2 = global_amax.float() / (6.0 * 448.0)
    wsf = (amax_d / 6.0 / wsf2).reshape(cout, cin // BLOCK_SIZE)  # FP8 units of s_d
    quant_scale = (quant_amax / 6.0).reshape(cout, cin // BLOCK_SIZE)  # absolute s_q

    qtensor, out_wsf, out_wsf2 = NVFP4QTensor.quantize(
        weight, BLOCK_SIZE, wsf, wsf2, quant_scaling_factor=quant_scale
    )
    dequant = qtensor.dequantize(
        dtype=torch.float32,
        scale=out_wsf,
        double_scale=out_wsf2,
        block_sizes={-1: BLOCK_SIZE},
    )

    fq = static_blockwise_fp4_fake_quant(
        weight, amax_d.reshape(cout, cin // BLOCK_SIZE), global_amax, quant_amax=quant_amax
    ).float()
    assert torch.allclose(dequant, fq, atol=1e-4), (
        f"maxdiff {(dequant - fq).abs().max().item():.3e}"
    )
