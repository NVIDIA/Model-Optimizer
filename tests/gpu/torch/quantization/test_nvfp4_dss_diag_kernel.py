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

"""Parity probe for Decoupled Scale Search (DSS) on the NVFP4 weight-MSE FP8 sweep.

DSS decouples the FP4 quantization scale ``s_q`` from the FP8-constrained dequantization
scale ``s_d``. For the plain per-block weight-L2 objective, the existing sweep already
enumerates every FP8 ``s_d`` with its per-element-optimal coupled assignment, so DSS cannot
reduce the loss: for any fixed ``s_d`` the best ``s_q`` is ``s_d`` itself (``beta == 1``).

These tests pin that proof: ``dss_loss`` must never beat ``coupled_loss`` by more than fp32
noise, on both random weights and the SOAR-style regression example where the *continuous*
optimal scale falls between two FP8 values.
"""

import pytest
import torch
from conftest import requires_triton

from modelopt.torch.kernels.quantization.gemm import default_dss_betas, nvfp4_dss_diag_sweep

BLOCK_SIZE = 16


def _rel_improvement(coupled_loss, dss_loss):
    """Relative loss reduction DSS achieves over coupling, per block (>= 0 by construction)."""
    return (coupled_loss - dss_loss) / coupled_loss.clamp_min(1e-12)


@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_blocks", [4, 4096])
@pytest.mark.parametrize("seed", [0, 1])
def test_dss_is_parity_on_random_weights(seed, num_blocks, dtype):
    """On random weights, DSS must not beat the coupled sweep beyond fp32 tie-break noise."""
    torch.manual_seed(seed)
    x = torch.randn(num_blocks, BLOCK_SIZE, device="cuda", dtype=dtype)
    global_amax = x.float().abs().max()
    betas = default_dss_betas("cuda")

    coupled_loss, dss_loss, best_beta = nvfp4_dss_diag_sweep(x, global_amax, betas, BLOCK_SIZE)

    # By construction (beta == 1 is in the grid) DSS is never worse.
    assert torch.all(dss_loss <= coupled_loss + 1e-6)
    # The substantive claim: the improvement is fp-noise, not a real gain.
    worst = _rel_improvement(coupled_loss, dss_loss).max().item()
    assert worst < 1e-5, (
        f"DSS beat coupling by relative {worst:.3e} on random weights — exceeds tie-break "
        f"noise; the L2-parity proof would be violated. beta!=1 on "
        f"{int((best_beta != 1.0).sum())}/{num_blocks} blocks."
    )


@requires_triton
def test_dss_parity_on_soar_regression_example():
    """SOAR's intuition case: block whose continuous-optimal scale lies between FP8 values.

    Even here DSS reproduces the coupled loss, because the dequant scale must still snap to
    FP8 and coupling at that FP8 scale already yields the per-element-optimal codes.
    """
    # [1, 1, 1, 1.8]-style block padded to BLOCK_SIZE; mild magnitude spread.
    block = torch.tensor([1.0, 1.0, 1.0, 1.8] * 4, device="cuda", dtype=torch.float32)
    x = block.view(1, BLOCK_SIZE)
    global_amax = x.abs().max()
    betas = default_dss_betas("cuda")

    coupled_loss, dss_loss, _ = nvfp4_dss_diag_sweep(x, global_amax, betas, BLOCK_SIZE)

    assert dss_loss.item() <= coupled_loss.item() + 1e-6
    assert _rel_improvement(coupled_loss, dss_loss).item() < 1e-5


@requires_triton
def test_coupled_loss_matches_reference_sweep_amax():
    """``coupled_loss`` (beta == 1 slice) must equal the loss of the existing sweep's amax.

    Ties the diagnostic's coupled baseline to ``nvfp4_fp8_scale_sweep``'s chosen per-block
    amax, so the reported gap is genuinely DSS-vs-today, not two different baselines.
    """
    from modelopt.torch.kernels.quantization.gemm import nvfp4_fp8_scale_sweep
    from modelopt.torch.quantization.tensor_quant import static_blockwise_fp4_fake_quant

    torch.manual_seed(3)
    num_blocks = 256
    x = torch.randn(num_blocks, BLOCK_SIZE, device="cuda", dtype=torch.float32)
    global_amax = x.abs().max()
    betas = default_dss_betas("cuda")

    coupled_loss, _, _ = nvfp4_dss_diag_sweep(x, global_amax, betas, BLOCK_SIZE)

    sweep_amax = nvfp4_fp8_scale_sweep(x, global_amax, block_size=BLOCK_SIZE).reshape(num_blocks)
    xq = static_blockwise_fp4_fake_quant(x, sweep_amax, global_amax)
    ref_loss = (x - xq).pow(2).sum(dim=-1)

    rel = (coupled_loss - ref_loss).abs() / ref_loss.clamp_min(1e-12)
    assert rel.max().item() < 1e-5, f"coupled_loss diverged from sweep amax loss: {rel.max():.3e}"


@requires_triton
def test_betas_must_contain_exactly_one_unit():
    """The coupled reference requires exactly one beta == 1.0."""
    x = torch.randn(8, BLOCK_SIZE, device="cuda")
    g = x.abs().max()

    with pytest.raises(ValueError, match="exactly one"):
        nvfp4_dss_diag_sweep(x, g, torch.tensor([0.9, 1.1], device="cuda"), BLOCK_SIZE)
