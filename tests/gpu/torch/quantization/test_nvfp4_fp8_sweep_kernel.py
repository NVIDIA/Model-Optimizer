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

"""Parity + speedup tests for the fused NVFP4 FP8 scale sweep Triton kernel.

Compares :class:`TritonNVFP4MSECalibrator` against the reference
:class:`NVFP4MSECalibrator` on the same inputs and asserts the resulting per-block
amax tensors are bit-identical. Also reports a wall-clock speedup number for the
weight-MSE search step on a representative LLM-sized weight.
"""

import time

import pytest
import torch
from conftest import requires_triton

from modelopt.torch.quantization.calib import NVFP4MSECalibrator, TritonNVFP4MSECalibrator
from modelopt.torch.quantization.tensor_quant import static_blockwise_fp4_fake_quant

BLOCK_SIZE = 16


def _reference_quant_func(global_amax):
    """Reference NVFP4 fake-quant matching what ``mse_calibrate`` plumbs in."""

    def quant_func(x, amax):
        return static_blockwise_fp4_fake_quant(x, amax, global_amax)

    return quant_func


def _run_reference(x, per_block_amax, global_amax):
    cal = NVFP4MSECalibrator(
        amax=per_block_amax,
        axis=0,
        global_amax=global_amax,
        quant_func=_reference_quant_func(global_amax),
    )
    cal.collect(x)
    return cal.compute_amax()


def _run_triton(x, per_block_amax, global_amax):
    cal = TritonNVFP4MSECalibrator(
        amax=per_block_amax,
        axis=0,
        global_amax=global_amax,
        quant_func=_reference_quant_func(global_amax),
    )
    cal.collect(x)
    return cal.compute_amax()


@requires_triton
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("num_blocks", [4, 64, 1024])
def test_parity_random_weights(seed, num_blocks):
    """Triton sweep must produce the exact same per-block amax as the reference."""
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    ref = _run_reference(x, per_block_amax, global_amax)
    tri = _run_triton(x, per_block_amax, global_amax)

    assert ref.shape == tri.shape
    # Both pick from the same 126-element discrete candidate set, so any disagreement
    # would show up as a non-zero diff (not a small float epsilon). Demand exact match.
    assert torch.equal(ref, tri), (
        f"Triton sweep diverged from reference: max |diff| = "
        f"{(ref - tri).abs().max().item():.3e}, "
        f"differing blocks = {(ref != tri).sum().item()} / {num_blocks}"
    )


@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_parity_dtypes(dtype):
    """Sweep must agree across the dtypes supported by the NVFP4 quantizer."""
    torch.manual_seed(42)
    device = "cuda"
    num_blocks = 256
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=dtype)
    # Promote to fp32 for the per-block amax (matches what max_calibrate produces).
    per_block_amax = x.float().abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    ref = _run_reference(x, per_block_amax, global_amax)
    tri = _run_triton(x, per_block_amax, global_amax)
    assert torch.equal(ref, tri)


@requires_triton
def test_quantized_output_matches():
    """Round-tripping x through the chosen amax should give the same fake-quant result."""
    torch.manual_seed(7)
    device = "cuda"
    num_blocks = 128
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    ref_amax = _run_reference(x, per_block_amax, global_amax)
    tri_amax = _run_triton(x, per_block_amax, global_amax)

    ref_xq = static_blockwise_fp4_fake_quant(x, ref_amax, global_amax)
    tri_xq = static_blockwise_fp4_fake_quant(x, tri_amax, global_amax)
    assert torch.equal(ref_xq, tri_xq)


@requires_triton
def test_reset_allows_recollect():
    torch.manual_seed(0)
    device = "cuda"
    num_blocks = 32
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    cal = TritonNVFP4MSECalibrator(
        amax=per_block_amax,
        axis=0,
        global_amax=global_amax,
    )
    cal.collect(x)
    first = cal.compute_amax().clone()

    # collect() is one-shot per cycle until reset() is called.
    with pytest.raises(RuntimeError, match="one-shot"):
        cal.collect(x)

    cal.reset()
    # After reset, the same calibrator instance can be re-used.
    cal.collect(x)
    assert torch.equal(first, cal.compute_amax())


@requires_triton
def test_input_validation():
    """``nvfp4_fp8_scale_sweep`` should reject malformed inputs cleanly."""
    from modelopt.torch.kernels.quantization.gemm import fp8_scale_candidates, nvfp4_fp8_scale_sweep

    device = "cuda"
    x = torch.randn(64, BLOCK_SIZE, device=device)
    g = x.abs().amax()

    # CPU tensor → ValueError (not bare AssertionError).
    with pytest.raises(ValueError, match="CUDA"):
        nvfp4_fp8_scale_sweep(x.cpu(), g.cpu())

    # block_size <= 0.
    with pytest.raises(ValueError, match="block_size"):
        nvfp4_fp8_scale_sweep(x, g, block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        nvfp4_fp8_scale_sweep(x, g, block_size=-1)

    # Non-divisible numel.
    with pytest.raises(ValueError, match="not divisible"):
        nvfp4_fp8_scale_sweep(x, g, block_size=15)

    # Empty / wrong-rank candidates.
    with pytest.raises(ValueError, match="non-empty 1-D"):
        nvfp4_fp8_scale_sweep(x, g, candidates=torch.empty(0, device=device))
    with pytest.raises(ValueError, match="non-empty 1-D"):
        nvfp4_fp8_scale_sweep(x, g, candidates=fp8_scale_candidates(device).reshape(2, -1))


def _bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


@requires_triton
def test_speedup_report(capsys):
    """Sanity-check that the Triton path is meaningfully faster on a realistic weight.

    Uses an 8192 x 4096 weight (~33M elements, ~2M NVFP4 blocks) — roughly the size
    of an LLM attention/MLP projection. Reports the speedup; does not gate on a
    minimum factor (kernel timing is noisy on shared CI), but does require parity
    on the chosen amax.
    """
    torch.manual_seed(123)
    device = "cuda"
    cout, cin = 8192, 4096
    x = torch.randn(cout, cin // BLOCK_SIZE, BLOCK_SIZE, device=device, dtype=torch.float32)
    x = x.reshape(-1, BLOCK_SIZE)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    ref_amax = _run_reference(x, per_block_amax, global_amax)
    tri_amax = _run_triton(x, per_block_amax, global_amax)
    # Bit-equality across millions of blocks isn't guaranteed: when two adjacent FP8
    # candidates yield near-identical per-block MSE (within fp32 noise), the reference's
    # CUDA fake_e4m3fy path and our Triton inline math can break ties differently. Demand
    # instead that the Triton choice produces a per-block MSE within fp32 epsilon of the
    # reference's choice.
    n_blocks = ref_amax.numel()
    n_diff = int((ref_amax != tri_amax).sum())
    if n_diff:
        ref_xq = static_blockwise_fp4_fake_quant(x, ref_amax, global_amax)
        tri_xq = static_blockwise_fp4_fake_quant(x, tri_amax, global_amax)
        per_block_mse_ref = (x - ref_xq).pow(2).sum(dim=-1)
        per_block_mse_tri = (x - tri_xq).pow(2).sum(dim=-1)
        # Reference is the formal argmin, so triton's loss should be ≥ reference's.
        # Allow at most 1e-5 relative gap on differing blocks (observed ~1e-7 in practice).
        rel_gap = (per_block_mse_tri - per_block_mse_ref).abs() / per_block_mse_ref.clamp_min(1e-12)
        worst = rel_gap.max().item()
        assert worst < 1e-5, (
            f"{n_diff}/{n_blocks} blocks disagree with worst relative MSE gap {worst:.3e} "
            "— exceeds tie-break tolerance"
        )

    ref_t = _bench(lambda: _run_reference(x, per_block_amax, global_amax))
    tri_t = _bench(lambda: _run_triton(x, per_block_amax, global_amax))
    speedup = ref_t / tri_t

    # Force-print regardless of pytest capture mode.
    with capsys.disabled():
        n_blocks = x.numel() // BLOCK_SIZE
        print(
            f"\n[NVFP4 FP8 sweep] weight=({cout},{cin}) "
            f"n_blocks={n_blocks} block_size={BLOCK_SIZE}\n"
            f"  reference NVFP4MSECalibrator: {ref_t * 1e3:8.2f} ms\n"
            f"  triton TritonNVFP4MSECalibrator: {tri_t * 1e3:8.2f} ms\n"
            f"  speedup: {speedup:.1f}x"
        )
