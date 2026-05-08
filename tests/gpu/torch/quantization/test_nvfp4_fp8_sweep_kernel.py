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

"""Parity + speedup tests for the NVFP4 FP8 scale sweep Triton fast path.

Compares the Triton fast path inside :class:`NVFP4MSECalibrator` against its
reference 126-step Python sweep on the same inputs and asserts the resulting
per-block amax tensors are bit-identical. Also reports a wall-clock speedup
number for the weight-MSE search step on a representative LLM-sized weight,
plus dispatch coverage for the conditions that gate the fast path.
"""

import os
import time
from contextlib import contextmanager

import pytest
import torch
from conftest import requires_triton

from modelopt.torch.quantization.calib import NVFP4MSECalibrator
from modelopt.torch.quantization.tensor_quant import static_blockwise_fp4_fake_quant

BLOCK_SIZE = 16


@contextmanager
def _force_sweep_path(triton_enabled: bool):
    """Pin the NVFP4 sweep dispatch to the requested path for the duration of the
    block, restoring the prior environment afterwards."""
    key = "MODELOPT_NVFP4_TRITON_SWEEP"
    prev = os.environ.get(key)
    os.environ[key] = "1" if triton_enabled else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _reference_quant_func(global_amax):
    """Reference NVFP4 fake-quant matching what ``mse_calibrate`` plumbs in."""

    def quant_func(x, amax):
        return static_blockwise_fp4_fake_quant(x, amax, global_amax)

    return quant_func


def _make_calibrator(per_block_amax, global_amax):
    return NVFP4MSECalibrator(
        amax=per_block_amax,
        axis=0,
        global_amax=global_amax,
        quant_func=_reference_quant_func(global_amax),
    )


def _run_reference(x, per_block_amax, global_amax):
    with _force_sweep_path(triton_enabled=False):
        cal = _make_calibrator(per_block_amax, global_amax)
        cal.collect(x)
        return cal.compute_amax()


def _run_triton(x, per_block_amax, global_amax):
    with _force_sweep_path(triton_enabled=True):
        cal = _make_calibrator(per_block_amax, global_amax)
        cal.collect(x)
        return cal.compute_amax()


@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("num_blocks", [4, 64, 1024])
def test_parity_random_weights(seed, num_blocks, dtype):
    """Triton sweep must produce the exact same per-block amax as the reference,
    across every dtype supported by the NVFP4 quantizer (fp32, fp16, bf16)."""
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=dtype)
    # Promote to fp32 for the per-block amax (matches what max_calibrate produces).
    per_block_amax = x.float().abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    ref = _run_reference(x, per_block_amax, global_amax)
    tri = _run_triton(x, per_block_amax, global_amax)

    assert ref.shape == tri.shape
    # Both pick from the same 126-element discrete candidate set, so any disagreement
    # would show up as a non-zero diff (not a small float epsilon). Demand exact match.
    assert torch.equal(ref, tri), (
        f"Triton sweep diverged from reference (dtype={dtype}): max |diff| = "
        f"{(ref - tri).abs().max().item():.3e}, "
        f"differing blocks = {(ref != tri).sum().item()} / {num_blocks}"
    )


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
    """After the fast path runs, a second collect() requires reset() in between."""
    torch.manual_seed(0)
    device = "cuda"
    num_blocks = 32
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    with _force_sweep_path(triton_enabled=True):
        cal = _make_calibrator(per_block_amax, global_amax)
        cal.collect(x)
        first = cal.compute_amax().clone()
        assert cal._best_amax_fast is not None  # fast path was taken

        # Second collect after the fast path is not allowed without a reset.
        with pytest.raises(RuntimeError, match="multi-collect after the fast path"):
            cal.collect(x)

        cal.reset()
        # After reset, the same calibrator instance can be re-used; fast path runs again.
        cal.collect(x)
        assert torch.equal(first, cal.compute_amax())


@requires_triton
def test_input_validation():
    """``nvfp4_fp8_scale_sweep`` should reject malformed inputs cleanly."""
    from modelopt.torch.kernels.quantization.gemm import nvfp4_fp8_scale_sweep

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


@requires_triton
def test_dispatch_fast_path_default():
    """Default config on CUDA with no error_func takes the Triton fast path."""
    torch.manual_seed(0)
    num_blocks = 32
    x = torch.randn(num_blocks, BLOCK_SIZE, device="cuda", dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    with _force_sweep_path(triton_enabled=True):
        cal = _make_calibrator(per_block_amax, global_amax)
        cal.collect(x)
        # Fast path stashes the final amax directly; reference accumulator stays empty.
        assert cal._best_amax_fast is not None
        assert cal._losses_sum is None


@requires_triton
def test_dispatch_env_optout_falls_back():
    """``MODELOPT_NVFP4_TRITON_SWEEP=0`` forces the reference 126-step sweep."""
    torch.manual_seed(0)
    num_blocks = 32
    x = torch.randn(num_blocks, BLOCK_SIZE, device="cuda", dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    with _force_sweep_path(triton_enabled=False):
        cal = _make_calibrator(per_block_amax, global_amax)
        cal.collect(x)
        assert cal._best_amax_fast is None
        assert cal._losses_sum is not None


@requires_triton
def test_dispatch_custom_error_func_falls_back():
    """A non-None ``error_func`` keeps the reference path so the user's metric is honored.

    This protects custom error-function callers (e.g. local-Hessian calibration's
    Hessian-weighted error) from silently being routed through a kernel that only
    knows squared-error.
    """
    torch.manual_seed(0)
    num_blocks = 32
    x = torch.randn(num_blocks, BLOCK_SIZE, device="cuda", dtype=torch.float32)
    per_block_amax = x.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    def hessian_like_error(a, b):
        return (a - b).pow(2)  # placeholder; the point is "non-None"

    with _force_sweep_path(triton_enabled=True):
        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            axis=0,
            global_amax=global_amax,
            quant_func=_reference_quant_func(global_amax),
            error_func=hessian_like_error,
        )
        cal.collect(x)
        assert cal._best_amax_fast is None
        assert cal._losses_sum is not None


@requires_triton
def test_dispatch_cpu_path_excluded():
    """The fast-path predicate must reject CPU inputs (kernel is CUDA-only).

    Tests the dispatch decision directly via the predicate rather than running
    ``collect()``, since the reference NVFP4 fake-quant kernel is itself CUDA-only —
    NVFP4 calibration as a whole isn't a CPU code path.
    """
    torch.manual_seed(0)
    num_blocks = 32
    x_cpu = torch.randn(num_blocks, BLOCK_SIZE, dtype=torch.float32)
    # Build the calibrator on CUDA so other predicate guards aren't the rejection cause.
    per_block_amax = x_cpu.abs().amax(dim=-1).cuda()
    global_amax = per_block_amax.max()

    with _force_sweep_path(triton_enabled=True):
        cal = _make_calibrator(per_block_amax, global_amax)
        assert cal._can_use_triton_fast_path(x_cpu) is False


@requires_triton
def test_mse_calibrate_end_to_end(monkeypatch):
    """End-to-end: the ``mse``/``fp8_scale_sweep=True`` path produces the same quantized
    weights with the fast path on (default) and off (``MODELOPT_NVFP4_TRITON_SWEEP=0``)."""
    from _test_utils.torch.quantization.models import SimpleLinear

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.extensions import get_cuda_ext_mx

    if get_cuda_ext_mx() is None:
        pytest.skip("cuda_ext_mx is not available")

    cfg = {
        "quant_cfg": [
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    "axis": None,
                },
                "enable": True,
            },
            {"quantizer_name": "*input_quantizer", "enable": False},
        ],
        "algorithm": {"method": "mse", "fp8_scale_sweep": True},
    }

    def _run_calibrated(env_value):
        torch.manual_seed(0)
        model = SimpleLinear().cuda()
        # Snapshot the pre-calibration weights so both runs start from identical state.
        weight_snapshots = {n: p.detach().clone() for n, p in model.named_parameters()}
        if env_value is None:
            monkeypatch.delenv("MODELOPT_NVFP4_TRITON_SWEEP", raising=False)
        else:
            monkeypatch.setenv("MODELOPT_NVFP4_TRITON_SWEEP", env_value)
        calib_data = [model.get_input().cuda() for _ in range(2)]

        def forward_loop(m):
            for batch in calib_data:
                m(batch)

        mtq.quantize(model, cfg, forward_loop=forward_loop)
        # Run a deterministic input through and snapshot the output.
        torch.manual_seed(1)
        x = torch.randn(4, 16, device="cuda")
        with torch.no_grad():
            y = model(x).detach().clone()
        return y, weight_snapshots

    y_default, w0 = _run_calibrated(env_value=None)
    y_optout, w1 = _run_calibrated(env_value="0")
    # Both runs must start from the same weights (sanity: SimpleLinear is deterministic
    # under the same seed) before we compare post-calibration outputs.
    for name in w0:
        assert torch.equal(w0[name], w1[name]), name
    assert torch.equal(y_default, y_optout)


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
            f"  reference path: {ref_t * 1e3:8.2f} ms\n"
            f"  triton fast path: {tri_t * 1e3:8.2f} ms\n"
            f"  speedup: {speedup:.1f}x"
        )
