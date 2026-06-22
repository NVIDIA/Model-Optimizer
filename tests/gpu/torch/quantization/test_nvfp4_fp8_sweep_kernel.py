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
from _test_utils.torch.quantization.models import SimpleLinear
from conftest import requires_triton

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.kernels.quantization.gemm import (
    nvfp4_fp8_scale_sweep,
    nvfp4_fp8_scale_sweep_hessian,
)
from modelopt.torch.quantization.calib import NVFP4MSECalibrator
from modelopt.torch.quantization.extensions import get_cuda_ext_mx
from modelopt.torch.quantization.model_calib import _LocalHessianAccumulator
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import FP8_E4M3_MAX
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


def _nvfp4_static_amax_dtypes(model):
    amax_dtypes = {}
    for name, module in model.named_modules():
        if (
            isinstance(module, TensorQuantizer)
            and module.is_nvfp4_static
            and module.amax is not None
        ):
            amax_dtypes[name] = module.amax.dtype
    return amax_dtypes


def _assert_nvfp4_static_amaxes_fp32(amax_dtypes, model_dtype, label):
    assert amax_dtypes, f"{label}: expected NVFP4 static amaxes for model dtype {model_dtype}"
    assert all(amax_dtype == torch.float32 for amax_dtype in amax_dtypes.values()), (
        f"{label}: expected all NVFP4 static amaxes to be fp32 for model dtype {model_dtype}, "
        f"got {amax_dtypes}"
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
@pytest.mark.parametrize("num_blocks", [4, 1024])
@pytest.mark.parametrize("seed", [0, 1])
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
@pytest.mark.parametrize("triton_enabled", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sweep_stores_fp32_amax_and_preserves_output_dtype(dtype, triton_enabled):
    """NVFP4 MSE stores selected amax in fp32 without changing fake-quant output dtype."""
    torch.manual_seed(11)
    device = "cuda"
    num_blocks = 64
    x = torch.randn(num_blocks, BLOCK_SIZE, device=device, dtype=dtype)
    per_block_amax = x.abs().amax(dim=-1).to(dtype=dtype)
    global_amax = per_block_amax.max()

    with _force_sweep_path(triton_enabled=triton_enabled):
        cal = _make_calibrator(per_block_amax, global_amax)
        cal.collect(x)
        amax = cal.compute_amax()

    assert amax.dtype == torch.float32
    xq = static_blockwise_fp4_fake_quant(x, amax, global_amax, True, FP8_E4M3_MAX, x.dtype)
    assert xq.dtype == x.dtype


@requires_triton
def test_reset_allows_recollect():
    """After collect() caches an amax, a second collect() requires reset() in between."""
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
        assert cal._best_amax is not None  # final amax was cached

        # Second collect after a final amax is cached is not allowed without a reset.
        with pytest.raises(RuntimeError, match="multi-collect"):
            cal.collect(x)

        cal.reset()
        # After reset, the same calibrator instance can be re-used; fast path runs again.
        cal.collect(x)
        assert torch.equal(first, cal.compute_amax())


@requires_triton
def test_input_validation():
    """``nvfp4_fp8_scale_sweep`` should reject malformed inputs cleanly."""
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
        assert cal._best_amax is not None
        assert cal._losses_sum is None


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
        assert cal._best_amax is not None
        assert cal._losses_sum is None


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mse_calibrate_end_to_end(monkeypatch, tmp_path, dtype):
    """End-to-end: the ``mse``/``fp8_scale_sweep=True`` path produces the same quantized
    weights with the fast path on (default) and off (``MODELOPT_NVFP4_TRITON_SWEEP=0``),
    and stores/restores NVFP4 static amax in fp32 for fp32 and bf16 model forwards."""
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

    def _run_calibrated(env_value, label):
        torch.manual_seed(0)
        model = SimpleLinear(dtype=dtype).cuda()
        # Snapshot the pre-calibration weights so both runs start from identical state.
        weight_snapshots = {n: p.detach().clone() for n, p in model.named_parameters()}
        if env_value is None:
            monkeypatch.delenv("MODELOPT_NVFP4_TRITON_SWEEP", raising=False)
        else:
            monkeypatch.setenv("MODELOPT_NVFP4_TRITON_SWEEP", env_value)
        calib_data = [model.get_input().cuda().to(dtype=dtype) for _ in range(2)]

        def forward_loop(m):
            for batch in calib_data:
                m(batch)

        mtq.quantize(model, cfg, forward_loop=forward_loop)
        amax_dtypes = _nvfp4_static_amax_dtypes(model)
        _assert_nvfp4_static_amaxes_fp32(amax_dtypes, dtype, label)

        ckpt_path = tmp_path / f"mse_calibrate_{label}_{str(dtype).rpartition('.')[-1]}.pt"
        mto.save(model, ckpt_path)
        restored_model = mto.restore(SimpleLinear(dtype=dtype).cuda(), ckpt_path)
        restored_amax_dtypes = _nvfp4_static_amax_dtypes(restored_model)
        _assert_nvfp4_static_amaxes_fp32(restored_amax_dtypes, dtype, f"{label} restored")

        # Run a deterministic input through and snapshot the output.
        torch.manual_seed(1)
        x = torch.randn(4, 16, device="cuda", dtype=dtype)
        with torch.no_grad():
            y = model(x).detach().clone()
            y_restored = restored_model(x).detach().clone()
        assert y_restored.dtype == dtype
        assert torch.equal(y, y_restored)
        return y, weight_snapshots, amax_dtypes, restored_amax_dtypes

    y_default, w0, dtypes_default, restored_dtypes_default = _run_calibrated(
        env_value=None, label="fast"
    )
    y_optout, w1, dtypes_optout, restored_dtypes_optout = _run_calibrated(
        env_value="0", label="reference"
    )
    # Both runs must start from the same weights (sanity: SimpleLinear is deterministic
    # under the same seed) before we compare post-calibration outputs.
    for name in w0:
        assert torch.equal(w0[name], w1[name]), name
    _assert_nvfp4_static_amaxes_fp32(dtypes_default, dtype, "fast")
    _assert_nvfp4_static_amaxes_fp32(dtypes_optout, dtype, "reference")
    _assert_nvfp4_static_amaxes_fp32(restored_dtypes_default, dtype, "fast restored")
    _assert_nvfp4_static_amaxes_fp32(restored_dtypes_optout, dtype, "reference restored")
    assert y_default.dtype == dtype
    assert y_optout.dtype == dtype
    assert torch.equal(y_default, y_optout)


# --------------------------------------------------------------------------------------
# Hessian-weighted sweep (local_hessian fast path)
# --------------------------------------------------------------------------------------


def _build_hessian_accumulator(cout, cin, hessian_input, block_size=BLOCK_SIZE):
    """Real ``_LocalHessianAccumulator`` so the test exercises the production metric."""
    acc = _LocalHessianAccumulator(cout, cin, block_size)
    acc.accumulate(hessian_input)
    return acc


def _run_hessian_reference(x_blocks, per_block_amax, global_amax, acc):
    """Reference 126-step sweep using the Hessian-weighted ``error_func`` (Triton off)."""
    with _force_sweep_path(triton_enabled=False):
        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            axis=0,
            global_amax=global_amax,
            quant_func=_reference_quant_func(global_amax),
            error_func=acc.build_error_func(keep_buffer=True),
        )
        cal.collect(x_blocks)
        return cal.compute_amax()


def _run_hessian_triton(x_blocks, per_block_amax, global_amax, acc):
    """Hessian-weighted Triton fast path (same metric as a raw per-cin-block tensor)."""
    with _force_sweep_path(triton_enabled=True):
        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            axis=0,
            global_amax=global_amax,
            quant_func=_reference_quant_func(global_amax),
            hessian=acc.normalized_hessian(),
        )
        cal.collect(x_blocks)
        return cal.compute_amax()


def _total_hessian_loss(x_blocks, per_block_amax, global_amax, hessian):
    """Total Hessian-weighted quantization error ``Σ dwᵀ H dw`` under the production
    (CUDA ``static_blockwise_fp4_fake_quant``) rounding used at deployment — the objective
    the sweep minimizes, summed over all blocks."""
    n_blocks = x_blocks.shape[0]
    n_cin = hessian.shape[0]
    h_per_block = hessian[torch.arange(n_blocks, device=x_blocks.device) % n_cin]
    xq = static_blockwise_fp4_fake_quant(x_blocks.float(), per_block_amax, global_amax)
    dw = x_blocks.float() - xq
    return (torch.einsum("nij,nj->ni", h_per_block, dw) * dw).sum()


@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("cout", "cin"),
    [(8, 64), (1, 256), (256, 2048)],  # multiple rows share a cin-block Hessian; MoE-expert-sized
)
@pytest.mark.parametrize("seed", [0, 1])
def test_hessian_parity_random_weights(seed, cout, cin, dtype):
    """The Hessian Triton sweep must select the same per-block scales as the reference
    Hessian-weighted 126-step sweep, exercising shapes where many output rows share one
    per-cin-block Hessian (the ``b % n_cin_blocks`` mapping).

    The kernel quantizes candidates with the SAME FP8-E4M3-quantized block scale the
    reference uses (precomputed via ``compute_fp4_scales``), so the per-block residual is
    bit-identical. fp32/fp16 are therefore bit-exact here; for bf16 only the occasional
    block whose two best candidates are within fp32 noise may flip (``tl.dot`` vs ``einsum``
    accumulation order), so we cap the mismatch fraction tightly and require the total
    objective (Hessian loss) to be essentially unchanged.
    """
    torch.manual_seed(seed)
    device = "cuda"
    weight = torch.randn(cout, cin, device=device, dtype=dtype)
    # Well-conditioned PSD Hessian from many tokens so the argmin is unambiguous.
    hessian_input = torch.randn(512, cin, device=device, dtype=torch.float32)
    acc = _build_hessian_accumulator(cout, cin, hessian_input)

    x_blocks = weight.reshape(-1, BLOCK_SIZE)
    per_block_amax = x_blocks.float().abs().amax(dim=-1)
    global_amax = per_block_amax.max()
    hessian = acc.normalized_hessian()

    ref = _run_hessian_reference(x_blocks, per_block_amax, global_amax, acc)
    tri = _run_hessian_triton(x_blocks, per_block_amax, global_amax, acc)

    assert ref.shape == tri.shape
    n_blocks = ref.numel()
    n_diff = int((ref != tri).sum())

    if dtype != torch.bfloat16:
        # Matching scales + matching FP4 rounding => bit-exact for fp32/fp16.
        assert torch.equal(ref, tri), (
            f"{dtype} must be bit-exact: {n_diff}/{n_blocks} blocks differ"
        )
        return

    # bf16: only rare fp32 reduction-order ties may flip, and the total objective the kernel
    # achieves must match the reference's to within fp32 noise.
    assert n_diff / n_blocks < 1e-3, f"{n_diff}/{n_blocks} blocks differ (>0.1%)"
    loss_ref = _total_hessian_loss(x_blocks, ref, global_amax, hessian)
    loss_tri = _total_hessian_loss(x_blocks, tri, global_amax, hessian)
    rel_gap = ((loss_tri - loss_ref) / loss_ref.abs().clamp_min(1e-12)).abs().item()
    assert rel_gap < 1e-3, (
        f"aggregate Hessian-loss gap {rel_gap:.3e} too large "
        f"({n_diff}/{n_blocks} boundary blocks flipped, dtype={dtype})"
    )


@requires_triton
def test_hessian_sweep_input_validation():
    """``nvfp4_fp8_scale_sweep_hessian`` should reject malformed inputs cleanly."""
    device = "cuda"
    cout, cin = 4, 64
    x = torch.randn(cout, cin, device=device).reshape(-1, BLOCK_SIZE)
    g = x.abs().amax()
    h = torch.randn(cin // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, device=device)

    with pytest.raises(ValueError, match="CUDA"):
        nvfp4_fp8_scale_sweep_hessian(x.cpu(), g.cpu(), h.cpu())
    with pytest.raises(ValueError, match="block_size"):
        nvfp4_fp8_scale_sweep_hessian(x, g, h, block_size=0)
    # Wrong Hessian block dims.
    with pytest.raises(ValueError, match="hessian must have shape"):
        nvfp4_fp8_scale_sweep_hessian(x, g, torch.randn(4, 8, 8, device=device))


@requires_triton
def test_hessian_speedup_report(capsys):
    """Report the Hessian fast-path speedup on a representative 8192x4096 weight (~2M NVFP4
    blocks); expect ~30x on A6000, higher on datacenter GPUs. Mirrors ``test_speedup_report`"""
    torch.manual_seed(123)
    device = "cuda"
    cout, cin = 8192, 4096
    weight = torch.randn(cout, cin, device=device, dtype=torch.float32)
    hessian_input = torch.randn(512, cin, device=device, dtype=torch.float32)
    acc = _build_hessian_accumulator(cout, cin, hessian_input)

    x_blocks = weight.reshape(-1, BLOCK_SIZE)
    per_block_amax = x_blocks.abs().amax(dim=-1)
    global_amax = per_block_amax.max()

    ref_amax = _run_hessian_reference(x_blocks, per_block_amax, global_amax, acc)
    tri_amax = _run_hessian_triton(x_blocks, per_block_amax, global_amax, acc)
    n_blocks = ref_amax.numel()
    n_diff = int((ref_amax != tri_amax).sum())
    # fp32 weights are bit-exact; any divergence (only on FP4-boundary blocks for lower
    # precision) must leave the total objective essentially unchanged.
    hessian = acc.normalized_hessian()
    loss_ref = _total_hessian_loss(x_blocks, ref_amax, global_amax, hessian)
    loss_tri = _total_hessian_loss(x_blocks, tri_amax, global_amax, hessian)
    rel_gap = ((loss_tri - loss_ref) / loss_ref.abs().clamp_min(1e-12)).abs().item()
    assert rel_gap < 1e-3, (
        f"{n_diff}/{n_blocks} blocks disagree, aggregate Hessian-loss gap {rel_gap:.3e}"
    )

    # Reference Hessian sweep is seconds-slow (126 einsums over a 2M-block weight), so use
    # fewer iters; the Triton path is sub-ms and gets the default count.
    ref_t = _bench(
        lambda: _run_hessian_reference(x_blocks, per_block_amax, global_amax, acc),
        warmup=1,
        iters=2,
    )
    tri_t = _bench(lambda: _run_hessian_triton(x_blocks, per_block_amax, global_amax, acc))
    speedup = ref_t / tri_t

    with capsys.disabled():
        print(
            f"\n[NVFP4 Hessian FP8 sweep] weight=({cout},{cin}) "
            f"n_blocks={n_blocks} block_size={BLOCK_SIZE} mismatched_blocks={n_diff}\n"
            f"  reference path: {ref_t * 1e3:8.2f} ms\n"
            f"  triton fast path: {tri_t * 1e3:8.2f} ms\n"
            f"  speedup: {speedup:.1f}x"
        )


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
