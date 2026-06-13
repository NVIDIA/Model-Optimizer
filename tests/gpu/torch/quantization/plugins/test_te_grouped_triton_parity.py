# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OMNIML-5072 AC2 — ATriton vs A parity on the N-modules per-expert path.

ATriton = A's `_per_expert_weight_quantizer == True` path with the Triton
fakequant dispatch added in OMNIML-5072 (see
`modelopt/torch/quantization/plugins/transformer_engine.py`).

A = the same N-modules path but going through `FakeTensorQuantFunction.apply`
per expert (cuda_ext under the hood).

Two checks at each shape:

  forward parity: max_abs_err <= 1 ULP. Known rounding-mode mismatch — Triton
    rounds via `libdevice.rint` while cuda_ext rounds via its own builtin;
    both are banker's rounding but disagree on one ULP at a fraction of
    bf16 boundary values.

  backward parity (pass_through_bwd=True): bit-exact (max_abs_err == 0.0).
    Under modelopt's default pass-through STE, both paths return grad_out
    unchanged regardless of the forward kernel — so gradient identity is
    required, not approximate.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ATriton parity test requires a CUDA GPU",
)


def _parity_atriton_vs_a(
    num_experts: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
):
    """Run both paths at a given shape; return (fwd_max_err, ulp_floor, bwd_max_err)."""
    from modelopt.torch.kernels.quantization.gemm import (
        IS_AVAILABLE,
        grouped_axis0_fakequant,
    )
    from modelopt.torch.quantization.extensions import get_cuda_ext
    from modelopt.torch.quantization.plugins.transformer_engine import (
        _GroupedAxis0FakeQuantFn,
    )
    from modelopt.torch.quantization.tensor_quant import FakeTensorQuantFunction

    if not IS_AVAILABLE:
        pytest.skip("triton kernels not loaded — IS_AVAILABLE is False")

    cuda_ext = get_cuda_ext()
    device = "cuda"

    torch.manual_seed(0)
    weights = [
        torch.randn(out_features, in_features, dtype=dtype, device=device)
        for _ in range(num_experts)
    ]
    amax_scalars = [w.abs().amax().to(torch.float32) for w in weights]
    amax_vec = torch.stack([a.view(1) for a in amax_scalars]).view(num_experts, 1, 1)

    # ---- forward parity ----
    q_a = [
        cuda_ext.fake_tensor_quant(w, a, 8, False, True)
        for w, a in zip(weights, amax_scalars)
    ]
    q_t = grouped_axis0_fakequant(weights, amax_vec, num_bits=8, narrow_range=True)

    fwd_max_err = 0.0
    for i in range(num_experts):
        diff = (q_a[i].float() - q_t[i].float()).abs()
        fwd_max_err = max(fwd_max_err, float(diff.max().item()))

    # 1 ULP at this quant scale is amax / 127 (narrow_range, 8-bit).
    ulp_floor = float(amax_vec.max().item() / 127.0)

    # ---- backward parity under pass_through_bwd=True ----
    # Use a sum-loss (no GEMM) so any divergence is from the quantizer's
    # autograd wrapping, not GEMM determinism.
    ws_a = [w.detach().clone().requires_grad_(True) for w in weights]
    # FakeTensorQuantFunction signature:
    # (inputs, amax, bias=None, num_bits, unsigned, narrow_range,
    #  trt_high_precision_dtype, pass_through_bwd, block_size, axis)
    qs_a = [
        FakeTensorQuantFunction.apply(w, a, None, 8, False, True, None, True, None, None)
        for w, a in zip(ws_a, amax_scalars)
    ]
    loss_a = sum(q.sum() for q in qs_a)
    loss_a.backward()

    ws_t = [w.detach().clone().requires_grad_(True) for w in weights]
    # _GroupedAxis0FakeQuantFn.apply(amax_vec, num_bits, narrow_range, pass_through_bwd, *weights)
    qs_t = _GroupedAxis0FakeQuantFn.apply(amax_vec, 8, True, True, *ws_t)
    loss_t = sum(q.sum() for q in qs_t)
    loss_t.backward()

    bwd_max_err = 0.0
    for i in range(num_experts):
        diff = (ws_a[i].grad.float() - ws_t[i].grad.float()).abs()
        bwd_max_err = max(bwd_max_err, float(diff.max().item()))

    return fwd_max_err, ulp_floor, bwd_max_err


# Small-to-moderate shapes — fast CI signal across shape regimes.
@pytest.mark.parametrize(
    "num_experts,out_features,in_features",
    [
        (4, 64, 128),
        (8, 128, 256),
        (32, 512, 1024),
    ],
)
def test_atriton_vs_a_parity(num_experts, out_features, in_features):
    """ATriton vs A: fwd within 1 ULP, bwd bit-exact (pass_through_bwd=True)."""
    fwd_err, ulp_floor, bwd_err = _parity_atriton_vs_a(
        num_experts, out_features, in_features, torch.bfloat16
    )
    assert fwd_err <= ulp_floor + 1e-6, (
        f"fwd_max_abs_err={fwd_err:.6f} > 1-ULP floor {ulp_floor:.6f} "
        f"at N={num_experts}, [out, in]=[{out_features}, {in_features}]"
    )
    assert bwd_err == 0.0, (
        f"bwd_max_abs_err={bwd_err:.6f} expected 0.0 "
        f"at N={num_experts}, [out, in]=[{out_features}, {in_features}]"
    )


@pytest.mark.slow
def test_atriton_vs_a_parity_ultra_production_shape():
    """AC2 — Ultra production shape (N=32, [5120, 8192] bf16).

    Marked `slow` because the unquantized + quantized + gradient copies of
    32 expert weights at [5120, 8192] bf16 use about 5 GB of GPU memory.
    """
    fwd_err, ulp_floor, bwd_err = _parity_atriton_vs_a(
        num_experts=32,
        out_features=5120,
        in_features=8192,
        dtype=torch.bfloat16,
    )
    assert fwd_err <= ulp_floor + 1e-6, (
        f"fwd_max_abs_err={fwd_err:.6f} > 1-ULP floor {ulp_floor:.6f} at Ultra shape"
    )
    assert bwd_err == 0.0, (
        f"bwd_max_abs_err={bwd_err:.6f} expected 0.0 at Ultra shape"
    )
