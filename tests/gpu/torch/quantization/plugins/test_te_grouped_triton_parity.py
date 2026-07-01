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
    from modelopt.torch.kernels.quantization.gemm import IS_AVAILABLE, grouped_axis0_fakequant
    from modelopt.torch.quantization.extensions import get_cuda_ext
    from modelopt.torch.quantization.plugins.transformer_engine import _GroupedAxis0INTFakeQuantFn
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
    q_a = [cuda_ext.fake_tensor_quant(w, a, 8, False, True) for w, a in zip(weights, amax_scalars)]
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
    # _GroupedAxis0INTFakeQuantFn.apply(
    #     amax_vec, num_bits, narrow_range, pass_through_bwd, *weights
    # )
    qs_t = _GroupedAxis0INTFakeQuantFn.apply(amax_vec, 8, True, True, *ws_t)
    loss_t = sum(q.sum() for q in qs_t)
    loss_t.backward()

    bwd_max_err = 0.0
    for i in range(num_experts):
        diff = (ws_a[i].grad.float() - ws_t[i].grad.float()).abs()
        bwd_max_err = max(bwd_max_err, float(diff.max().item()))

    return fwd_max_err, ulp_floor, bwd_max_err


# Small-to-moderate shapes — fast CI signal across shape regimes.
@pytest.mark.parametrize(
    ("num_experts", "out_features", "in_features"),
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
    assert bwd_err == 0.0, f"bwd_max_abs_err={bwd_err:.6f} expected 0.0 at Ultra shape"


@pytest.mark.parametrize(
    ("num_experts", "out_features", "in_features"),
    [(4, 64, 128), (8, 128, 256), (32, 512, 1024)],
)
@pytest.mark.parametrize("calibrated_amax", [False, True])
def test_grouped_nvfp4_vs_n_quantizer_parity(
    num_experts, out_features, in_features, calibrated_amax
):
    """Grouped dynamic NVFP4 matches N independent TensorQuantizer calls."""
    import modelopt.torch.kernels.quantization.gemm as triton_kernels
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.quantization.plugins.transformer_engine import _GroupedAxis0NVFP4FakeQuantFn

    if not triton_kernels.IS_AVAILABLE or not hasattr(triton_kernels, "grouped_nvfp4_fakequant"):
        pytest.skip("grouped NVFP4 Triton kernel requires compute capability >= 8.9")

    torch.manual_seed(0)
    weights = [
        torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda")
        for _ in range(num_experts)
    ]
    config = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        pass_through_bwd=True,
    )

    ref_weights = [weight.detach().clone().requires_grad_(True) for weight in weights]
    quantizers = [TensorQuantizer(config).cuda() for _ in range(num_experts)]
    expert_amax = None
    if calibrated_amax:
        expert_amax = torch.stack([weight.detach().abs().max() * 1.25 for weight in weights])
        for quantizer, amax in zip(quantizers, expert_amax):
            quantizer.amax = amax
    ref_outputs = [quantizer(weight) for quantizer, weight in zip(quantizers, ref_weights)]

    fused_weights = [weight.detach().clone().requires_grad_(True) for weight in weights]
    fused_outputs = _GroupedAxis0NVFP4FakeQuantFn.apply(expert_amax, 16, *fused_weights)

    for reference, fused in zip(ref_outputs, fused_outputs):
        torch.testing.assert_close(fused, reference, rtol=0, atol=0)

    upstream = [torch.randn_like(output) for output in ref_outputs]
    torch.autograd.backward(ref_outputs, upstream)
    torch.autograd.backward(fused_outputs, upstream)
    for reference, fused in zip(ref_weights, fused_weights):
        torch.testing.assert_close(fused.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.slow
def test_grouped_nvfp4_n_quantizer_nano_speedup(capsys):
    """Grouped NVFP4 is at least 2x faster on public Nano routed-MLP shapes."""
    import modelopt.torch.kernels.quantization.gemm as triton_kernels
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    if not triton_kernels.IS_AVAILABLE or not hasattr(triton_kernels, "grouped_nvfp4_fakequant"):
        pytest.skip("grouped NVFP4 Triton kernel requires compute capability >= 8.9")

    config = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        pass_through_bwd=True,
    )

    def elapsed_ms(fn, warmup=3, iterations=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations

    # Public config: huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
    # It has hidden_size=2688,
    # moe_intermediate_size=1856, and 128 routed experts. EP=4 gives 32
    # experts/GPU. TE's first grouped projection carries both gate branches.
    shapes = [(3712, 2688), (2688, 1856)]
    total_reference_ms = 0.0
    total_fused_ms = 0.0
    rows = []
    with torch.no_grad():
        for out_features, in_features in shapes:
            torch.manual_seed(0)
            weights = [
                torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda")
                for _ in range(32)
            ]
            quantizers = [TensorQuantizer(config).cuda() for _ in weights]
            reference_ms = elapsed_ms(
                lambda: tuple(quantizer(weight) for quantizer, weight in zip(quantizers, weights))
            )
            fused_ms = elapsed_ms(lambda: triton_kernels.grouped_nvfp4_fakequant(weights))
            total_reference_ms += reference_ms
            total_fused_ms += fused_ms
            rows.append((out_features, in_features, reference_ms, fused_ms))

    speedup = total_reference_ms / total_fused_ms
    with capsys.disabled():
        for out_features, in_features, reference_ms, fused_ms in rows:
            print(
                f"NVFP4 N-quantizer [out,in]=[{out_features},{in_features}]: "
                f"unfused={reference_ms:.3f} ms fused={fused_ms:.3f} ms"
            )
        print(f"NVFP4 N-quantizer combined speedup: {speedup:.2f}x")
    assert speedup >= 2.0
