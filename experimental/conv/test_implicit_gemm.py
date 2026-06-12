# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for Conv3D implicit GEMM CUDA kernel.

Tests both non-quantized path (vs cuDNN) and FP4-quantized path (vs Triton reference).
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def cuda_conv3d():
    """Import and return the CUDA implicit GEMM conv3d function."""
    from experimental.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda

    return conv3d_implicit_gemm_cuda


def _triton_fp4_available():
    """Check if the Triton FP4 fake quant kernel is available (requires compute >= 8.9)."""
    try:
        import modelopt.torch.quantization.triton as triton_kernel

        return hasattr(triton_kernel, "fp4_fake_quant_block")
    except ImportError:
        return False


requires_triton_fp4 = pytest.mark.skipif(
    not _triton_fp4_available(),
    reason="Triton fp4_fake_quant_block not available (requires compute >= 8.9)",
)


# BF16 WMMA accumulates in FP32 but inputs are rounded to BF16, so expect diffs.
# For large K (e.g. 3456 = 128*27), max abs diff can reach ~0.8 due to BF16 rounding
# and different accumulation order vs cuDNN's FP32 path.
ATOL = 1.0
RTOL = 1e-3


def _run_conv3d_test(cuda_conv3d, x, w, bias, stride, padding, dilation):
    """Helper: run both cuDNN and implicit GEMM, compare results."""
    ref = F.conv3d(x, w, bias=bias, stride=stride, padding=padding, dilation=dilation)
    out = cuda_conv3d(
        x, w, bias=bias, stride=stride, padding=padding, dilation=dilation, quant_act=False
    )
    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    abs_diff = (out - ref).abs()
    max_diff = abs_diff.max().item()
    # Scale tolerance with K (reduction dimension) — BF16 rounding accumulates
    cin = w.shape[1]
    k_size = cin * w.shape[2] * w.shape[3] * w.shape[4]
    scaled_atol = ATOL * (k_size / 1000.0) ** 0.5
    assert max_diff < scaled_atol, (
        f"Max abs diff {max_diff:.6e} exceeds tolerance {scaled_atol:.4f} (K={k_size})"
    )
    # Check mean diff is small (more robust than quantile for large tensors)
    mean_diff = abs_diff.mean().item()
    assert mean_diff < scaled_atol * 0.1, f"Mean diff {mean_diff:.6e} too high"
    return max_diff


class TestConv3dBasic:
    """Basic correctness tests with simple shapes."""

    def test_minimal(self, cuda_conv3d):
        """Smallest possible conv3d: 1x1x1 kernel, single channel."""
        x = torch.randn(1, 1, 1, 1, 1, device="cuda", dtype=torch.float32)
        w = torch.randn(1, 1, 1, 1, 1, device="cuda", dtype=torch.float32)
        diff = _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1))
        # K=1, so BF16 rounding is the only source of error
        assert diff < 1e-2

    def test_single_channel_3x3x3(self, cuda_conv3d):
        """Single input/output channel with 3x3x3 kernel."""
        x = torch.randn(1, 1, 5, 5, 5, device="cuda", dtype=torch.float32)
        w = torch.randn(1, 1, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_multi_channel(self, cuda_conv3d):
        """Multiple input and output channels."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_with_bias(self, cuda_conv3d):
        """Conv3d with bias."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        b = torch.randn(32, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, b, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_batch_size(self, cuda_conv3d):
        """Batch size > 1."""
        x = torch.randn(4, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))


class TestConv3dStride:
    """Tests with various stride configurations."""

    def test_stride_2(self, cuda_conv3d):
        """Uniform stride of 2."""
        x = torch.randn(1, 16, 16, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (2, 2, 2), (1, 1, 1), (1, 1, 1))

    def test_asymmetric_stride(self, cuda_conv3d):
        """Different stride per dimension."""
        x = torch.randn(1, 16, 16, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 2, 2), (1, 1, 1), (1, 1, 1))


class TestConv3dPadding:
    """Tests with various padding configurations."""

    def test_no_padding(self, cuda_conv3d):
        """Zero padding."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1))

    def test_large_padding(self, cuda_conv3d):
        """Padding larger than kernel radius."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (2, 2, 2), (1, 1, 1))

    def test_asymmetric_padding(self, cuda_conv3d):
        """Different padding per dimension."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (0, 1, 2), (1, 1, 1))


class TestConv3dDilation:
    """Tests with dilation."""

    def test_dilation_2(self, cuda_conv3d):
        """Uniform dilation of 2."""
        x = torch.randn(1, 16, 16, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (2, 2, 2), (2, 2, 2))

    def test_asymmetric_dilation(self, cuda_conv3d):
        """Different dilation per dimension."""
        x = torch.randn(1, 16, 16, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 2, 2), (1, 2, 2))


class TestConv3dKernelSizes:
    """Tests with non-3x3x3 kernels."""

    def test_1x1x1_kernel(self, cuda_conv3d):
        """Pointwise 1x1x1 kernel."""
        x = torch.randn(1, 64, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(128, 64, 1, 1, 1, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1))

    def test_asymmetric_kernel(self, cuda_conv3d):
        """Kernel with different sizes per dimension (e.g. 1x3x3)."""
        x = torch.randn(1, 16, 8, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 1, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (0, 1, 1), (1, 1, 1))

    def test_5x5x5_kernel(self, cuda_conv3d):
        """Larger 5x5x5 kernel."""
        x = torch.randn(1, 8, 16, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(16, 8, 5, 5, 5, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (2, 2, 2), (1, 1, 1))


class TestConv3dRealisticShapes:
    """Tests with shapes resembling real video diffusion models."""

    def test_wan22_shape(self, cuda_conv3d):
        """Shape from Wan2.2 video diffusion backbone."""
        x = torch.randn(1, 128, 21, 60, 106, device="cuda", dtype=torch.float32)
        w = torch.randn(512, 128, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_large_cout(self, cuda_conv3d):
        """Large output channel count."""
        x = torch.randn(1, 64, 8, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(512, 64, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_large_cin(self, cuda_conv3d):
        """Large input channel count."""
        x = torch.randn(1, 512, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(64, 512, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))


class TestConv3dEdgeCases:
    """Edge cases for tile boundary handling."""

    def test_m_not_aligned_to_block(self, cuda_conv3d):
        """M (N*OD*OH*OW) not a multiple of BLOCK_M=64."""
        # 1*3*5*7 = 105, not divisible by 64
        x = torch.randn(1, 8, 5, 7, 9, device="cuda", dtype=torch.float32)
        w = torch.randn(16, 8, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_cout_not_aligned_to_block(self, cuda_conv3d):
        """Cout not a multiple of BLOCK_N=64."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(17, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_k_not_aligned_to_block(self, cuda_conv3d):
        """K (Cin*kD*kH*kW) not a multiple of BLOCK_K."""
        # Cin=7, kDHW=27, K=189 -- not a multiple of 128 or 256
        x = torch.randn(1, 7, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(16, 7, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (1, 1, 1), (1, 1, 1))

    def test_output_size_1x1x1(self, cuda_conv3d):
        """Output spatial dims are all 1."""
        x = torch.randn(1, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        _run_conv3d_test(cuda_conv3d, x, w, None, (1, 1, 1), (0, 0, 0), (1, 1, 1))

    def test_single_output_element(self, cuda_conv3d):
        """M=1: batch=1, output 1x1x1.

        With only one output element, mean diff == max diff, so the generic
        helper's mean_diff < scaled_atol * 0.1 check is too tight. Use max diff only.
        """
        x = torch.randn(1, 4, 3, 3, 3, device="cuda", dtype=torch.float32)
        w = torch.randn(1, 4, 3, 3, 3, device="cuda", dtype=torch.float32)
        ref = F.conv3d(x, w, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1))
        out = cuda_conv3d(
            x, w, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), quant_act=False
        )
        assert out.shape == ref.shape
        max_diff = (out - ref).abs().max().item()
        assert max_diff < ATOL, f"Max abs diff {max_diff:.6e} exceeds tolerance {ATOL}"


class TestConv3dFP4BlockSize:
    """Test all FP4 block size configs (BLOCK_K=256 always, FP4_BLOCK_SIZE varies).

    Non-quantized path ignores FP4_BLOCK_SIZE, so all should match cuDNN.
    """

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_non_quant_all_block_sizes(self, cuda_conv3d, fp4_block_size):
        """Non-quant conv should match cuDNN regardless of fp4_block_size."""
        x = torch.randn(1, 32, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(64, 32, 3, 3, 3, device="cuda", dtype=torch.float32)
        ref = F.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1))
        out = cuda_conv3d(
            x,
            w,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            quant_act=False,
            fp4_block_size=fp4_block_size,
        )
        assert out.shape == ref.shape
        assert (out - ref).abs().max().item() < ATOL


class TestConv3dDeterminism:
    """Verify deterministic output across repeated calls."""

    def test_deterministic(self, cuda_conv3d):
        """Repeated calls produce identical output."""
        torch.manual_seed(123)
        x = torch.randn(1, 32, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(64, 32, 3, 3, 3, device="cuda", dtype=torch.float32)
        out1 = cuda_conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1), quant_act=False)
        out2 = cuda_conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1), quant_act=False)
        assert torch.equal(out1, out2), "Kernel is not deterministic"


# =============================================================================
# FP4 Quantized Conv3D Tests (fused activation quantization)
# =============================================================================


@pytest.fixture(scope="module")
def cuda_fp4_quant():
    """Import FP4 fake quant for reference comparisons."""
    from experimental.conv.implicit_gemm_cuda import fp4_fake_quant

    return fp4_fake_quant


class TestConv3dFP4QuantBlockSizes:
    """Test fused FP4 activation quantization with all supported block sizes.

    The kernel applies blockwise FP4 quantization to the im2col'd activation tiles
    along the K dimension. We verify correctness by comparing the fused kernel output
    against an unfused reference: fp4_fake_quant(im2col) @ fp4_fake_quant(weight).
    """

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_quant_runs_all_block_sizes(self, cuda_conv3d, fp4_block_size):
        """All FP4 block sizes should run without errors and produce valid output."""
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)

        out = cuda_conv3d(
            x,
            w,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            act_amax=act_amax,
            quant_act=True,
            fp4_block_size=fp4_block_size,
        )
        assert out.shape == (1, 32, 8, 8, 8)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert out.abs().max() > 0, "Output is all zeros"

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_quant_deterministic(self, cuda_conv3d, fp4_block_size):
        """Quantized conv should be deterministic for all block sizes."""
        torch.manual_seed(42)
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)

        kwargs = {
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "dilation": (1, 1, 1),
            "act_amax": act_amax,
            "quant_act": True,
            "fp4_block_size": fp4_block_size,
        }
        out1 = cuda_conv3d(x, w, **kwargs)
        out2 = cuda_conv3d(x, w, **kwargs)
        assert torch.equal(out1, out2), f"Non-deterministic for fp4_block_size={fp4_block_size}"

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_quant_vs_unfused_reference(self, cuda_conv3d, cuda_fp4_quant, fp4_block_size):
        """Compare fused kernel vs unfused: fp4(im2col) @ fp4(weight).

        Uses a shape where K is a multiple of 256 so all K-tiles are full
        and block boundaries align perfectly between fused and unfused paths.
        """
        torch.manual_seed(123)
        # K = Cin * kD * kH * kW. Choose Cin so K is a multiple of 256.
        # Cin=256, k=1x1x1 -> K=256 (exactly 1 full K-tile)
        cin, cout = 256, 64
        x = torch.randn(1, cin, 4, 4, 4, device="cuda", dtype=torch.float32)
        w = torch.randn(cout, cin, 1, 1, 1, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)
        w_amax = w.abs().max().unsqueeze(0)

        # Unfused reference:
        # 1. Build im2col matrix (for 1x1x1 kernel, it's just reshape)
        n, c, d, h, w_dim = x.shape
        im2col = x.permute(0, 2, 3, 4, 1).reshape(-1, cin)  # [M, K]

        # 2. FP4 fake-quant both matrices along K with the same block_size
        im2col_q = cuda_fp4_quant(im2col, act_amax, fp4_block_size)
        w_flat = w.reshape(cout, cin).transpose(0, 1).contiguous()  # [K, Cout]
        w_flat_q = cuda_fp4_quant(w_flat, w_amax, fp4_block_size)

        # 3. Matmul (in BF16 to match kernel's WMMA path)
        ref_out = (im2col_q.bfloat16() @ w_flat_q.bfloat16()).float()
        ref_out = ref_out.view(n, d, h, w_dim, cout).permute(0, 4, 1, 2, 3)

        # Note: the fused kernel does NOT quantize weights — weights are passed as-is.
        # So for a proper comparison we need the fused kernel with pre-quantized weights.
        fused_out_preq = cuda_conv3d(
            x,
            w_flat_q.transpose(0, 1).reshape(cout, cin, 1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            act_amax=act_amax,
            quant_act=True,
            fp4_block_size=fp4_block_size,
        )

        # The fused kernel and unfused reference should match closely.
        # Differences come from BF16 accumulation order (WMMA 16x16x16 tiles vs flat matmul).
        max_diff = (fused_out_preq - ref_out).abs().max().item()
        mean_diff = (fused_out_preq - ref_out).abs().mean().item()
        # Scale tolerance with K
        scaled_atol = ATOL * (cin / 1000.0) ** 0.5
        assert max_diff < scaled_atol, (
            f"fp4_block_size={fp4_block_size}: fused vs unfused max diff {max_diff:.4f} "
            f"exceeds tolerance {scaled_atol:.4f}"
        )
        assert mean_diff < scaled_atol * 0.1, (
            f"fp4_block_size={fp4_block_size}: mean diff {mean_diff:.6e} too high"
        )

    def test_smaller_block_less_error(self, cuda_conv3d):
        """Smaller FP4 block sizes should generally produce lower quantization error.

        Finer-grained blocks capture local ranges better, reducing quant error vs cuDNN.
        Test monotonicity: error(16) <= error(32) <= ... <= error(256) (with some tolerance).
        Reports detailed accuracy metrics for each block size vs cuDNN baseline.
        """
        torch.manual_seed(42)

        # Test multiple shapes to get a comprehensive picture
        configs = [
            ("Small K=432", 1, 16, 8, 8, 8, 32, 3, 3, 3),
            ("Medium K=1728", 1, 64, 8, 8, 8, 64, 3, 3, 3),
            ("Large K=3456", 1, 128, 5, 8, 8, 256, 3, 3, 3),
        ]

        block_sizes = [16, 32, 64, 128, 256]
        all_errors = {}

        for desc, n, cin, d, h, w_s, cout, kd, kh, kw in configs:
            x = torch.randn(n, cin, d, h, w_s, device="cuda", dtype=torch.float32)
            w = torch.randn(cout, cin, kd, kh, kw, device="cuda", dtype=torch.float32)
            act_amax = x.abs().max().unsqueeze(0)
            k_size = cin * kd * kh * kw

            ref = F.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1))
            ref_abs_mean = ref.abs().mean().item()

            # Also compute no-quant baseline (BF16 rounding only)
            out_nq = cuda_conv3d(
                x,
                w,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                quant_act=False,
            )
            nq_diff = (out_nq - ref).abs()

            print(
                f"\n  {desc} (K={k_size}), output range [{ref.min().item():.1f}, {ref.max().item():.1f}]"
            )
            print(
                f"  {'Block Size':>10} | {'Max Diff':>10} | {'Mean Diff':>10} | {'RMSE':>10} | {'Rel Err%':>8}"
            )
            print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")
            print(
                f"  {'no-quant':>10} | {nq_diff.max().item():>10.4f} | "
                f"{nq_diff.mean().item():>10.6f} | "
                f"{((out_nq - ref) ** 2).mean().sqrt().item():>10.4f} | "
                f"{nq_diff.mean().item() / ref_abs_mean * 100:>7.3f}%"
            )

            errors = {}
            for bs in block_sizes:
                out = cuda_conv3d(
                    x,
                    w,
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    dilation=(1, 1, 1),
                    act_amax=act_amax,
                    quant_act=True,
                    fp4_block_size=bs,
                )
                diff = (out - ref).abs()
                max_d = diff.max().item()
                mean_d = diff.mean().item()
                rmse = ((out - ref) ** 2).mean().sqrt().item()
                rel_err = mean_d / ref_abs_mean * 100
                errors[bs] = mean_d
                print(
                    f"  {bs:>10} | {max_d:>10.4f} | {mean_d:>10.6f} | "
                    f"{rmse:>10.4f} | {rel_err:>7.3f}%"
                )
            all_errors[desc] = errors

        # Monotonicity check on the medium config
        errors = all_errors["Medium K=1728"]
        for smaller, larger in [(16, 64), (16, 256), (32, 256), (64, 256)]:
            assert errors[smaller] <= errors[larger] * 1.2, (
                f"Expected error({smaller})={errors[smaller]:.6f} <= "
                f"error({larger})={errors[larger]:.6f} * 1.2"
            )

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_quant_with_bias(self, cuda_conv3d, fp4_block_size):
        """FP4 quantized conv with bias for all block sizes."""
        torch.manual_seed(42)
        x = torch.randn(1, 16, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float32)
        b = torch.randn(32, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)

        out = cuda_conv3d(
            x,
            w,
            bias=b,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            act_amax=act_amax,
            quant_act=True,
            fp4_block_size=fp4_block_size,
        )
        assert out.shape == (1, 32, 8, 8, 8)
        assert not torch.isnan(out).any()
        # Bias should shift output values
        out_no_bias = cuda_conv3d(
            x,
            w,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            act_amax=act_amax,
            quant_act=True,
            fp4_block_size=fp4_block_size,
        )
        assert not torch.equal(out, out_no_bias), "Bias had no effect"

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_quant_k_not_aligned(self, cuda_conv3d, fp4_block_size):
        """FP4 quant with K not aligned to BLOCK_K or fp4_block_size.

        K = Cin * kDHW = 7 * 27 = 189. The last K-tile has partial data (zeros padded).
        """
        torch.manual_seed(42)
        x = torch.randn(1, 7, 8, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(16, 7, 3, 3, 3, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)

        out = cuda_conv3d(
            x,
            w,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            act_amax=act_amax,
            quant_act=True,
            fp4_block_size=fp4_block_size,
        )
        assert out.shape == (1, 16, 8, 8, 8)
        assert not torch.isnan(out).any()
        assert out.abs().max() > 0

    @pytest.mark.parametrize("fp4_block_size", [16, 32, 64, 128, 256])
    def test_quant_realistic_shape(self, cuda_conv3d, fp4_block_size):
        """Realistic video diffusion shape with all FP4 block sizes."""
        torch.manual_seed(42)
        x = torch.randn(1, 128, 5, 8, 8, device="cuda", dtype=torch.float32)
        w = torch.randn(256, 128, 3, 3, 3, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)

        out = cuda_conv3d(
            x,
            w,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            act_amax=act_amax,
            quant_act=True,
            fp4_block_size=fp4_block_size,
        )
        assert out.shape == (1, 256, 5, 8, 8)
        assert not torch.isnan(out).any()
        assert out.abs().max() > 0


# =============================================================================
# FP4 Fake Quantization Tests
# =============================================================================


@pytest.fixture(scope="module")
def cuda_fp4():
    """Import and return the CUDA FP4 fake quant function."""
    from experimental.conv.implicit_gemm_cuda import fp4_fake_quant

    return fp4_fake_quant


def _py_fp4_fake_quant_ref(x_flat, global_amax, block_size):
    """Pure Python reference for FP4 fake quant (no BF16 rounding).

    This implements the exact same algorithm as the CUDA kernel:
    1. Compute global_scale = global_amax / (6 * 448)
    2. Per block: block_max = max(|x|), scale = fp8_e4m3_roundtrip(block_max / (6 * global_scale)) * global_scale
    3. Quantize each element to nearest E2M1 level, then dequantize.
    """
    import math

    # E2M1 quantization levels: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # Boundaries (midpoints): <=0.25->0, <0.75->0.5, <=1.25->1, <1.75->1.5, <=2.5->2, <3.5->3, <=5->4, >5->6
    def quantize_e2m1(scaled_abs):
        if scaled_abs <= 0.25:
            return 0.0
        elif scaled_abs < 0.75:
            return 0.5
        elif scaled_abs <= 1.25:
            return 1.0
        elif scaled_abs < 1.75:
            return 1.5
        elif scaled_abs <= 2.5:
            return 2.0
        elif scaled_abs < 3.5:
            return 3.0
        elif scaled_abs <= 5.0:
            return 4.0
        else:
            return 6.0

    def fp8_e4m3_roundtrip(val):
        """Simulate FP8 E4M3 round-trip in Python."""
        if val == 0.0:
            return 0.0
        sign = 1.0 if val >= 0 else -1.0
        val = abs(val)
        # FP8 E4M3: bias=7, 3 mantissa bits, max=448, no inf/nan
        if val > 448.0:
            return sign * 448.0
        # Compute exponent
        exp = math.floor(math.log2(val))
        exp = max(exp, -6)  # min normal exponent for E4M3
        # Compute mantissa (3 bits)
        mantissa = val / (2.0**exp)  # 1.xxx
        mantissa_bits = round((mantissa - 1.0) * 8.0)  # 3 bits
        if mantissa_bits > 7:
            mantissa_bits = 0
            exp += 1
            if exp > 8:
                return sign * 448.0
        # Reconstruct
        result = (1.0 + mantissa_bits / 8.0) * (2.0**exp)
        return sign * result

    global_scale = float(global_amax) / (6.0 * 448.0)
    x_np = x_flat.cpu().float().numpy().copy()
    num_blocks = len(x_np) // block_size

    for b in range(num_blocks):
        block = x_np[b * block_size : (b + 1) * block_size]
        block_max = float(max(abs(v) for v in block))

        # Scale quantization
        scaled = block_max / (6.0 * global_scale)
        scaled = min(scaled, 448.0)
        quantized_scale = fp8_e4m3_roundtrip(scaled) * global_scale
        if quantized_scale < 1e-5:
            quantized_scale = 1.0
        inv_scale = 1.0 / quantized_scale

        for i in range(block_size):
            val = block[i]
            sign = 1.0 if val >= 0 else -1.0
            q = quantize_e2m1(abs(val) * inv_scale)
            x_np[b * block_size + i] = sign * q * quantized_scale

    return torch.tensor(x_np, device=x_flat.device)


class TestFP4FakeQuantValues:
    """Test FP4 fake quant with known E2M1 table values."""

    def test_exact_e2m1_values(self, cuda_fp4):
        """E2M1 representable values should round-trip exactly (when scale=1 via amax=6*448)."""
        # With global_amax = 6*448 = 2688, global_scale = 1.0
        # A single-block input with max=6 -> block_max=6, scaled=6/(6*1)=1.0
        # fp8_e4m3(1.0)=1.0, scale = 1.0*1.0 = 1.0
        block_size = 8
        vals = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda", dtype=torch.float32)
        amax = torch.tensor([6.0 * 448.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(vals, amax, block_size)
        assert torch.allclose(out, vals, atol=1e-5), f"Got {out} vs expected {vals}"

    def test_exact_e2m1_negative(self, cuda_fp4):
        """Negative E2M1 values should also round-trip."""
        block_size = 8
        vals = torch.tensor([0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda", dtype=torch.float32)
        amax = torch.tensor([6.0 * 448.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(vals, amax, block_size)
        assert torch.allclose(out, vals, atol=1e-5), f"Got {out} vs expected {vals}"

    def test_below_boundary(self, cuda_fp4):
        """Values slightly below E2M1 boundaries should quantize down."""
        block_size = 8
        # Boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
        # Slightly below -> quantize to lower level
        inp = torch.tensor(
            [0.15, 0.65, 1.15, 1.65, 2.4, 3.4, 4.9, 6.0], device="cuda", dtype=torch.float32
        )
        expected = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda", dtype=torch.float32
        )
        amax = torch.tensor([6.0 * 448.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(inp, amax, block_size)
        assert torch.allclose(out, expected, atol=1e-5), f"Got {out} vs expected {expected}"

    def test_above_boundary(self, cuda_fp4):
        """Values slightly above E2M1 boundaries should quantize up."""
        block_size = 8
        inp = torch.tensor(
            [0.35, 0.85, 1.35, 1.85, 2.6, 3.6, 5.1, 6.0], device="cuda", dtype=torch.float32
        )
        expected = torch.tensor(
            [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 6.0], device="cuda", dtype=torch.float32
        )
        amax = torch.tensor([6.0 * 448.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(inp, amax, block_size)
        assert torch.allclose(out, expected, atol=1e-5), f"Got {out} vs expected {expected}"

    def test_mixed_signs(self, cuda_fp4):
        """Mixed positive/negative values."""
        block_size = 8
        inp = torch.tensor([-6, -3, -1, 0, 0.5, 2, 4, 6], device="cuda", dtype=torch.float32)
        expected = torch.tensor([-6, -3, -1, 0, 0.5, 2, 4, 6], device="cuda", dtype=torch.float32)
        amax = torch.tensor([6.0 * 448.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(inp, amax, block_size)
        assert torch.allclose(out, expected, atol=1e-5), f"Got {out} vs expected {expected}"


class TestFP4FakeQuantScale:
    """Test FP4 scale computation and FP8 round-trip."""

    def test_scale_factor(self, cuda_fp4):
        """When amax != 6*448, scale should adjust values proportionally."""
        block_size = 8
        # global_amax = 12*448 = 5376, global_scale = 2.0
        # Input block max = 12 -> scaled = 12/(6*2) = 1.0 -> fp8(1.0) = 1.0 -> scale = 2.0
        # So input 12 -> |12|/2 = 6.0 -> q=6 -> 6*2 = 12
        inp = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], device="cuda", dtype=torch.float32)
        amax = torch.tensor([12.0 * 448.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(inp, amax, block_size)
        # Expected: each val/2.0 -> quantize to E2M1 -> * 2.0
        expected = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], device="cuda", dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-4), f"Got {out} vs expected {expected}"

    def test_zero_block(self, cuda_fp4):
        """All-zero block should produce all zeros."""
        block_size = 16
        inp = torch.zeros(block_size, device="cuda", dtype=torch.float32)
        amax = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        out = cuda_fp4(inp, amax, block_size)
        assert torch.equal(out, inp)

    def test_multiple_blocks(self, cuda_fp4):
        """Multiple blocks with different ranges."""
        block_size = 8
        # Block 0: small values, Block 1: large values
        block0 = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], device="cuda")
        block1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 6], device="cuda")
        inp = torch.cat([block0, block1])
        amax = inp.abs().max().unsqueeze(0)
        out = cuda_fp4(inp, amax, block_size)
        # Each block should be independently quantized
        assert out.shape == inp.shape
        # Block 1 exact values should be close to E2M1 levels
        assert out[8:].abs().max() <= 6.0 + 1e-5


class TestFP4FakeQuantBlockSizes:
    """Test different block sizes."""

    @pytest.mark.parametrize("block_size", [8, 16, 32, 64, 128, 256])
    def test_block_sizes(self, cuda_fp4, block_size):
        """FP4 quant should work for various block sizes."""
        torch.manual_seed(42)
        num_blocks = 4
        inp = torch.randn(num_blocks * block_size, device="cuda", dtype=torch.float32) * 5
        amax = inp.abs().max().unsqueeze(0)
        out = cuda_fp4(inp, amax, block_size)
        assert out.shape == inp.shape
        # Output should not be all zeros for non-zero input
        assert out.abs().max() > 0
        # Output should be <= max possible after quant
        assert out.abs().max() <= inp.abs().max() * 1.5  # generous bound


class TestFP4FakeQuantVsReference:
    """Compare CUDA FP4 fake quant against Python reference implementation."""

    @pytest.mark.parametrize("block_size", [8, 16, 32])
    def test_vs_python_ref(self, cuda_fp4, block_size):
        """CUDA kernel should match the Python reference exactly."""
        torch.manual_seed(123)
        num_blocks = 8
        inp = torch.randn(num_blocks * block_size, device="cuda") * 10
        amax = inp.abs().max().unsqueeze(0)

        cuda_out = cuda_fp4(inp, amax, block_size)
        ref_out = _py_fp4_fake_quant_ref(inp, amax, block_size)

        assert torch.allclose(cuda_out, ref_out, atol=1e-5), (
            f"CUDA vs Python ref max diff: {(cuda_out - ref_out).abs().max().item():.6e}"
        )

    @pytest.mark.parametrize("block_size", [16, 32])
    def test_vs_python_ref_large(self, cuda_fp4, block_size):
        """Larger tensor test against Python reference."""
        torch.manual_seed(456)
        num_blocks = 64
        inp = torch.randn(num_blocks * block_size, device="cuda") * 20
        amax = inp.abs().max().unsqueeze(0)

        cuda_out = cuda_fp4(inp, amax, block_size)
        ref_out = _py_fp4_fake_quant_ref(inp, amax, block_size)

        assert torch.allclose(cuda_out, ref_out, atol=1e-4), (
            f"CUDA vs Python ref max diff: {(cuda_out - ref_out).abs().max().item():.6e}"
        )


class TestFP4FakeQuantVsTriton:
    """Compare CUDA FP4 fake quant against Triton fp4_fake_quant_block reference."""

    @requires_triton_fp4
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    @pytest.mark.parametrize("num_blocks", [4, 16, 64])
    def test_vs_triton(self, cuda_fp4, block_size, num_blocks):
        """CUDA kernel should match the Triton fp4_fake_quant_block."""
        from modelopt.torch.quantization.triton import fp4_fake_quant_block

        torch.manual_seed(42)
        x = torch.randn(num_blocks, block_size, device="cuda", dtype=torch.float32) * 10
        global_amax = x.abs().max()

        cuda_out = cuda_fp4(x, global_amax.unsqueeze(0), block_size)
        triton_out = fp4_fake_quant_block(
            x,
            global_amax=global_amax,
            block_size=block_size,
            tile_rows=num_blocks,
            tile_cols=block_size,
        )

        assert torch.allclose(cuda_out, triton_out, atol=1e-5), (
            f"CUDA vs Triton max diff: {(cuda_out - triton_out).abs().max().item():.6e}\n"
            f"Mean diff: {(cuda_out - triton_out).abs().mean().item():.6e}"
        )


class TestFP4FakeQuantDeterminism:
    """Verify FP4 quant is deterministic."""

    def test_deterministic(self, cuda_fp4):
        """Repeated calls produce identical output."""
        torch.manual_seed(99)
        inp = torch.randn(256, device="cuda") * 5
        amax = inp.abs().max().unsqueeze(0)
        out1 = cuda_fp4(inp, amax, 16)
        out2 = cuda_fp4(inp, amax, 16)
        assert torch.equal(out1, out2), "FP4 fake quant is not deterministic"


# =============================================================================
# Cross-validation: experimental FP4 vs modelopt FP4 implementations
# =============================================================================


def _modelopt_cuda_ext_mx_available():
    """Check if the modelopt CUDA MX extension is available."""
    try:
        from modelopt.torch.quantization.extensions import get_cuda_ext_mx

        return get_cuda_ext_mx() is not None
    except Exception:
        return False


def _modelopt_dynamic_block_quantize_available():
    """Check if dynamic_block_quantize_op is available."""
    try:
        from modelopt.torch.quantization.tensor_quant import dynamic_block_quantize_op

        return dynamic_block_quantize_op is not None
    except Exception:
        return False


requires_cuda_ext_mx = pytest.mark.skipif(
    not _modelopt_cuda_ext_mx_available(),
    reason="modelopt cuda_ext_mx not available",
)

requires_dynamic_block_quantize = pytest.mark.skipif(
    not _modelopt_dynamic_block_quantize_available(),
    reason="modelopt dynamic_block_quantize_op not available",
)


class TestFP4FakeQuantVsModelopt:
    """Compare experimental CUDA FP4 fake quant against all modelopt FP4 implementations.

    This ensures the standalone FP4 kernel in experimental/conv produces the same
    results as the official modelopt quantization paths:
    1. Triton fp4_fake_quant_block (Hopper+ dynamic blockwise)
    2. cuda_ext_mx.fused_amax_convert (CUDA extension fallback)
    3. dynamic_block_quantize_op (high-level API that dispatches to either)
    """

    @requires_triton_fp4
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_vs_triton_fp4_fake_quant_block(self, cuda_fp4, block_size, seed):
        """Compare against modelopt Triton fp4_fake_quant_block."""
        from modelopt.torch.quantization.triton import fp4_fake_quant_block

        torch.manual_seed(seed)
        num_blocks = 16
        x = torch.randn(num_blocks, block_size, device="cuda", dtype=torch.float32) * 10
        global_amax = x.abs().max()

        ours = cuda_fp4(x, global_amax.unsqueeze(0), block_size)
        theirs = fp4_fake_quant_block(
            x,
            global_amax=global_amax,
            block_size=block_size,
            tile_rows=num_blocks,
            tile_cols=block_size,
        )

        assert torch.allclose(ours, theirs, atol=1e-5), (
            f"experimental vs modelopt Triton max diff: {(ours - theirs).abs().max().item():.6e}"
        )

    @requires_cuda_ext_mx
    @pytest.mark.parametrize("block_size", [16, 32])
    @pytest.mark.parametrize("seed", [42, 123])
    def test_vs_cuda_ext_mx(self, cuda_fp4, block_size, seed):
        """Compare against modelopt cuda_ext_mx.fused_amax_convert."""
        from modelopt.torch.quantization.extensions import get_cuda_ext_mx
        from modelopt.torch.quantization.tensor_quant import mx_format_map

        cuda_ext_mx = get_cuda_ext_mx()
        torch.manual_seed(seed)
        num_blocks = 16
        x = torch.randn(num_blocks, block_size, device="cuda", dtype=torch.float32) * 10
        global_amax = x.abs().max()

        ours = cuda_fp4(x, global_amax.unsqueeze(0), block_size)
        theirs = cuda_ext_mx.fused_amax_convert(
            x,
            block_size,
            getattr(cuda_ext_mx.Types, mx_format_map[(2, 1)]),
            getattr(cuda_ext_mx.Types, mx_format_map[(4, 3)]),
            global_amax,
        )

        assert torch.allclose(ours, theirs, atol=1e-5), (
            f"experimental vs modelopt cuda_ext_mx max diff: "
            f"{(ours - theirs).abs().max().item():.6e}"
        )

    @requires_dynamic_block_quantize
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_vs_dynamic_block_quantize_op(self, cuda_fp4, seed):
        """Compare against modelopt dynamic_block_quantize_op (high-level API).

        This is the function used by the actual quantization pipeline with
        num_bits=4 (E2M1) and scale_bits=8 (E4M3).
        Note: dynamic_block_quantize_op dispatches to Triton with default block_size=16.
        """
        from modelopt.torch.quantization.tensor_quant import dynamic_block_quantize_op

        block_size = 16  # dynamic_block_quantize_op uses block_size=16 for Triton path
        torch.manual_seed(seed)
        num_blocks = 16
        x = torch.randn(num_blocks, block_size, device="cuda", dtype=torch.float32) * 10
        global_amax = x.abs().max()

        ours = cuda_fp4(x, global_amax.unsqueeze(0), block_size)
        theirs = dynamic_block_quantize_op(
            x,
            block_size,
            global_amax,
            num_bits=4,  # total bits = 1 sign + 2 exp + 1 mantissa
            exponent_bits=2,
            scale_num_bits=8,  # FP8 E4M3 for scales
            scale_exponent_bits=4,
        )

        assert torch.allclose(ours, theirs, atol=1e-5), (
            f"experimental vs modelopt dynamic_block_quantize_op max diff: "
            f"{(ours - theirs).abs().max().item():.6e}"
        )

    @requires_triton_fp4
    def test_vs_triton_realistic_shape(self, cuda_fp4):
        """Realistic activation shape from a Conv3D layer (flattened)."""
        torch.manual_seed(42)
        block_size = 16
        # Simulate a large tensor: 256 blocks of 16 elements
        # (tile_rows must be power-of-2 for Triton block_ptr)
        num_blocks = 256
        x = torch.randn(num_blocks, block_size, device="cuda", dtype=torch.float32) * 5
        global_amax = x.abs().max()

        from modelopt.torch.quantization.triton import fp4_fake_quant_block

        ours = cuda_fp4(x, global_amax.unsqueeze(0), block_size)
        theirs = fp4_fake_quant_block(
            x,
            global_amax=global_amax,
            block_size=block_size,
            tile_rows=16,
            tile_cols=block_size,
        )

        max_diff = (ours - theirs).abs().max().item()
        mean_diff = (ours - theirs).abs().mean().item()
        assert torch.allclose(ours, theirs, atol=1e-5), (
            f"Realistic shape: experimental vs Triton max diff: {max_diff:.6e}, "
            f"mean diff: {mean_diff:.6e}"
        )

    @requires_triton_fp4
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_vs_triton_input_dtypes(self, cuda_fp4, dtype):
        """Test that our kernel handles different input dtypes correctly.

        Our kernel casts to float32 internally, so the result should match
        Triton's output when both receive the same dtype input.
        """
        from modelopt.torch.quantization.triton import fp4_fake_quant_block

        torch.manual_seed(42)
        block_size = 16
        num_blocks = 8
        x = (torch.randn(num_blocks, block_size, device="cuda") * 5).to(dtype)
        global_amax = x.float().abs().max()

        ours = cuda_fp4(x, global_amax.unsqueeze(0), block_size)
        theirs = fp4_fake_quant_block(
            x,
            global_amax=global_amax,
            block_size=block_size,
            tile_rows=num_blocks,
            tile_cols=block_size,
        )

        # Both should return the input dtype
        assert ours.dtype == dtype
        assert theirs.dtype == dtype

        # Compare in float32
        max_diff = (ours.float() - theirs.float()).abs().max().item()
        # BF16/FP16 input rounding may cause small diffs
        tol = 1e-2 if dtype != torch.float32 else 1e-5
        assert max_diff < tol, f"dtype={dtype}: experimental vs Triton max diff: {max_diff:.6e}"
