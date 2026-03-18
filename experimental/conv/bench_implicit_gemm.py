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

"""Latency benchmark: implicit GEMM (quant / non-quant) vs cuDNN conv3d.

Usage:
    python -m experimental.conv.bench_implicit_gemm
    python -m experimental.conv.bench_implicit_gemm --shapes wan22
    python -m experimental.conv.bench_implicit_gemm --shapes all --warmup 20 --iters 100
"""

import argparse

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Benchmark shapes
# ---------------------------------------------------------------------------

# (name, N, Cin, D, H, W, Cout, kD, kH, kW, stride, padding, dilation)
SHAPES = {
    "small": [
        ("small_16x32_3x3x3", 1, 16, 8, 8, 8, 32, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ],
    "medium": [
        ("med_64x128_3x3x3", 1, 64, 16, 32, 32, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ("med_128x256_3x3x3", 1, 128, 8, 16, 16, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ("med_128x128_1x3x3", 1, 128, 16, 32, 32, 128, 1, 3, 3, (1, 1, 1), (0, 1, 1), (1, 1, 1)),
    ],
    "wan22": [
        ("wan22_128x512", 1, 128, 21, 60, 106, 512, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ("wan22_512x512", 1, 512, 21, 60, 106, 512, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ("wan22_512x128", 1, 512, 21, 60, 106, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ],
    "stride": [
        ("stride2_64x128", 1, 64, 16, 32, 32, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1), (1, 1, 1)),
        ("stride2_128x256", 1, 128, 16, 32, 32, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    ],
}


def get_shapes(name: str):
    """Return list of benchmark shapes by name or all shapes."""
    if name == "all":
        result = []
        for v in SHAPES.values():
            result.extend(v)
        return result
    return SHAPES[name]


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def bench_fn(fn, warmup: int, iters: int) -> float:
    """Benchmark a callable, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmark(shapes_name: str, warmup: int, iters: int, fp4_block_size: int):
    """Run latency benchmark for the given shapes."""
    from experimental.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda

    shapes = get_shapes(shapes_name)

    # Header
    print(f"\n{'=' * 100}")
    print(
        f"Conv3D Latency Benchmark  |  warmup={warmup}  iters={iters}  fp4_block_size={fp4_block_size}"
    )
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'=' * 100}")
    print(
        f"{'Shape':<25} {'M':>10} {'K':>8} {'N':>6} "
        f"{'cuDNN':>9} {'GEMM':>9} {'GEMM+FP4':>9} "
        f"{'GEMM/cuDNN':>11} {'FP4/cuDNN':>10}"
    )
    print("-" * 100)

    for name, n, cin, d, h, w, cout, kd, kh, kw, stride, padding, dilation in shapes:
        torch.manual_seed(42)
        x = torch.randn(n, cin, d, h, w, device="cuda", dtype=torch.float32)
        weight = torch.randn(cout, cin, kd, kh, kw, device="cuda", dtype=torch.float32)
        act_amax = x.abs().max().unsqueeze(0)

        # Compute GEMM dimensions for display
        sd, sh, sw = stride
        dd, dh, dw = dilation
        pd, ph, pw = padding
        od = (d + 2 * pd - dd * (kd - 1) - 1) // sd + 1
        oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        gemm_m = n * od * oh * ow
        gemm_k = cin * kd * kh * kw
        gemm_n = cout

        # cuDNN (torch.nn.functional.conv3d)
        t_cudnn = bench_fn(
            lambda: F.conv3d(x, weight, stride=stride, padding=padding, dilation=dilation),
            warmup,
            iters,
        )

        # Implicit GEMM (non-quantized)
        t_gemm = bench_fn(
            lambda: conv3d_implicit_gemm_cuda(
                x,
                weight,
                stride=stride,
                padding=padding,
                dilation=dilation,
                quant_act=False,
                fp4_block_size=fp4_block_size,
            ),
            warmup,
            iters,
        )

        # Implicit GEMM (FP4 quantized)
        t_fp4 = bench_fn(
            lambda: conv3d_implicit_gemm_cuda(
                x,
                weight,
                stride=stride,
                padding=padding,
                dilation=dilation,
                act_amax=act_amax,
                quant_act=True,
                fp4_block_size=fp4_block_size,
            ),
            warmup,
            iters,
        )

        ratio_gemm = t_gemm / t_cudnn
        ratio_fp4 = t_fp4 / t_cudnn

        print(
            f"{name:<25} {gemm_m:>10,} {gemm_k:>8,} {gemm_n:>6,} "
            f"{t_cudnn:>8.3f}ms {t_gemm:>8.3f}ms {t_fp4:>8.3f}ms "
            f"{ratio_gemm:>10.2f}x {ratio_fp4:>9.2f}x"
        )

    print(f"{'=' * 100}")
    print("Ratios > 1.0x mean slower than cuDNN; < 1.0x mean faster.")
    print()


def main():
    """Entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(description="Conv3D latency benchmark")
    parser.add_argument(
        "--shapes",
        default="all",
        choices=[*list(SHAPES.keys()), "all"],
        help="Which shape set to benchmark (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--fp4-block-size",
        type=int,
        default=128,
        choices=[128, 256],
        help="FP4 block size (default: 128)",
    )
    args = parser.parse_args()

    run_benchmark(args.shapes, args.warmup, args.iters, args.fp4_block_size)


if __name__ == "__main__":
    main()
