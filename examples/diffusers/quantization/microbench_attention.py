#!/usr/bin/env python3
"""Kernel-only microbenchmark for attention variants on Wan2.2 shapes.

Measures isolated kernel latency (CUDA events) at Wan2.2 self-attention shapes,
comparable to the MXFP8 cuDNN study (REPORT.html §3b).

Shapes from Wan-2.2 T2V-A14B (B=1, H=40, D=128):
  S=4680   480×832 / 9f   (warmup shape)
  S=14040  480×832 / 33f  (480p production)
  S=27000  720×1280 / 33f
  S=75600  720×1280 / 81f (720p production)

FLOPS formula: 4 × B × H × S² × D  (2 matmuls, each counted as 2 flops per mul-add).
At B=1, H=40, D=128 → 20480 × S² flops per call.

Usage::

    # All variants at all shapes
    python microbench_attention.py

    # Specific shapes and variants
    python microbench_attention.py --shapes 14040 75600 --variants baseline nvfp4 nvfp4-v3

    # CSV output
    python microbench_attention.py --csv results/microbench.csv

    # Custom warmup/iters
    python microbench_attention.py --warmup 10 --iters 50
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

# Wan2.2 T2V-A14B self-attention config
B, H, D = 1, 40, 128

# Shapes from the reference study
SHAPES = {
    4680: "480×832 / 9f (warmup)",
    14040: "480×832 / 33f",
    27000: "720×1280 / 33f",
    75600: "720×1280 / 81f",
}

# B200 theoretical SoL for reference
B200_SOL_BF16_TFLOPS = 2250
B200_SOL_FP8_TFLOPS = 4500


def flops_per_call(S: int) -> int:
    """4 × B × H × S² × D."""
    return 4 * B * H * S * S * D


def tflops(S: int, latency_ms: float) -> float:
    return flops_per_call(S) / (latency_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Kernel variants
# ---------------------------------------------------------------------------

def _bench_baseline(q, k, v, warmup, iters):
    """Standard bf16 SDPA (cuDNN / FlashAttention auto-dispatch)."""
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        _ = F.scaled_dot_product_attention(q, k, v)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sorted(times)


def _bench_triton(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len,
                  warmup, iters, **kwargs):
    """Generic runner for the ModelOpt Triton attention kernel."""
    from modelopt.torch.kernels.triton_fa import attention as triton_attention

    def call():
        return triton_attention(
            q_flat, k_flat, v_flat, b_start_loc, b_seq_len,
            max_input_len=max_len, is_causal=False, **kwargs,
        )

    for _ in range(warmup):
        _ = call()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        _ = call()
        end_events[i].record()

    torch.cuda.synchronize()
    return sorted([s.elapsed_time(e) for s, e in zip(start_events, end_events)])


def _bench_triton_baseline(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters):
    """ModelOpt Triton flash-attention kernel (no quantization, no sparsity)."""
    return _bench_triton(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters)


def _bench_triton_nvfp4_p(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters):
    """ModelOpt Triton flash-attention with NVFP4 P-matrix quantization."""
    return _bench_triton(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters,
                         quantize_p=True)


def _bench_triton_nvfp4_v3(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters):
    """ModelOpt Triton flash-attention with SageAttn v3 (Q/K/V/P NVFP4)."""
    return _bench_triton(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters,
                         quantize_qkv=True)


def _bench_triton_sparse(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters):
    """Triton flash-attention with 2:4 sparse softmax."""
    return _bench_triton(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, max_len, warmup, iters,
                         sparsity_n=2, sparsity_m=4)


def _bench_sage_kernels(q, k, v, warmup, iters):
    """SageAttention CUDA kernels (sage1, sage2-fp16, sage2-fp8)."""
    results = {}
    try:
        import sageattention
    except ImportError:
        return results

    sage_fns = {}
    if hasattr(sageattention, "sageattn"):
        sage_fns["sage1"] = lambda q, k, v: sageattention.sageattn(q, k, v, tensor_layout="HND")
    if hasattr(sageattention, "sageattn_qk_int8_pv_fp16_cuda"):
        sage_fns["sage2-fp16"] = lambda q, k, v: sageattention.sageattn_qk_int8_pv_fp16_cuda(
            q, k, v, tensor_layout="HND", qk_quant_gran="per_thread", smooth_k=True,
        )
    if hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda"):
        sage_fns["sage2-fp8"] = lambda q, k, v: sageattention.sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16", smooth_k=True,
        )

    for name, fn in sage_fns.items():
        try:
            for _ in range(warmup):
                _ = fn(q, k, v)
            torch.cuda.synchronize()

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

            for i in range(iters):
                start_events[i].record()
                _ = fn(q, k, v)
                end_events[i].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            results[name] = sorted(times)
        except (RuntimeError, AssertionError) as e:
            print(f"  [{name}] FAILED: {e}")

    return results


# All Triton-based variants
TRITON_VARIANTS = {
    "triton-baseline": _bench_triton_baseline,
    "nvfp4": _bench_triton_nvfp4_p,
    "nvfp4-v3": _bench_triton_nvfp4_v3,
    "triton-sparse": _bench_triton_sparse,
}


def run_microbench(shapes, variants, warmup, iters):
    """Run microbench for all shapes × variants. Returns list of result dicts."""
    results = []

    for S in shapes:
        label = SHAPES.get(S, f"S={S}")
        flops = flops_per_call(S)
        print(f"\n{'='*70}")
        print(f"S = {S:,}  ({label})  |  {flops/1e9:.1f} GFLOP per call")
        print(f"{'='*70}")

        # Create random inputs
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

        # Flat layout for Triton kernel: [total_tokens, num_heads, head_dim]
        # Input is [B, H, S, D], need [S, H, D] (B=1)
        q_flat = q.squeeze(0).permute(1, 0, 2).contiguous()  # [H, S, D] -> [S, H, D]
        k_flat = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_flat = v.squeeze(0).permute(1, 0, 2).contiguous()
        b_start_loc = torch.tensor([0], dtype=torch.int64, device="cuda")
        b_seq_len = torch.tensor([S], dtype=torch.int64, device="cuda")

        # Baseline SDPA
        baseline_median = None
        if "baseline" in variants:
            times = _bench_baseline(q, k, v, warmup, iters)
            baseline_median = times[len(times) // 2]
            tf = tflops(S, baseline_median)
            print(f"  {'baseline (SDPA)':<20} median={baseline_median:.3f}ms  min={times[0]:.3f}ms  "
                  f"p10={times[len(times)//10]:.3f}ms  p90={times[int(len(times)*0.9)]:.3f}ms  "
                  f"TFLOPS={tf:.0f}")
            results.append({
                "S": S, "variant": "baseline", "median_ms": f"{baseline_median:.3f}",
                "min_ms": f"{times[0]:.3f}", "tflops": f"{tf:.0f}",
                "speedup": "1.00",
            })

        # Triton variants
        for name, fn in TRITON_VARIANTS.items():
            if name not in variants:
                continue
            try:
                times = fn(q_flat, k_flat, v_flat, b_start_loc, b_seq_len, S, warmup, iters)
                median = times[len(times) // 2]
                tf = tflops(S, median)
                speedup = baseline_median / median if baseline_median else 0
                print(f"  {name:<20} median={median:.3f}ms  min={times[0]:.3f}ms  "
                      f"p10={times[len(times)//10]:.3f}ms  p90={times[int(len(times)*0.9)]:.3f}ms  "
                      f"TFLOPS={tf:.0f}  speedup={speedup:.2f}x")
                results.append({
                    "S": S, "variant": name, "median_ms": f"{median:.3f}",
                    "min_ms": f"{times[0]:.3f}", "tflops": f"{tf:.0f}",
                    "speedup": f"{speedup:.2f}",
                })
            except Exception as e:
                print(f"  {name:<20} FAILED: {e}")

        # Sage kernels
        sage_variants_to_run = [v for v in variants if v.startswith("sage")]
        if sage_variants_to_run:
            sage_results = _bench_sage_kernels(q, k, v, warmup, iters)
            for name, times in sage_results.items():
                if name not in variants:
                    continue
                median = times[len(times) // 2]
                tf = tflops(S, median)
                speedup = baseline_median / median if baseline_median else 0
                print(f"  {name:<20} median={median:.3f}ms  min={times[0]:.3f}ms  "
                      f"p10={times[len(times)//10]:.3f}ms  p90={times[int(len(times)*0.9)]:.3f}ms  "
                      f"TFLOPS={tf:.0f}  speedup={speedup:.2f}x")
                results.append({
                    "S": S, "variant": name, "median_ms": f"{median:.3f}",
                    "min_ms": f"{times[0]:.3f}", "tflops": f"{tf:.0f}",
                    "speedup": f"{speedup:.2f}",
                })

        # Free memory
        del q, k, v, q_flat, k_flat, v_flat
        torch.cuda.empty_cache()

    return results


def print_summary_table(results):
    """Print a cross-backend comparison table per shape."""
    from collections import defaultdict
    by_shape = defaultdict(list)
    for r in results:
        by_shape[r["S"]].append(r)

    all_variants = []
    seen = set()
    for r in results:
        if r["variant"] not in seen:
            all_variants.append(r["variant"])
            seen.add(r["variant"])

    print(f"\n{'='*90}")
    print(f"  Kernel-only microbenchmark summary (B={B}, H={H}, D={D})")
    print(f"{'='*90}")

    header = f"  {'S':>8}"
    for v in all_variants:
        header += f"  |  {v:>12} ms  {'spdup':>5}  {'TFLOPS':>6}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in sorted(by_shape.keys()):
        by_variant = {r["variant"]: r for r in by_shape[S]}
        line = f"  {S:>8}"
        for v in all_variants:
            if v in by_variant:
                r = by_variant[v]
                line += f"  |  {r['median_ms']:>12}  {r['speedup']:>5}x  {r['tflops']:>6}"
            else:
                line += f"  |  {'N/A':>12}  {'N/A':>5}   {'N/A':>6}"
        print(line)
    print()


def parse_args():
    all_variant_names = ["baseline"] + list(TRITON_VARIANTS.keys()) + ["sage1", "sage2-fp16", "sage2-fp8"]

    parser = argparse.ArgumentParser(
        description="Kernel-only microbenchmark for attention variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shapes", nargs="+", type=int, default=list(SHAPES.keys()),
                        help="Sequence lengths to benchmark")
    parser.add_argument("--variants", nargs="+", default=all_variant_names,
                        choices=all_variant_names,
                        help="Kernel variants to benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV file")
    return parser.parse_args()


def main():
    args = parse_args()

    sm = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{sm[0]*10+sm[1]})")
    print(f"Shapes: {args.shapes}")
    print(f"Variants: {args.variants}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")

    results = run_microbench(args.shapes, args.variants, args.warmup, args.iters)

    print_summary_table(results)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["S", "variant", "median_ms", "min_ms", "tflops", "speedup"])
            w.writeheader()
            w.writerows(results)
        print(f"[CSV] Wrote {len(results)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
