#!/usr/bin/env python3
"""Systematic accuracy + speed benchmark for attention quantization on Wan2.2.

Sweeps all kernel variants × resolutions × prompts, producing CSV tables
comparable to the MXFP8 cuDNN reference study (REPORT.html / REPORT_PROMPTS.html).

Output structure (under ``--output-dir``)::

    results/
        accuracy.csv          # Per-prompt accuracy: kernel, resolution, prompt_id, LPIPS, PSNR, SSIM
        accuracy_summary.csv  # Aggregate: kernel, resolution, LPIPS_mean/min/max, PSNR_mean, SSIM_mean
        timing.csv            # E2E timing: kernel, resolution, prompt_id, elapsed_s, speedup
        timing_summary.csv    # Aggregate: kernel, resolution, mean_elapsed_s, mean_speedup
        videos/               # All generated videos: {prompt_id}_{kernel}_{resolution}.mp4

Usage::

    # Full sweep — all kernels, both resolutions, 10 prompts (requires B200)
    python benchmark_sweep.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --output-dir results/

    # Quick test — single kernel, single resolution, 2 prompts
    python benchmark_sweep.py --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --kernels nvfp4 nvfp4-v3 --resolutions 480p --prompts 0 1 --output-dir results/

    # Speed-only (skip accuracy, no baseline generation per prompt)
    python benchmark_sweep.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --speed-only --output-dir results/

Requirements::

    pip install diffusers transformers accelerate ftfy lpips scikit-image sageattention
"""

import argparse
import csv
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Prompt suite — matches the MXFP8 reference study (10 prompts)
# ---------------------------------------------------------------------------

PROMPT_SUITE = [
    {"id": "cat_windowsill", "prompt": "A cat sitting still on a windowsill", "failure_mode": "static detail"},
    {"id": "busy_street", "prompt": "A person walking across a busy street", "failure_mode": "many subjects, motion"},
    {"id": "ocean_sunset", "prompt": "Ocean waves crashing on rocks at sunset", "failure_mode": "texture, color"},
    {"id": "clouds_timelapse", "prompt": "Clouds drifting across the sky (timelapse)", "failure_mode": "slow motion"},
    {"id": "dancer_jump", "prompt": "A dancer performing a spinning jump", "failure_mode": "fast motion"},
    {"id": "flower_blooming", "prompt": "Close-up of a flower blooming", "failure_mode": "fine detail"},
    {"id": "drone_city_night", "prompt": "Drone shot over a city at night", "failure_mode": "lighting, complexity"},
    {"id": "text_hello", "prompt": "HELLO written on a chalkboard", "failure_mode": "text (edge case)"},
    {"id": "ball_bouncing", "prompt": "A ball bouncing on a table", "failure_mode": "physics"},
    {"id": "empty_room_sun", "prompt": "Empty room with sunlight through window", "failure_mode": "minimal content"},
]

RESOLUTION_CONFIGS = {
    "720p": {"height": 720, "width": 1280, "num_frames": 81, "label": "720×1280 / 81f"},
    "480p": {"height": 480, "width": 832, "num_frames": 33, "label": "480×832 / 33f"},
}

# Kernel categories
SDPA_KERNELS = ["fp8", "sage1", "sage2-fp16", "sage2-fp8"]
MODELOPT_KERNELS = ["triton-sparse", "triton-skip", "nvfp4", "nvfp4-v3"]
ALL_KERNELS = SDPA_KERNELS + MODELOPT_KERNELS

DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text, cropped, overexposed"


# ---------------------------------------------------------------------------
# Pipeline management — reload for in-place kernels
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str):
    """Load Wan2.2 pipeline. Returns (pipe, model_id)."""
    from diffusers import AutoencoderKLWan, WanPipeline

    print(f"[Pipeline] Loading VAE (fp32) from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    print("[Pipeline] Loading transformer + text encoder (bf16)...")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def apply_kernel(pipe, kernel: str):
    """Apply a kernel to the pipeline. For ModelOpt kernels, modifies transformer in-place."""
    # Import from the example script in the same directory
    from wan2_sage_attention import (
        enable_attention_kernel,
        apply_triton_sparse_kernel,
        KERNEL_NVFP4,
        KERNEL_NVFP4_V3,
        _TRITON_MODELOPT_KERNELS,
    )

    if kernel == KERNEL_NVFP4:
        from modelopt.torch.quantization import apply_sage_attention
        apply_sage_attention(pipe.transformer)
    elif kernel == KERNEL_NVFP4_V3:
        from modelopt.torch.quantization import apply_sage_attention_v3
        apply_sage_attention_v3(pipe.transformer)
    elif kernel in _TRITON_MODELOPT_KERNELS:
        apply_triton_sparse_kernel(pipe.transformer, kernel)
    else:
        enable_attention_kernel(kernel)


def cleanup_kernel(kernel: str):
    """Cleanup after SDPA-patching kernels (not needed for ModelOpt in-place)."""
    from wan2_sage_attention import disable_attention_kernel, _TRITON_MODELOPT_KERNELS, KERNEL_NVFP4, KERNEL_NVFP4_V3

    if kernel not in _TRITON_MODELOPT_KERNELS and kernel not in (KERNEL_NVFP4, KERNEL_NVFP4_V3):
        disable_attention_kernel()


def generate_video(pipe, prompt: str, height: int, width: int, num_frames: int,
                   num_steps: int = 40, guidance_scale: float = 4.0, seed: int = 42):
    """Generate a video and return (elapsed_s, frames)."""
    generator = torch.Generator("cuda").manual_seed(seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    frames = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=generator,
    ).frames[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed, frames


def save_video(frames, path: str):
    """Save frames to mp4."""
    from diffusers.utils import export_to_video
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, path, fps=16)


# ---------------------------------------------------------------------------
# Metrics (delegates to wan2_sage_attention)
# ---------------------------------------------------------------------------

def compute_metrics(frames_ref, frames_quant):
    """Compute LPIPS, PSNR, SSIM between reference and quantized frames."""
    from wan2_sage_attention import compute_video_metrics
    return compute_video_metrics(frames_ref, frames_quant)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_accuracy_sweep(pipe, model_id, kernels, resolutions, prompt_indices,
                       output_dir, num_steps, seed):
    """Run accuracy sweep: baseline + each kernel, compute metrics per prompt."""
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    accuracy_rows = []
    timing_rows = []

    for res_name in resolutions:
        res = RESOLUTION_CONFIGS[res_name]
        h, w, nf = res["height"], res["width"], res["num_frames"]
        print(f"\n{'='*70}")
        print(f"Resolution: {res['label']} ({res_name})")
        print(f"{'='*70}")

        # --- Generate baselines for all prompts at this resolution ---
        baselines = {}  # prompt_id -> (elapsed, frames)
        for pi in prompt_indices:
            p = PROMPT_SUITE[pi]
            print(f"\n[baseline] {p['id']}: \"{p['prompt']}\"")
            elapsed, frames = generate_video(pipe, p["prompt"], h, w, nf,
                                             num_steps=num_steps, seed=seed)
            video_path = str(video_dir / f"{p['id']}_baseline_{res_name}.mp4")
            save_video(frames, video_path)
            print(f"  {elapsed:.1f}s -> {video_path}")
            baselines[p["id"]] = (elapsed, frames)

            timing_rows.append({
                "kernel": "baseline",
                "resolution": res_name,
                "prompt_id": p["id"],
                "elapsed_s": f"{elapsed:.2f}",
                "speedup": "1.00",
            })

        # --- Run each kernel ---
        for kernel in kernels:
            is_inplace = kernel in MODELOPT_KERNELS
            if is_inplace:
                # Reload pipeline for in-place kernels
                print(f"\n[{kernel}] Reloading pipeline for in-place kernel...")
                del pipe
                torch.cuda.empty_cache()
                pipe = load_pipeline(model_id)

            apply_kernel(pipe, kernel)

            for pi in prompt_indices:
                p = PROMPT_SUITE[pi]
                print(f"\n[{kernel}] {p['id']}: \"{p['prompt']}\"")
                elapsed, frames = generate_video(pipe, p["prompt"], h, w, nf,
                                                 num_steps=num_steps, seed=seed)
                video_path = str(video_dir / f"{p['id']}_{kernel}_{res_name}.mp4")
                save_video(frames, video_path)

                base_elapsed = baselines[p["id"]][0]
                speedup = base_elapsed / elapsed if elapsed > 0 else 0

                print(f"  {elapsed:.1f}s (speedup {speedup:.2f}x) -> {video_path}")

                # Compute accuracy vs baseline
                metrics = compute_metrics(baselines[p["id"]][1], frames)
                lpips_str = f"{metrics.get('lpips', float('nan')):.4f}"
                psnr_str = f"{metrics['psnr']:.2f}"
                ssim_str = f"{metrics['ssim']:.4f}"
                print(f"  LPIPS={lpips_str}  PSNR={psnr_str}dB  SSIM={ssim_str}")

                accuracy_rows.append({
                    "kernel": kernel,
                    "resolution": res_name,
                    "prompt_id": p["id"],
                    "lpips": lpips_str,
                    "psnr": psnr_str,
                    "ssim": ssim_str,
                    "mae_pct": f"{metrics['mae_pct']:.4f}",
                    "cos_sim": f"{metrics['cos_sim']:.6f}",
                })

                timing_rows.append({
                    "kernel": kernel,
                    "resolution": res_name,
                    "prompt_id": p["id"],
                    "elapsed_s": f"{elapsed:.2f}",
                    "speedup": f"{speedup:.2f}",
                })

            if not is_inplace:
                cleanup_kernel(kernel)
            else:
                # For the next kernel, we'll reload anyway
                pass

    return accuracy_rows, timing_rows, pipe


def run_speed_sweep(pipe, model_id, kernels, resolutions, prompt_indices,
                    output_dir, num_steps, seed):
    """Speed-only sweep: time each kernel, no baseline frames retained."""
    timing_rows = []

    for res_name in resolutions:
        res = RESOLUTION_CONFIGS[res_name]
        h, w, nf = res["height"], res["width"], res["num_frames"]
        print(f"\n{'='*70}")
        print(f"Resolution: {res['label']} ({res_name}) — speed only")
        print(f"{'='*70}")

        # Baseline timing
        base_times = {}
        for pi in prompt_indices:
            p = PROMPT_SUITE[pi]
            print(f"\n[baseline] {p['id']}")
            elapsed, _ = generate_video(pipe, p["prompt"], h, w, nf,
                                        num_steps=num_steps, seed=seed)
            base_times[p["id"]] = elapsed
            print(f"  {elapsed:.1f}s")
            timing_rows.append({
                "kernel": "baseline",
                "resolution": res_name,
                "prompt_id": p["id"],
                "elapsed_s": f"{elapsed:.2f}",
                "speedup": "1.00",
            })

        for kernel in kernels:
            is_inplace = kernel in MODELOPT_KERNELS
            if is_inplace:
                print(f"\n[{kernel}] Reloading pipeline...")
                del pipe
                torch.cuda.empty_cache()
                pipe = load_pipeline(model_id)

            apply_kernel(pipe, kernel)

            for pi in prompt_indices:
                p = PROMPT_SUITE[pi]
                print(f"\n[{kernel}] {p['id']}")
                elapsed, _ = generate_video(pipe, p["prompt"], h, w, nf,
                                            num_steps=num_steps, seed=seed)
                speedup = base_times[p["id"]] / elapsed if elapsed > 0 else 0
                print(f"  {elapsed:.1f}s (speedup {speedup:.2f}x)")

                timing_rows.append({
                    "kernel": kernel,
                    "resolution": res_name,
                    "prompt_id": p["id"],
                    "elapsed_s": f"{elapsed:.2f}",
                    "speedup": f"{speedup:.2f}",
                })

            if not is_inplace:
                cleanup_kernel(kernel)

    return timing_rows, pipe


# ---------------------------------------------------------------------------
# CSV I/O + summary
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] Wrote {len(rows)} rows -> {path}")


def compute_accuracy_summary(rows: list[dict]) -> list[dict]:
    """Aggregate accuracy rows into per-kernel-per-resolution summary."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kernel"], r["resolution"])].append(r)

    summary = []
    for (kernel, res), group in sorted(groups.items()):
        lpips_vals = [float(r["lpips"]) for r in group if r["lpips"] != "nan"]
        psnr_vals = [float(r["psnr"]) for r in group]
        ssim_vals = [float(r["ssim"]) for r in group]

        best_prompt = min(group, key=lambda r: float(r["lpips"]) if r["lpips"] != "nan" else 999)
        worst_prompt = max(group, key=lambda r: float(r["lpips"]) if r["lpips"] != "nan" else -1)

        summary.append({
            "kernel": kernel,
            "resolution": res,
            "lpips_mean": f"{np.mean(lpips_vals):.4f}" if lpips_vals else "n/a",
            "lpips_min": f"{np.min(lpips_vals):.4f}" if lpips_vals else "n/a",
            "lpips_max": f"{np.max(lpips_vals):.4f}" if lpips_vals else "n/a",
            "psnr_mean": f"{np.mean(psnr_vals):.2f}",
            "ssim_mean": f"{np.mean(ssim_vals):.4f}",
            "best_prompt": f"{best_prompt['prompt_id']} ({best_prompt['lpips']})",
            "worst_prompt": f"{worst_prompt['prompt_id']} ({worst_prompt['lpips']})",
        })
    return summary


def compute_timing_summary(rows: list[dict]) -> list[dict]:
    """Aggregate timing rows."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kernel"], r["resolution"])].append(r)

    summary = []
    for (kernel, res), group in sorted(groups.items()):
        elapsed_vals = [float(r["elapsed_s"]) for r in group]
        speedup_vals = [float(r["speedup"]) for r in group]
        summary.append({
            "kernel": kernel,
            "resolution": res,
            "mean_elapsed_s": f"{np.mean(elapsed_vals):.2f}",
            "mean_speedup": f"{np.mean(speedup_vals):.2f}",
            "min_elapsed_s": f"{np.min(elapsed_vals):.2f}",
            "max_elapsed_s": f"{np.max(elapsed_vals):.2f}",
        })
    return summary


def print_summary_table(summary_rows: list[dict], title: str):
    """Pretty-print a summary table to stdout."""
    if not summary_rows:
        return
    keys = list(summary_rows[0].keys())
    widths = {k: max(len(k), max(len(str(r[k])) for r in summary_rows)) for k in keys}

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for r in summary_rows:
        line = "  ".join(str(r[k]).ljust(widths[k]) for k in keys)
        print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Systematic attention quantization benchmark on Wan2.2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                        help="HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for CSVs and videos")
    parser.add_argument("--kernels", nargs="+", default=ALL_KERNELS,
                        choices=ALL_KERNELS, help="Kernels to benchmark")
    parser.add_argument("--resolutions", nargs="+", default=["720p", "480p"],
                        choices=list(RESOLUTION_CONFIGS.keys()),
                        help="Resolutions to sweep")
    parser.add_argument("--prompts", nargs="+", type=int, default=None,
                        help="Prompt indices (0-9). Default: all 10.")
    parser.add_argument("--num-steps", type=int, default=40, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed-only", action="store_true",
                        help="Skip accuracy metrics, only measure timing")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_indices = args.prompts if args.prompts is not None else list(range(len(PROMPT_SUITE)))
    for pi in prompt_indices:
        if pi < 0 or pi >= len(PROMPT_SUITE):
            print(f"ERROR: prompt index {pi} out of range (0-{len(PROMPT_SUITE)-1})")
            sys.exit(1)

    print(f"Model:       {args.model}")
    print(f"Kernels:     {args.kernels}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Prompts:     {[PROMPT_SUITE[i]['id'] for i in prompt_indices]}")
    print(f"Steps:       {args.num_steps}")
    print(f"Seed:        {args.seed}")
    print(f"Output:      {output_dir}")
    print(f"Mode:        {'speed-only' if args.speed_only else 'accuracy + speed'}")

    pipe = load_pipeline(args.model)

    if args.speed_only:
        timing_rows, pipe = run_speed_sweep(
            pipe, args.model, args.kernels, args.resolutions,
            prompt_indices, output_dir, args.num_steps, args.seed,
        )
        accuracy_rows = []
    else:
        accuracy_rows, timing_rows, pipe = run_accuracy_sweep(
            pipe, args.model, args.kernels, args.resolutions,
            prompt_indices, output_dir, args.num_steps, args.seed,
        )

    # Write CSVs
    if accuracy_rows:
        write_csv(
            output_dir / "accuracy.csv", accuracy_rows,
            ["kernel", "resolution", "prompt_id", "lpips", "psnr", "ssim", "mae_pct", "cos_sim"],
        )
        acc_summary = compute_accuracy_summary(accuracy_rows)
        write_csv(
            output_dir / "accuracy_summary.csv", acc_summary,
            ["kernel", "resolution", "lpips_mean", "lpips_min", "lpips_max",
             "psnr_mean", "ssim_mean", "best_prompt", "worst_prompt"],
        )
        print_summary_table(acc_summary, "Accuracy Summary (vs bf16 baseline)")

    if timing_rows:
        write_csv(
            output_dir / "timing.csv", timing_rows,
            ["kernel", "resolution", "prompt_id", "elapsed_s", "speedup"],
        )
        time_summary = compute_timing_summary(timing_rows)
        write_csv(
            output_dir / "timing_summary.csv", time_summary,
            ["kernel", "resolution", "mean_elapsed_s", "mean_speedup",
             "min_elapsed_s", "max_elapsed_s"],
        )
        print_summary_table(time_summary, "Timing Summary")

    print("\nDone! All results saved to:", output_dir)


if __name__ == "__main__":
    main()
