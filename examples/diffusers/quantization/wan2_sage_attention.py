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

"""Wan2.2 Text-to-Video inference with SageAttention.

SageAttention (https://arxiv.org/pdf/2410.02367) replaces
``F.scaled_dot_product_attention`` with a faster INT8/FP8 kernel:

- Q and K are quantized to INT8 (K is smoothed via mean-subtraction first)
- V stays in FP16 or FP8 depending on the kernel variant
- ~2× faster attention than FlashAttention2 with negligible accuracy loss

Three kernel variants are supported via ``--kernel``:

``sage1``
    ``sageattn`` — SageAttention v1 auto-selector. INT8 QK, FP16 PV.
    Works on Ampere, Ada, Hopper.

``sage2-fp16``
    ``sageattn_qk_int8_pv_fp16_cuda`` — SageAttention2, INT8 QK, FP16 PV.
    Per-thread quantization granularity. Faster than sage1 on Ada/Hopper.

``sage2-fp8`` (default on Ada/Hopper)
    ``sageattn_qk_int8_pv_fp8_cuda`` — SageAttention2++. INT8 QK, FP8 PV
    with two-stage FP32+FP16 accumulator. Fastest option on Ada (SM89) and
    Hopper (SM90).

Requirements::

    pip install sageattention diffusers transformers accelerate ftfy

Usage::

    # SageAttention2++ (default, fastest on Ada)
    python wan2_sage_attention.py --prompt "Two cats boxing on a stage"

    # Explicit kernel selection
    python wan2_sage_attention.py --prompt "..." --kernel sage2-fp16

    # Baseline — standard SDPA (uses flash_attn if installed)
    python wan2_sage_attention.py --prompt "..." --baseline

    # Benchmark all kernels back-to-back
    python wan2_sage_attention.py --prompt "..." --benchmark

    # Smaller 5B model (fits on a single 24 GB GPU)
    python wan2_sage_attention.py \\
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --prompt "Two cats boxing on a stage"
"""

import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# Model IDs available on HuggingFace Hub
MODEL_T2V_14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

DEFAULT_MODEL = MODEL_T2V_14B
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text, cropped, overexposed"

# Kernel choices
KERNEL_SAGE1 = "sage1"
KERNEL_SAGE2_FP16 = "sage2-fp16"
KERNEL_SAGE2_FP8 = "sage2-fp8"
KERNEL_CHOICES = [KERNEL_SAGE1, KERNEL_SAGE2_FP16, KERNEL_SAGE2_FP8]

_KERNEL_DESCRIPTIONS = {
    KERNEL_SAGE1: "sageattn (SA1, INT8 QK + FP16 PV, auto-select)",
    KERNEL_SAGE2_FP16: "sageattn_qk_int8_pv_fp16_cuda (SA2, INT8 QK + FP16 PV, per-thread)",
    KERNEL_SAGE2_FP8: "sageattn_qk_int8_pv_fp8_cuda (SA2++, INT8 QK + FP8 PV, fp32+fp16 accum)",
}


def _detect_available_kernels() -> list[str]:
    """Return the list of kernels available in the installed sageattention version."""
    try:
        import sageattention as _sa
    except ImportError:
        return []

    available = []
    if hasattr(_sa, "sageattn"):
        available.append(KERNEL_SAGE1)
    if hasattr(_sa, "sageattn_qk_int8_pv_fp16_cuda"):
        available.append(KERNEL_SAGE2_FP16)
    if hasattr(_sa, "sageattn_qk_int8_pv_fp8_cuda"):
        available.append(KERNEL_SAGE2_FP8)
    return available


# Populated once at startup; used to warn and skip unavailable kernels.
AVAILABLE_KERNELS: list[str] = _detect_available_kernels()


# ---------------------------------------------------------------------------
# SageAttention patching
# ---------------------------------------------------------------------------

# Keep a reference to the original SDPA so we can restore it and fall back
# when SageAttention cannot handle a particular call.
_orig_sdpa = F.scaled_dot_product_attention

# Active kernel name — set by enable_sage_attention()
_active_kernel: str = KERNEL_SAGE2_FP8

# Call counters — reset by enable_sage_attention()
_sage_calls: int = 0
_fallback_calls: int = 0


def _run_sage_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """Dispatch to the selected SageAttention kernel."""
    if _active_kernel == KERNEL_SAGE1:
        from sageattention import sageattn

        return sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)

    elif _active_kernel == KERNEL_SAGE2_FP16:
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        return sageattn_qk_int8_pv_fp16_cuda(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran="per_thread",
            sm_scale=scale,
            smooth_k=True,
        )

    else:  # KERNEL_SAGE2_FP8
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        return sageattn_qk_int8_pv_fp8_cuda(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran="per_thread",
            sm_scale=scale,
            pv_accum_dtype="fp32+fp16",  # SA2++ two-stage accumulator, fastest on Ada
            smooth_k=True,
        )


def _sageattn_sdpa_compat(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for ``F.scaled_dot_product_attention`` using SageAttention.

    Falls back to standard SDPA when SageAttention cannot be applied:
    - ``attn_mask`` is not None (SageAttention does not support arbitrary masks)
    - ``dropout_p > 0`` (training-time dropout)
    - Input dtype is not FP16 or BF16 (e.g. FP32 during VAE encode/decode)
    """
    global _sage_calls, _fallback_calls
    if (
        attn_mask is not None
        or dropout_p > 0.0
        or query.dtype not in (torch.float16, torch.bfloat16)
    ):
        _fallback_calls += 1
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    _sage_calls += 1
    return _run_sage_kernel(query, key, value, is_causal=is_causal, scale=scale)


def enable_sage_attention(kernel: str = KERNEL_SAGE2_FP8) -> None:
    """Patch ``F.scaled_dot_product_attention`` globally with SageAttention.

    Args:
        kernel: One of ``"sage1"``, ``"sage2-fp16"``, ``"sage2-fp8"``.
            If the requested kernel is not available in the installed
            sageattention version, falls back to ``"sage1"``.
    """
    global _active_kernel, _sage_calls, _fallback_calls
    if not AVAILABLE_KERNELS:
        raise ImportError("SageAttention is not installed. Run: pip install sageattention")

    if kernel not in KERNEL_CHOICES:
        raise ValueError(f"Unknown kernel {kernel!r}. Choose from {KERNEL_CHOICES}")

    if kernel not in AVAILABLE_KERNELS:
        print(
            f"[SageAttention] WARNING: kernel={kernel!r} not available in installed "
            f"sageattention version (available: {AVAILABLE_KERNELS}). Falling back to sage1."
        )
        kernel = KERNEL_SAGE1

    _active_kernel = kernel
    _sage_calls = 0
    _fallback_calls = 0

    F.scaled_dot_product_attention = _sageattn_sdpa_compat
    import torch.nn.functional as _F

    _F.scaled_dot_product_attention = _sageattn_sdpa_compat
    print(f"[SageAttention] kernel={kernel}  {_KERNEL_DESCRIPTIONS[kernel]}")


def disable_sage_attention() -> None:
    """Restore the original ``F.scaled_dot_product_attention``."""
    F.scaled_dot_product_attention = _orig_sdpa
    import torch.nn.functional as _F

    _F.scaled_dot_product_attention = _orig_sdpa


@contextmanager
def sage_attention_ctx(kernel: str = KERNEL_SAGE2_FP8):
    """Context manager that enables SageAttention for the duration of the block."""
    enable_sage_attention(kernel)
    try:
        yield
    finally:
        disable_sage_attention()


def print_sage_stats() -> None:
    """Print how many attention calls went through sageattn vs fallback."""
    total = _sage_calls + _fallback_calls
    print(
        f"[SageAttention] calls: {_sage_calls} sageattn, {_fallback_calls} fallback (total {total})"
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def load_pipeline(model_id: str):
    """Load the Wan2.2 pipeline.

    The VAE must be loaded in FP32 to avoid numerical issues; the transformer
    and text encoder use BF16.
    """
    from diffusers import AutoencoderKLWan, WanPipeline

    print(f"[Pipeline] Loading VAE (fp32) from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    print("[Pipeline] Loading transformer + text encoder (bf16)...")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def run_inference(pipe, args, label: str = "") -> float:
    """Run one generation pass and return wall-clock time in seconds."""
    generator = torch.Generator("cuda").manual_seed(args.seed)

    if label:
        print(f"\n[{label}] Generating {args.num_frames} frames @ {args.height}x{args.width}...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output = pipe(
        prompt=args.prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=generator,
    ).frames[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    from diffusers.utils import export_to_video

    out_path = args.output if not label else args.output.replace(".mp4", f"_{label}.mp4")
    export_to_video(output, out_path, fps=16)
    print(f"[{label or 'result'}] Saved to {out_path}  ({elapsed:.1f}s)")
    return elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 T2V inference with SageAttention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=[MODEL_T2V_14B, MODEL_TI2V_5B],
        help="HuggingFace model ID",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num-steps", type=int, default=40, help="Denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--kernel",
        type=str,
        default=KERNEL_SAGE2_FP8,
        choices=KERNEL_CHOICES,
        help=(
            "SageAttention kernel variant. "
            "sage1: SA1 auto-select; "
            "sage2-fp16: SA2 INT8+FP16; "
            "sage2-fp8: SA2++ INT8+FP8 (fastest on Ada/Hopper)"
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run with standard SDPA instead of SageAttention",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run baseline + all three SageAttention kernels back-to-back",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipe = load_pipeline(args.model)

    if args.benchmark:
        results: dict[str, float] = {}

        # Baseline
        results["baseline"] = run_inference(pipe, args, label="baseline")

        # Available sage kernels
        for kernel in KERNEL_CHOICES:
            if kernel not in AVAILABLE_KERNELS:
                print(f"\n[{kernel}] Skipped — not available in installed sageattention version")
                continue
            enable_sage_attention(kernel)
            results[kernel] = run_inference(pipe, args, label=kernel)
            print_sage_stats()
            disable_sage_attention()

        t_base = results["baseline"]
        print(f"\n{'=' * 55}")
        print(f"  {'Kernel':<20} {'Time':>8}   {'Speedup':>8}")
        print(f"  {'-' * 40}")
        print(f"  {'baseline (SDPA)':<20} {t_base:>7.1f}s   {'1.00x':>8}")
        for kernel in KERNEL_CHOICES:
            if kernel not in results:
                print(f"  {kernel:<20} {'N/A':>8}   {'N/A':>8}  (not available)")
                continue
            t = results[kernel]
            print(f"  {kernel:<20} {t:>7.1f}s   {t_base / t:>7.2f}x")
        print(f"{'=' * 55}")

    elif args.baseline:
        run_inference(pipe, args, label="baseline")

    else:
        enable_sage_attention(args.kernel)
        run_inference(pipe, args, label=args.kernel)
        print_sage_stats()
        disable_sage_attention()


if __name__ == "__main__":
    main()
