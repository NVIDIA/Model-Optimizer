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

"""Wan2.2 Text-to-Video inference with SageAttention and FP8 attention quantization.

Four attention kernel variants are supported via ``--kernel``:

``sage1``
    ``sageattn`` — SageAttention v1. INT8 QK, FP16 PV. Ampere/Ada/Hopper.

``sage2-fp16``
    ``sageattn_qk_int8_pv_fp16_cuda`` — SageAttention2. INT8 QK, FP16 PV, per-thread.

``sage2-fp8`` (default SageAttention kernel)
    ``sageattn_qk_int8_pv_fp8_cuda`` — SageAttention2++. INT8 QK, FP8 PV.
    Fastest on Ada (SM89) / Hopper (SM90).

``fp8`` (default for accuracy testing, always available)
    Python-level FP8 E4M3 attention. Inspired by SageAttention2:
    - Q, K, V are channel-smoothed (per-channel mean subtraction) before quantization
    - Per-token FP8 E4M3 quantization with max-based scales
    - Dequantize back to BF16, then run standard SDPA
    - No CUDA kernel required. Use for accuracy verification on any GPU.

``nvfp4`` (always available)
    Python-level NVFP4 E2M1 attention. Inspired by SageAttention3:
    - Q and K: channel-smoothed, per-token scaled, rounded to nearest of 8 levels {0,0.5,1,1.5,2,3,4,6}
    - V: quantized to FP8 E4M3 (448 levels vs 8 for FP4 — preserves content quality)
    - No CUDA kernel required. Use to measure accuracy vs FP8.

Requirements::

    pip install sageattention diffusers transformers accelerate ftfy

Usage::

    # FP8 accuracy check vs baseline (CLIP score + pixel metrics)
    python wan2_sage_attention.py --prompt "..." --compare

    # Single run with FP8 attention
    python wan2_sage_attention.py --prompt "..." --kernel fp8

    # Compare a specific kernel vs baseline with accuracy metrics
    python wan2_sage_attention.py --prompt "..." --kernel sage2-fp16 --compare

    # Baseline — standard SDPA
    python wan2_sage_attention.py --prompt "..." --baseline

    # Benchmark all kernels (timing only)
    python wan2_sage_attention.py --prompt "..." --benchmark

    # Smaller 5B model (fits on a single 24 GB GPU)
    python wan2_sage_attention.py \\
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --prompt "Two cats boxing on a stage"
"""

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

# Model IDs available on HuggingFace Hub
MODEL_T2V_14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

DEFAULT_MODEL = MODEL_TI2V_5B
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text, cropped, overexposed"

# Kernel choices
KERNEL_FP8 = "fp8"
KERNEL_NVFP4 = "nvfp4"
KERNEL_SAGE1 = "sage1"
KERNEL_SAGE2_FP16 = "sage2-fp16"
KERNEL_SAGE2_FP8 = "sage2-fp8"
KERNEL_CHOICES = [KERNEL_FP8, KERNEL_NVFP4, KERNEL_SAGE1, KERNEL_SAGE2_FP16, KERNEL_SAGE2_FP8]

_KERNEL_DESCRIPTIONS = {
    KERNEL_FP8: "FP8 E4M3 QKV (Python-level, SA2-inspired smoothing, no CUDA kernel required)",
    KERNEL_NVFP4: "NVFP4 E2M1 QK + FP8 E4M3 V (Python-level, SA3-inspired)",
    KERNEL_SAGE1: "sageattn (SA1, INT8 QK + FP16 PV, auto-select)",
    KERNEL_SAGE2_FP16: "sageattn_qk_int8_pv_fp16_cuda (SA2, INT8 QK + FP16 PV, per-thread)",
    KERNEL_SAGE2_FP8: "sageattn_qk_int8_pv_fp8_cuda (SA2++, INT8 QK + FP8 PV, fp32+fp16 accum)",
}

# SageAttention CUDA kernel support by GPU compute capability:
#   SM80  Ampere  (A100, RTX 3090)              sage1, sage2-fp16
#   SM89  Ada     (RTX 4090, RTX PRO 6000 Ada)  sage1, sage2-fp16, sage2-fp8
#   SM90  Hopper  (H100)                        sage1, sage2-fp16, sage2-fp8
#   SM100 Blackwell datacenter (B100/B200)      sage1, sage2-fp16, sage2-fp8
#   SM120 Blackwell consumer (RTX 50-series,
#          RTX PRO 6000 Blackwell)              NOT supported by SA 2.2.0
#                                               fp8 kernel always works (pure Python)
_SUPPORTED_SM = {80, 86, 89, 90, 100}

# FP8 max value for float8_e4m3fn
_FP8_MAX = 448.0

# NVFP4 E2M1 representable positive values and their rounding boundaries.
# Format: 1 sign + 2 exponent + 1 mantissa bit, exponent bias = 1.
# Positive levels:  0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
# Rounding boundaries (midpoints between adjacent levels):
#                   0.25 0.75 1.25 1.75 2.5  3.5  5.0
_FP4_MAX = 6.0
_FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_FP4_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def _get_gpu_sm() -> int | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _fp8_available() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def _detect_available_kernels() -> list[str]:
    """Return kernels available given the installed packages and GPU."""
    available = []

    # fp8 and nvfp4 are pure Python — available on any GPU / PyTorch version
    if _fp8_available():
        available.append(KERNEL_FP8)
    else:
        print(
            "[FP8] WARNING: torch.float8_e4m3fn not found. "
            "Upgrade to PyTorch >= 2.1 to use the fp8 kernel."
        )
    # nvfp4 uses only standard float ops — always available
    available.append(KERNEL_NVFP4)

    try:
        import sageattention as _sa
    except ImportError:
        return available

    sm = _get_gpu_sm()
    if sm is not None and sm not in _SUPPORTED_SM:
        print(
            f"[SageAttention] WARNING: GPU SM{sm} not officially supported by SA 2.2.0 "
            f"(supported: SM{sorted(_SUPPORTED_SM)}). CUDA kernels may fail. "
            "Try: TORCH_CUDA_ARCH_LIST='8.9+PTX' pip install --no-cache-dir sageattention"
        )

    if hasattr(_sa, "sageattn"):
        available.append(KERNEL_SAGE1)
    if hasattr(_sa, "sageattn_qk_int8_pv_fp16_cuda"):
        available.append(KERNEL_SAGE2_FP16)
    if hasattr(_sa, "sageattn_qk_int8_pv_fp8_cuda"):
        available.append(KERNEL_SAGE2_FP8)
    return available


AVAILABLE_KERNELS: list[str] = _detect_available_kernels()


# ---------------------------------------------------------------------------
# FP8 attention — Python-level, SA2-inspired
# ---------------------------------------------------------------------------


def _smooth_quantize_fp8(x: torch.Tensor) -> torch.Tensor:
    """Smooth + FP8-quantize + dequantize a Q, K, or V tensor.

    Implements the channel-wise mean-subtraction smoothing from SageAttention2
    (arXiv 2411.10958):

    1. Subtract per-channel mean across the token dimension (removes systematic
       outliers in each head-dim channel, compressing the dynamic range).
    2. Quantize the zero-centred tensor to FP8 E4M3 with a per-token max scale.
    3. Dequantize back to the original dtype.
    4. Add the channel mean back so the result is mathematically equivalent
       to the original, up to FP8 rounding error.

    Args:
        x: Attention tensor with layout ``(B, H, N, D)`` — batch, heads,
           tokens, head-dim.

    Returns:
        Tensor of the same shape and dtype as ``x``, simulating FP8 precision.
    """
    orig_dtype = x.dtype

    # Step 1 — channel smoothing: mean over token dimension → (B, H, 1, D)
    mean = x.mean(dim=-2, keepdim=True)
    x_smooth = x - mean

    # Step 2 — per-token scale: max over head-dim → (B, H, N, 1)
    scale = x_smooth.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / _FP8_MAX

    # Step 3 — quantize to FP8 E4M3, then immediately dequantize
    x_fp8 = (x_smooth.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    x_dq = x_fp8.to(orig_dtype) * scale.to(orig_dtype)

    # Step 4 — restore mean
    return x_dq + mean


def _fp8_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """FP8 E4M3 attention with SA2-inspired Q/K/V smoothing.

    All three matrices are independently smoothed and FP8-quantized before
    being passed to standard SDPA. This simulates the precision loss of a
    true FP8 attention kernel without requiring any compiled CUDA code.
    """
    q_dq = _smooth_quantize_fp8(query)
    k_dq = _smooth_quantize_fp8(key)
    v_dq = _smooth_quantize_fp8(value)
    return _orig_sdpa(q_dq, k_dq, v_dq, is_causal=is_causal, scale=scale)


# ---------------------------------------------------------------------------
# NVFP4 attention — Python-level, SA3-inspired
# ---------------------------------------------------------------------------


def _quantize_to_fp4_levels(x_abs: torch.Tensor) -> torch.Tensor:
    """Round absolute values to the nearest NVFP4 E2M1 representable level.

    Uses vectorised ``torch.bucketize`` against the 7 midpoint boundaries
    between the 8 positive FP4 levels ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``.
    """
    boundaries = torch.tensor(_FP4_BOUNDARIES, dtype=x_abs.dtype, device=x_abs.device)
    levels = torch.tensor(_FP4_LEVELS, dtype=x_abs.dtype, device=x_abs.device)
    idx = torch.bucketize(x_abs.contiguous(), boundaries)
    return levels[idx]


def _smooth_quantize_fp4(x: torch.Tensor) -> torch.Tensor:
    """Smooth + NVFP4 E2M1-quantize + dequantize a Q, K, or V tensor.

    Mirrors the SA3 (arXiv 2505.11594) channel-wise smoothing strategy:

    1. Subtract per-channel mean across the token dimension.
    2. Scale so the per-token max maps to ``_FP4_MAX`` (6.0).
    3. Round to the nearest NVFP4 E2M1 level via a boundary lookup.
    4. Dequantize back to the original dtype and restore the mean.

    No CUDA kernel required — works on any GPU or CPU.

    Args:
        x: Attention tensor ``(B, H, N, D)``.

    Returns:
        Tensor of the same shape and dtype, simulating NVFP4 E2M1 precision.
    """
    orig_dtype = x.dtype
    x = x.float()

    # Step 1 — channel smoothing
    mean = x.mean(dim=-2, keepdim=True)
    x_smooth = x - mean

    # Step 2 — per-token scale to fit in [-FP4_MAX, FP4_MAX]
    scale = x_smooth.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _FP4_MAX
    x_scaled = (x_smooth / scale).clamp(-_FP4_MAX, _FP4_MAX)

    # Step 3 — quantize: preserve sign, round magnitude to nearest FP4 level
    sign = x_scaled.sign()
    x_dq = sign * _quantize_to_fp4_levels(x_scaled.abs())

    # Step 4 — dequantize and restore mean
    return (x_dq * scale + mean).to(orig_dtype)


def _nvfp4_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """NVFP4 E2M1 attention: Q and K quantized to NVFP4, V kept at original precision.

    In SageAttention3 (arXiv 2505.11594), NVFP4 is applied to the post-softmax
    attention probability matrix P = softmax(QK^T / sqrt(d)), not to V directly.
    V is multiplied with the NVFP4-quantized P in BF16/FP16 precision.

    Quantizing V to FP4 causes severe color/texture loss because V carries the
    actual content that gets weighted and summed into the output.  Q and K are
    only used for computing routing weights, so their quantization noise is
    far less visible.  V is quantized to FP8 E4M3 instead, which preserves
    content quality while still reducing precision beyond BF16.
    """
    q_dq = _smooth_quantize_fp4(query)
    k_dq = _smooth_quantize_fp4(key)
    v_dq = _smooth_quantize_fp8(value)
    return _orig_sdpa(q_dq, k_dq, v_dq, is_causal=is_causal, scale=scale)


# ---------------------------------------------------------------------------
# SDPA patching
# ---------------------------------------------------------------------------

_orig_sdpa = F.scaled_dot_product_attention
_active_kernel: str = KERNEL_FP8
_sage_calls: int = 0
_fallback_calls: int = 0


def _run_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    if _active_kernel == KERNEL_FP8:
        return _fp8_sdpa(query, key, value, is_causal=is_causal, scale=scale)

    if _active_kernel == KERNEL_NVFP4:
        return _nvfp4_sdpa(query, key, value, is_causal=is_causal, scale=scale)

    if _active_kernel == KERNEL_SAGE1:
        from sageattention import sageattn

        return sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)

    if _active_kernel == KERNEL_SAGE2_FP16:
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

    # KERNEL_SAGE2_FP8
    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    return sageattn_qk_int8_pv_fp8_cuda(
        query,
        key,
        value,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran="per_thread",
        sm_scale=scale,
        pv_accum_dtype="fp32+fp16",
        smooth_k=True,
    )


def _patched_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs,
) -> torch.Tensor:
    global _sage_calls, _fallback_calls
    # Fall back to standard SDPA for unsupported cases
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
    try:
        return _run_kernel(query, key, value, is_causal=is_causal, scale=scale)
    except (AssertionError, RuntimeError) as e:
        print(f"[Attention] WARNING: kernel={_active_kernel!r} failed ({e}). Falling back to SDPA.")
        return _orig_sdpa(query, key, value, is_causal=is_causal, scale=scale)


def enable_attention_kernel(kernel: str = KERNEL_FP8) -> None:
    """Patch ``F.scaled_dot_product_attention`` with the selected kernel."""
    global _active_kernel, _sage_calls, _fallback_calls

    if kernel not in KERNEL_CHOICES:
        raise ValueError(f"Unknown kernel {kernel!r}. Choose from {KERNEL_CHOICES}")
    if kernel not in AVAILABLE_KERNELS:
        raise RuntimeError(f"Kernel {kernel!r} is not available. Available: {AVAILABLE_KERNELS}")

    _active_kernel = kernel
    _sage_calls = 0
    _fallback_calls = 0

    F.scaled_dot_product_attention = _patched_sdpa
    import torch.nn.functional as _F

    _F.scaled_dot_product_attention = _patched_sdpa

    print(f"[Attention] kernel={kernel}  {_KERNEL_DESCRIPTIONS[kernel]}")


def disable_attention_kernel() -> None:
    F.scaled_dot_product_attention = _orig_sdpa
    import torch.nn.functional as _F

    _F.scaled_dot_product_attention = _orig_sdpa


@contextmanager
def attention_kernel_ctx(kernel: str = KERNEL_FP8):
    enable_attention_kernel(kernel)
    try:
        yield
    finally:
        disable_attention_kernel()


def print_kernel_stats() -> None:
    total = _sage_calls + _fallback_calls
    print(f"[Attention] calls: {_sage_calls} quantized, {_fallback_calls} fallback (total {total})")


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------


def _frames_to_uint8(frames: list) -> np.ndarray:
    """Convert a list of PIL images to a uint8 numpy array of shape (N, H, W, 3)."""
    import numpy as np

    arrays = []
    for f in frames:
        if isinstance(f, np.ndarray):
            arr = f if f.dtype == np.uint8 else (f * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = np.array(f.convert("RGB"), dtype=np.uint8)
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def compute_video_metrics(
    frames_ref: list,
    frames_quant: list,
) -> dict[str, float]:
    """Compute frame-level accuracy metrics between two video frame sequences.

    Metrics:
        psnr      Peak Signal-to-Noise Ratio (dB). Higher = better.
                  >40 dB: excellent (barely noticeable).
                  30-40 dB: good.
                  20-30 dB: noticeable but acceptable.
        mae_pct   Mean Absolute Error as % of max pixel value (255). Lower = better.
        cos_sim   Mean cosine similarity of flattened frames. Closer to 1 = better.

    Args:
        frames_ref:   List of PIL images from the baseline run.
        frames_quant: List of PIL images from the quantized run.

    Returns:
        Dict with keys ``"psnr"``, ``"mae_pct"``, ``"cos_sim"``.
    """
    ref = _frames_to_uint8(frames_ref).astype(np.float32)  # (N, H, W, 3)
    quant = _frames_to_uint8(frames_quant).astype(np.float32)

    # PSNR
    mse_per_frame = ((ref - quant) ** 2).mean(axis=(1, 2, 3))  # (N,)
    # Avoid log(0) for identical frames
    psnr_per_frame = np.where(
        mse_per_frame < 1e-10,
        100.0,
        10.0 * np.log10(255.0**2 / mse_per_frame),
    )
    psnr = float(psnr_per_frame.mean())

    # MAE as % of 255
    mae_pct = float(np.abs(ref - quant).mean() / 255.0 * 100.0)

    # Cosine similarity: flatten each frame to a vector
    ref_flat = ref.reshape(ref.shape[0], -1)
    quant_flat = quant.reshape(quant.shape[0], -1)
    dot = (ref_flat * quant_flat).sum(axis=1)
    norm_ref = np.linalg.norm(ref_flat, axis=1)
    norm_quant = np.linalg.norm(quant_flat, axis=1)
    cos_sim = float((dot / (norm_ref * norm_quant + 1e-12)).mean())

    return {"psnr": psnr, "mae_pct": mae_pct, "cos_sim": cos_sim}


def compute_clip_score(
    frames: list,
    prompt: str,
    clip_model_id: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    max_frames: int = 16,
) -> float:
    """Compute mean CLIP score (text-image cosine similarity) over video frames.

    Samples up to ``max_frames`` evenly from the sequence and returns the
    average cosine similarity between the CLIP text embedding of ``prompt``
    and the CLIP image embedding of each frame.  Higher = more semantically
    aligned with the prompt.

    Args:
        frames:        List of PIL images (the generated video).
        prompt:        The text prompt used to generate the video.
        clip_model_id: HuggingFace model ID or local path for CLIP.
                       Pass ``HF_TOKEN`` env var for authenticated downloads.
        device:        Device for the CLIP model (``"cuda"`` or ``"cpu"``).
        max_frames:    Maximum number of frames to score (evenly sampled).

    Returns:
        Mean CLIP cosine similarity score in ``[-1, 1]``.
        Typical values for good text-video alignment: ~0.15-0.30
        (varies by model and prompt; compare baseline vs quantized delta).
    """
    from transformers import CLIPModel, CLIPProcessor

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = CLIPProcessor.from_pretrained(clip_model_id, token=token)
    clip_model = CLIPModel.from_pretrained(clip_model_id, token=token).to(device)
    clip_model.eval()

    # Evenly sample frames
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
    sampled = [frames[int(i)] for i in indices]

    with torch.no_grad():
        text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = []
        for frame in sampled:
            img_inputs = processor(images=frame, return_tensors="pt").to(device)
            img_feat = clip_model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            scores.append((text_feat * img_feat).sum().item())

    # Free CLIP model from GPU memory
    del clip_model
    torch.cuda.empty_cache()

    return float(np.mean(scores))


def print_metrics(metrics: dict[str, float], label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Accuracy vs baseline:")
    print(f"  PSNR:         {metrics['psnr']:.2f} dB  (>40 excellent, 30-40 good, <30 noticeable)")
    print(f"  MAE:          {metrics['mae_pct']:.4f}%  of max pixel value")
    print(f"  Cosine sim:   {metrics['cos_sim']:.6f}  (1.0 = identical)")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def load_pipeline(model_id: str):
    """Load the Wan2.2 pipeline (VAE in FP32, transformer + text encoder in BF16)."""
    from diffusers import AutoencoderKLWan, WanPipeline

    print(f"[Pipeline] Loading VAE (fp32) from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    print("[Pipeline] Loading transformer + text encoder (bf16)...")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def run_inference(pipe, args, label: str = "") -> tuple[float, list]:
    """Run one generation pass.

    Returns:
        (elapsed_seconds, frames) where frames is the list of PIL images.
    """
    generator = torch.Generator("cuda").manual_seed(args.seed)

    if label:
        print(f"\n[{label}] Generating {args.num_frames} frames @ {args.height}x{args.width}...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    frames = pipe(
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
    export_to_video(frames, out_path, fps=16)
    print(f"[{label or 'result'}] Saved to {out_path}  ({elapsed:.1f}s)")
    return elapsed, frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 T2V with quantized attention (FP8, SageAttention)",
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
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=40, help="Denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kernel",
        type=str,
        default=KERNEL_FP8,
        choices=KERNEL_CHOICES,
        help=(
            "fp8: Python-level FP8 E4M3 (SA2 smoothing, no CUDA kernel, accuracy testing); "
            "sage1: SA1 INT8+FP16; "
            "sage2-fp16: SA2 INT8+FP16; "
            "sage2-fp8: SA2++ INT8+FP8"
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run with standard SDPA, no quantization",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run baseline + selected kernel, then report accuracy metrics "
            "(PSNR, MAE, cosine similarity). Default kernel is fp8."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run baseline + all available kernels, report timing table",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help=(
            "CLIP model ID or local path for --compare CLIP scoring. "
            "Set HF_TOKEN env var for authenticated HuggingFace downloads."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = load_pipeline(args.model)

    if args.compare:
        # --- Baseline ---
        _, frames_base = run_inference(pipe, args, label="baseline")

        # --- Quantized ---
        enable_attention_kernel(args.kernel)
        _, frames_quant = run_inference(pipe, args, label=args.kernel)
        print_kernel_stats()
        disable_attention_kernel()

        # --- CLIP scores (per-video semantic alignment with prompt) ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\nComputing CLIP scores (prompt-video semantic alignment)...")
        try:
            clip_base = compute_clip_score(
                frames_base, args.prompt, clip_model_id=args.clip_model, device=device
            )
            clip_quant = compute_clip_score(
                frames_quant, args.prompt, clip_model_id=args.clip_model, device=device
            )
            print(f"  baseline CLIP:  {clip_base:.4f}")
            print(f"  {args.kernel} CLIP:  {clip_quant:.4f}  (delta {clip_quant - clip_base:+.4f})")
            print(
                "  (absolute value varies by model; focus on the delta between baseline and quantized)"
            )
            print(
                "  Tip: set HF_TOKEN env var or use --clip-model <local-path> to avoid rate limits"
            )
        except OSError as e:
            print(f"  WARNING: CLIP scoring failed ({e})")
            print("  To fix: set HF_TOKEN env var or pass --clip-model <local-path-to-clip>")

        # --- Pixel-level metrics ---
        metrics = compute_video_metrics(frames_base, frames_quant)
        print_metrics(metrics, label=args.kernel)

    elif args.benchmark:
        timing: dict[str, float] = {}

        timing["baseline"], _ = run_inference(pipe, args, label="baseline")

        for kernel in KERNEL_CHOICES:
            if kernel not in AVAILABLE_KERNELS:
                print(f"\n[{kernel}] Skipped — not available")
                continue
            enable_attention_kernel(kernel)
            timing[kernel], _ = run_inference(pipe, args, label=kernel)
            print_kernel_stats()
            disable_attention_kernel()

        t_base = timing["baseline"]
        print(f"\n{'=' * 55}")
        print(f"  {'Kernel':<20} {'Time':>8}   {'Speedup':>8}")
        print(f"  {'-' * 40}")
        print(f"  {'baseline (SDPA)':<20} {t_base:>7.1f}s   {'1.00x':>8}")
        for kernel in KERNEL_CHOICES:
            if kernel not in timing:
                print(f"  {kernel:<20} {'N/A':>8}   {'N/A':>8}  (not available)")
                continue
            t = timing[kernel]
            print(f"  {kernel:<20} {t:>7.1f}s   {t_base / t:>7.2f}x")
        print(f"{'=' * 55}")

    elif args.baseline:
        run_inference(pipe, args, label="baseline")

    else:
        enable_attention_kernel(args.kernel)
        run_inference(pipe, args, label=args.kernel)
        print_kernel_stats()
        disable_attention_kernel()


if __name__ == "__main__":
    main()
