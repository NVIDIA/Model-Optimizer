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

"""Wan 2.2 inference with skip-softmax sparse attention.

This example applies skip-softmax sparse attention to the Wan 2.2 video
generation model (text-to-video). Four modes are supported:

1. **Baseline** — pass ``--baseline`` for dense inference (default diffusers backend).
2. **Triton baseline** — pass ``--triton-baseline`` for dense Triton FA kernel
   (no skip-softmax, same kernel as sparse runs for apples-to-apples comparison).
3. **Fixed raw threshold** — pass ``--raw-threshold`` to supply a log2-space
   threshold directly to the Triton kernel. No calibration data is needed.
4. **Calibrated threshold** — pass ``--calibrate`` to run exponential-model
   calibration (``scale_factor = a * exp(b * target_sparsity)``).

During calibration, ``triton_skip_softmax`` with the Triton calibration kernel
collects sparsity statistics across multiple threshold trials. The fitted
exponential model then allows runtime control of the target sparsity ratio
without recalibration.

The Wan 2.2 5B model has 40 transformer blocks with self-attention (attn1)
and cross-attention (attn2). Only self-attention is sparsified.

Usage::

    # Baseline (dense, no sparsity)
    python wan22_skip_softmax.py --baseline --prompt "A cat playing piano" \\
        --output baseline.mp4

    # Fixed raw threshold (no calibration needed)
    python wan22_skip_softmax.py --raw-threshold -5.0 --report-avg-sparsity \\
        --prompt "A cat playing piano" --output out.mp4

    # With calibration
    python wan22_skip_softmax.py --calibrate --target-sparsity 0.25 \\
        --report-avg-sparsity --prompt "A cat playing piano" --output out.mp4
"""

import argparse
import gc
import os

import torch
from datasets import load_dataset
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

DEFAULT_MODEL_PATH = os.environ.get("WAN22_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

# fmt: off
# ruff: noqa: RUF001
DEFAULT_NEGATIVE_PROMPT = (  # Official Wan 2.2 negative prompt (Chinese)
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
# fmt: on

# Default threshold trials for calibration
DEFAULT_THRESHOLD_TRIALS = [
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
    2e-1,
    3e-1,
    5e-1,
    7e-1,
    8e-1,
    9e-1,
    9.9e-1,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan 2.2 video generation with skip-softmax sparse attention"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation (optional, skips generation if not set)",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Wan 2.2 model path or HF ID"
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of frames (must be 4k+1)"
    )
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num-steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument(
        "--guidance-scale", type=float, default=4.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=3.0,
        help="Second guidance scale for 14B dual-transformer model (ignored by 5B)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sparse attention options
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run dense inference with default diffusers backend (no sparsity)",
    )
    parser.add_argument(
        "--triton-baseline",
        action="store_true",
        help="Run dense inference with Triton FA kernel (no skip-softmax, "
        "apples-to-apples comparison with sparse runs)",
    )
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=None,
        help="Raw skip_threshold_log2 value passed directly to the Triton kernel. "
        "Negative values (e.g., -5.0 means tile must be within 5 units of running max). "
        "Bypasses calibration and lambda conversion. Typical range: -1 to -30.",
    )
    parser.add_argument(
        "--skip-first-last",
        type=int,
        default=2,
        help="Number of first/last transformer layers to keep dense (default: 2)",
    )
    parser.add_argument(
        "--report-avg-sparsity",
        action="store_true",
        help="Report per-layer and overall average tile sparsity after generation",
    )

    # Calibration options
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate threshold via exponential model (recommended)",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.5,
        help="Target sparsity ratio for calibration (0.0-1.0)",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=40,
        help="Inference steps for calibration",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=151,
        help="Number of frames for calibration",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=4,
        help="Number of calibration prompts from OpenVid-1M dataset",
    )
    return parser.parse_args()


def build_pipeline(model_path: str) -> WanPipeline:
    """Build the Wan 2.2 text-to-video pipeline."""
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def build_sparse_config(args: argparse.Namespace, num_blocks: int) -> dict:
    """Build sparse attention config from CLI args.

    Two modes:
    - **Raw threshold**: ``--raw-threshold`` sets ``skip_softmax_raw_threshold``
      directly on the Triton kernel — no calibration needed.
    - **Calibrated**: ``--calibrate`` collects multi-threshold sparsity statistics
      via the Triton calibration kernel, then fits an exponential model:
      ``scale_factor = a * exp(b * sparsity)``.
    """
    attn_cfg: dict = {
        "method": "triton_skip_softmax",
        "skip_softmax_threshold": 0.0 if args.triton_baseline else 0.1,
        "backend": "triton",
        "is_causal": False,  # Diffusion = bidirectional attention
        "collect_stats": True,
        "enable": True,
    }

    # Raw threshold bypasses calibration and lambda conversion
    if args.raw_threshold is not None:
        attn_cfg["skip_softmax_raw_threshold"] = args.raw_threshold

    sparse_cfg: dict = {
        "*.attn1*": attn_cfg,  # Self-attention only
        "*.attn2*": {"enable": False},  # Text cross-attention
        "default": {"enable": False},
    }

    # Keep first/last N layers dense for quality
    for i in range(args.skip_first_last):
        sparse_cfg[f"*blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*blocks.{num_blocks - 1 - i}.attn*"] = {"enable": False}

    config: dict = {"sparse_cfg": sparse_cfg}

    # Add calibration config only when calibrating (not with raw threshold)
    if args.calibrate and args.raw_threshold is None:
        sparse_cfg["calibration"] = {
            "target_sparse_ratio": {"prefill": args.target_sparsity},
            "threshold_trials": DEFAULT_THRESHOLD_TRIALS,
            "fit_logspace": True,
        }

    return config


def load_calib_prompts(calib_size: int) -> list[str]:
    """Load calibration prompts from OpenVid-1M dataset."""
    dataset = load_dataset("nkp37/OpenVid-1M", split="train")
    prompts = list(dataset["caption"][:calib_size])
    print(f"Loaded {len(prompts)} calibration prompts from OpenVid-1M")
    return prompts


def build_calibration_forward_loop(
    pipe: WanPipeline,
    calib_size: int = 4,
    num_steps: int = 40,
    num_frames: int = 151,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    guidance_scale: float = 4.0,
    guidance_scale_2: float | None = 3.0,
    negative_prompt: str = "",
):
    """Build a forward loop for exponential model calibration.

    Uses prompts from OpenVid-1M dataset (same as quantization examples).
    Each prompt is run individually (batch_size=1).
    """
    calib_prompts = load_calib_prompts(calib_size)

    def forward_loop(model):
        for i, prompt in enumerate(calib_prompts):
            print(f"Calibration [{i + 1}/{len(calib_prompts)}]: {prompt[:60]}...")
            kw: dict = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator(device="cuda").manual_seed(seed),
            }
            if guidance_scale_2 is not None:
                kw["guidance_scale_2"] = guidance_scale_2
            pipe(**kw)

    return forward_loop


def enable_sparsity_measurement(model: torch.nn.Module) -> None:
    """Enable runtime sparsity measurement on all sparse attention modules."""
    for _name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule) and module.is_enabled:
            method = module._sparse_method_instance
            if hasattr(method, "enable_measure_sparsity"):
                method.reset_sparsity_counters()
                method.enable_measure_sparsity(True)


def print_sparsity_summary(model: torch.nn.Module) -> None:
    """Print per-module sparsity statistics including runtime kernel counters."""
    enabled, disabled = [], []
    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule):
            if module.is_enabled:
                enabled.append((name, module))
            else:
                disabled.append(name)

    print(f"\nSparse attention: {len(enabled)} enabled, {len(disabled)} disabled")
    for name, module in enabled:
        info = module.get_threshold_info()
        print(f"  {name}: {info}")


def print_runtime_sparsity(model: torch.nn.Module) -> None:
    """Print runtime tile sparsity measured via kernel atomic counters."""
    total_all = 0
    skipped_all = 0
    per_module: list[tuple[str, int, int]] = []

    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule) and module.is_enabled:
            method = module._sparse_method_instance
            if hasattr(method, "get_sparsity_counters"):
                total, skipped = method.get_sparsity_counters()
                if total > 0:
                    per_module.append((name, total, skipped))
                    total_all += total
                    skipped_all += skipped

    if total_all == 0:
        print("\nNo runtime sparsity data collected.")
        return

    print("\n" + "=" * 70)
    print("Runtime tile sparsity (measured via kernel atomic counters)")
    print("=" * 70)
    for name, total, skipped in per_module:
        ratio = skipped / total
        print(f"  {name}: {skipped:,}/{total:,} tiles skipped ({ratio:.1%})")
    ratio_all = skipped_all / total_all
    print("-" * 70)
    print(f"  Overall: {skipped_all:,}/{total_all:,} tiles skipped ({ratio_all:.1%})")
    print("=" * 70)


def _get_num_blocks(transformer: torch.nn.Module) -> int:
    """Count transformer blocks by looking for *.blocks.N.* submodules."""
    max_idx = -1
    for name, _ in transformer.named_modules():
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                max_idx = max(max_idx, int(parts[i + 1]))
    if max_idx < 0:
        raise ValueError(
            "Could not detect transformer blocks (expected submodules matching *.blocks.N.*). "
            "Check that the model architecture uses 'blocks' as the layer container name."
        )
    return max_idx + 1


def main() -> None:
    args = parse_args()

    # ---- Build pipeline ----
    print(f"Loading Wan 2.2 from {args.model_path}...")
    pipe = build_pipeline(args.model_path)

    # ---- Collect transformers ----
    # Wan 2.2 5B has one transformer; 14B has two (transformer + transformer_2)
    transformers = []
    if pipe.transformer is not None:
        transformers.append(("transformer", pipe.transformer))
    if getattr(pipe, "transformer_2", None) is not None:
        transformers.append(("transformer_2", pipe.transformer_2))
    is_14b = len(transformers) > 1

    # ---- Sparsify (unless baseline) ----
    if args.baseline:
        print("Baseline mode: running dense inference (default diffusers backend)")
    elif args.triton_baseline:
        print("Triton baseline: dense Triton FA kernel (no skip-softmax)")
        for name, transformer in transformers:
            num_blocks = _get_num_blocks(transformer)
            print(f"Applying Triton backend to {name} ({num_blocks} blocks)...")
            config = build_sparse_config(args, num_blocks=num_blocks)
            mtsa.sparsify(transformer, config, forward_loop=None)
    else:
        # Build calibration forward loop if needed
        forward_loop = None
        if args.raw_threshold is not None:
            print(f"Using fixed raw threshold: {args.raw_threshold} (skipping calibration)")
            if args.calibrate:
                print("Warning: --calibrate is ignored when --raw-threshold is set")
        elif args.calibrate:
            forward_loop = build_calibration_forward_loop(
                pipe,
                calib_size=args.calib_size,
                num_steps=args.calib_steps,
                num_frames=args.calib_frames,
                height=args.height,
                width=args.width,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_2 if is_14b else None,
                negative_prompt=args.negative_prompt,
            )
        else:
            print(
                "Warning: neither --baseline, --raw-threshold, nor --calibrate specified; "
                "using default static threshold"
            )

        for name, transformer in transformers:
            num_blocks = _get_num_blocks(transformer)
            print(f"Applying skip-softmax to {name} ({num_blocks} blocks)...")
            config = build_sparse_config(args, num_blocks=num_blocks)
            mtsa.sparsify(transformer, config, forward_loop=forward_loop)

    # ---- Free calibration memory before inference ----
    if not args.baseline and not args.triton_baseline and forward_loop is not None:
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleared CUDA cache after calibration")

    # ---- Generate (optional) ----
    if args.prompt:
        # Enable runtime sparsity measurement before generation
        if args.report_avg_sparsity and not args.baseline:
            for _name, transformer in transformers:
                enable_sparsity_measurement(transformer)

        print(f"Generating: {args.prompt[:80]}...")
        pipe_kwargs: dict = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        }
        if is_14b and args.guidance_scale_2 is not None:
            pipe_kwargs["guidance_scale_2"] = args.guidance_scale_2
        output = pipe(**pipe_kwargs)

        try:
            export_to_video(output.frames[0], args.output, fps=16)
            print(f"Saved to {args.output}")
        except ImportError as exc:
            # Minimal CI envs may lack opencv/imageio — skip export silently,
            # the inference itself already ran successfully.
            print(f"Video export skipped (no opencv/imageio backend): {exc}")

    # ---- Print stats ----
    if not args.baseline:
        for name, transformer in transformers:
            print(f"\n{name}:")
            print_sparsity_summary(transformer)
            if args.report_avg_sparsity:
                print_runtime_sparsity(transformer)


if __name__ == "__main__":
    main()
