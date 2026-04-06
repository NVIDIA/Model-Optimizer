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
generation model (text-to-video) using exponential model calibration
(``scale_factor = a * exp(b * target_sparsity)``).

During calibration, ``flash_skip_softmax`` with the eager attention backend
collects sparsity statistics across multiple threshold trials. The fitted
exponential model then allows runtime control of the target sparsity ratio
without recalibration.

The Wan 2.2 5B model has 40 transformer blocks with self-attention (attn1)
and cross-attention (attn2). Only self-attention is sparsified.

Usage::

    # With calibration (recommended)
    python wan22_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \\
        --calibrate --target-sparsity 0.25

    # Custom model path
    python wan22_skip_softmax.py --model-path /path/to/Wan2.2-T2V-5B \\
        --prompt "A sunset over mountains" --output sunset.mp4 --calibrate
"""

import argparse
import os

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

DEFAULT_MODEL_PATH = os.environ.get("WAN22_MODEL_PATH", "Wan-AI/Wan2.2-T2V-5B")
NUM_TRANSFORMER_BLOCKS = 40

# Default threshold trials for calibration
DEFAULT_THRESHOLD_TRIALS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan 2.2 video generation with skip-softmax sparse attention"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Wan 2.2 model path or HF ID"
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of frames (must be 4k+1)"
    )
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument(
        "--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sparse attention options
    parser.add_argument(
        "--skip-first-last",
        type=int,
        default=2,
        help="Number of first/last transformer layers to keep dense (default: 2)",
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
        default=0.25,
        help="Target sparsity ratio for calibration (0.0-1.0)",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=10,
        help="Inference steps for calibration",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=33,
        help="Number of frames for calibration (fewer = faster)",
    )
    return parser.parse_args()


def build_pipeline(model_path: str) -> WanPipeline:
    """Build the Wan 2.2 text-to-video pipeline."""
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def build_sparse_config(args: argparse.Namespace) -> dict:
    """Build sparse attention config from CLI args.

    Uses flash_skip_softmax which supports both calibration (eager attention
    with F.softmax patching) and inference. Calibration fits an exponential
    model: scale_factor = a * exp(b * sparsity).
    """
    attn_cfg: dict = {
        "method": "flash_skip_softmax",
        "thresholds": {"prefill": [1e-3]},
        "br": 128,
        "bc": 128,
        "backend": "pytorch",
        "is_causal": False,  # Diffusion = bidirectional attention
        "collect_stats": True,
        "enable": True,
    }

    sparse_cfg: dict = {
        "*.attn1*": attn_cfg,  # Self-attention only
        "*.attn2*": {"enable": False},  # Text cross-attention
        "default": {"enable": False},
    }

    # Keep first/last N layers dense for quality
    for i in range(args.skip_first_last):
        sparse_cfg[f"*blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*blocks.{NUM_TRANSFORMER_BLOCKS - 1 - i}.attn*"] = {"enable": False}

    config: dict = {"sparse_cfg": sparse_cfg}

    # Add calibration config with threshold trials
    if args.calibrate:
        sparse_cfg["calibration"] = {
            "target_sparse_ratio": {"prefill": args.target_sparsity},
            "samples": 1,
            "threshold_trials": DEFAULT_THRESHOLD_TRIALS,
        }

    return config


def build_calibration_forward_loop(
    pipe: WanPipeline,
    prompt: str,
    num_steps: int = 10,
    num_frames: int = 33,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
):
    """Build a forward loop for exponential model calibration."""

    def forward_loop(model):
        print(f"Calibration: generating {num_frames} frames @ {height}x{width}...")
        pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )

    return forward_loop


def print_sparsity_summary(model: torch.nn.Module) -> None:
    """Print per-module sparsity statistics."""
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


def main() -> None:
    args = parse_args()

    # ---- Build pipeline ----
    print(f"Loading Wan 2.2 from {args.model_path}...")
    pipe = build_pipeline(args.model_path)

    # ---- Get and sparsify the transformer ----
    transformer = pipe.transformer

    config = build_sparse_config(args)
    forward_loop = None
    if args.calibrate:
        forward_loop = build_calibration_forward_loop(
            pipe,
            prompt=args.prompt,
            num_steps=args.calib_steps,
            num_frames=args.calib_frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )

    print("Applying skip-softmax sparse attention...")
    mtsa.sparsify(transformer, config, forward_loop=forward_loop)

    # ---- Generate ----
    print(f"Generating: {args.prompt[:80]}...")
    output = pipe(
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )

    export_to_video(output.frames[0], args.output, fps=16)
    print(f"Saved to {args.output}")

    # ---- Print stats ----
    print_sparsity_summary(transformer)


if __name__ == "__main__":
    main()
