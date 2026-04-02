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

"""LTX-2 inference with skip-softmax sparse attention.

This example applies skip-softmax sparse attention to the LTX-2 video
generation model using exponential model calibration
(``scale_factor = a * exp(b * target_sparsity)``).

During calibration, ``flash_skip_softmax`` with the eager attention backend
collects sparsity statistics across multiple threshold trials. The fitted
exponential model then allows runtime control of the target sparsity ratio
without recalibration.

Only the stage-1 backbone is sparsified.  Stage 2 (spatial upsampler +
distilled LoRA) runs unmodified.

Usage::

    # With calibration (recommended)
    python ltx2_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \\
        --calibrate --target-sparsity 0.25

    # Disable sparsity on first/last 2 layers (higher quality, less speedup)
    python ltx2_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \\
        --calibrate --target-sparsity 0.25 --skip-first-last 2
"""

import argparse
import functools
import os

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_2_STAGE_HEIGHT,
    DEFAULT_2_STAGE_WIDTH,
    DEFAULT_AUDIO_GUIDER_PARAMS,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_VIDEO_GUIDER_PARAMS,
)
from ltx_pipelines.utils.media_io import encode_video

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

# ---- Model paths (edit these or override via environment variables) ----
CHECKPOINT_PATH = os.environ.get(
    "LTX2_CHECKPOINT",
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-dev.safetensors",
)
DISTILLED_LORA_PATH = os.environ.get(
    "LTX2_DISTILLED_LORA",
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-distilled-lora-384.safetensors",
)
SPATIAL_UPSAMPLER_PATH = os.environ.get(
    "LTX2_SPATIAL_UPSAMPLER",
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
)
GEMMA_ROOT = os.environ.get(
    "LTX2_GEMMA_ROOT",
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/gemma-3-12b-it-qat-q4_0-unquantized",
)

DEFAULT_NUM_FRAMES = 121
NUM_TRANSFORMER_BLOCKS = 48

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
        description="LTX-2 video generation with skip-softmax sparse attention"
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=None,
        help="Directory of .txt prompt files (one prompt per file). Overrides --prompt.",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save videos when using --prompt-dir",
    )
    parser.add_argument(
        "--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of frames"
    )
    parser.add_argument("--height", type=int, default=DEFAULT_2_STAGE_HEIGHT, help="Video height")
    parser.add_argument("--width", type=int, default=DEFAULT_2_STAGE_WIDTH, help="Video width")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

    # Sparse attention options
    parser.add_argument(
        "--skip-first-last",
        type=int,
        default=0,
        help="Number of first/last transformer layers to keep dense (default: 0)",
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
        help="Inference steps per calibration sample",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=81,
        help="Number of frames per calibration sample",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=1,
        help="Number of prompts to use for calibration",
    )
    return parser.parse_args()


def _patch_vae_requires_grad(pipeline: TI2VidTwoStagesPipeline):
    """Ensure VAE decoder weights have requires_grad=False to avoid autograd issues."""
    for ledger_attr in ("stage_1_model_ledger", "stage_2_model_ledger"):
        ledger = getattr(pipeline, ledger_attr, None)
        if ledger is None:
            continue
        for loader_name in ("video_decoder", "audio_decoder"):
            orig_loader = getattr(ledger, loader_name, None)
            if orig_loader is None:
                continue

            def _make_patched(fn):
                @functools.wraps(fn)
                def patched():
                    model = fn()
                    model.requires_grad_(False)
                    return model

                return patched

            setattr(ledger, loader_name, _make_patched(orig_loader))


def build_pipeline() -> TI2VidTwoStagesPipeline:
    """Build the LTX-2 two-stage video generation pipeline."""
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora=[
            LoraPathStrengthAndSDOps(DISTILLED_LORA_PATH, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)
        ],
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=[],
    )
    _patch_vae_requires_grad(pipeline)
    return pipeline


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
        "*.attn1": attn_cfg,  # Self-attention only
        # Disable on all cross-attention and cross-modal attention
        "*.attn2": {"enable": False},
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    # Keep first/last N layers dense for quality
    for i in range(args.skip_first_last):
        sparse_cfg[f"*transformer_blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*transformer_blocks.{NUM_TRANSFORMER_BLOCKS - 1 - i}.attn*"] = {
            "enable": False
        }

    config: dict = {"sparse_cfg": sparse_cfg}

    # Add calibration config with threshold trials
    if args.calibrate:
        sparse_cfg["calibration"] = {
            "target_sparse_ratio": {"prefill": args.target_sparsity},
            "samples": args.calib_size,
            "threshold_trials": DEFAULT_THRESHOLD_TRIALS,
        }

    return config


def load_calib_prompts(calib_size: int) -> list[str]:
    """Load calibration prompts from OpenVid-1M dataset."""
    from datasets import load_dataset

    dataset = load_dataset("nkp37/OpenVid-1M")
    prompts = list(dataset["train"]["caption"][:calib_size])
    print(f"Loaded {len(prompts)} calibration prompts from OpenVid-1M")
    return prompts


def build_calibration_forward_loop(
    pipeline: TI2VidTwoStagesPipeline,
    num_steps: int = 10,
    num_frames: int = 81,
    calib_size: int = 1,
):
    """Build a forward loop for exponential model calibration.

    Generates short videos to exercise the attention mechanism at various
    threshold trials, collecting sparsity statistics for the exponential fit.
    """
    calib_prompts = load_calib_prompts(calib_size)
    tiling_config = TilingConfig.default()

    def forward_loop(model):
        for i, prompt in enumerate(calib_prompts):
            print(f"Calibration [{i + 1}/{len(calib_prompts)}]: {prompt[:60]}...")
            pipeline(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=DEFAULT_SEED,
                height=DEFAULT_2_STAGE_HEIGHT,
                width=DEFAULT_2_STAGE_WIDTH,
                num_frames=num_frames,
                frame_rate=DEFAULT_FRAME_RATE,
                num_inference_steps=num_steps,
                video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
                audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
                images=[],
                tiling_config=tiling_config,
            )

    return forward_loop


def print_sparsity_summary(transformer: torch.nn.Module) -> None:
    """Print per-module sparsity statistics."""
    enabled, disabled = [], []
    for name, module in transformer.named_modules():
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
    print("Building LTX-2 pipeline...")
    pipeline = build_pipeline()

    # ---- Get and sparsify the stage-1 transformer ----
    transformer = pipeline.stage_1_model_ledger.transformer()
    # Pin transformer in memory so pipeline reuses the sparsified version
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    config = build_sparse_config(args)
    forward_loop = None
    if args.calibrate:
        forward_loop = build_calibration_forward_loop(
            pipeline,
            num_steps=args.calib_steps,
            num_frames=args.calib_frames,
            calib_size=args.calib_size,
        )

    print("Applying skip-softmax sparse attention...")
    mtsa.sparsify(transformer, config, forward_loop=forward_loop)

    # ---- Build prompt list ----
    prompts_and_outputs: list[tuple[str, str]] = []
    if args.prompt_dir:
        output_dir = args.output_dir or "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        prompt_files = sorted(f for f in os.listdir(args.prompt_dir) if f.endswith(".txt"))
        for pf in prompt_files:
            with open(os.path.join(args.prompt_dir, pf)) as f:
                prompt = f.read().strip()
            stem = os.path.splitext(pf)[0]
            prompts_and_outputs.append((prompt, os.path.join(output_dir, f"{stem}.mp4")))
    elif args.prompt:
        prompts_and_outputs.append((args.prompt, args.output))
    else:
        raise ValueError("Either --prompt or --prompt-dir must be provided")

    # ---- Generate ----
    tiling_config = TilingConfig.default()
    for i, (prompt, output_path) in enumerate(prompts_and_outputs):
        print(f"\nGenerating [{i + 1}/{len(prompts_and_outputs)}]: {prompt[:80]}...")

        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=DEFAULT_FRAME_RATE,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
            audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
            images=[],
            tiling_config=tiling_config,
        )

        encode_video(
            video=video,
            fps=DEFAULT_FRAME_RATE,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(args.num_frames, tiling_config),
        )
        print(f"Saved to {output_path}")

    # ---- Print stats ----
    print_sparsity_summary(transformer)


if __name__ == "__main__":
    main()
