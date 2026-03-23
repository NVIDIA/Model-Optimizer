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

"""LTX-2 inference with skip-softmax sparse attention for diffusion models.

This example applies skip-softmax sparse attention to the LTX-2 video
generation model using the diffusion-specific percentile calibration method.
Unlike the LLM exponential model, this uses gap/log(seq_k) normalization
which is sequence-length-invariant (~2% drift across 3x resolution change).

Only the stage-1 backbone is sparsified.  Stage 2 (spatial upsampler +
distilled LoRA) runs unmodified.

Usage::

    # With calibration (recommended — generates a short video to calibrate)
    python ltx2_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \
        --calibrate --target-sparsity 0.2

    # Disable on first/last 2 layers (higher quality, less speedup)
    python ltx2_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \
        --calibrate --target-sparsity 0.2 --skip-first-last 2
"""

import argparse
import functools

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
from ltx_core.quantization.policy import QuantizationPolicy
from ltx_pipelines.utils.media_io import encode_video

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

# ---- Model paths (edit these or override via args) ----
CHECKPOINT_PATH = "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-dev.safetensors"
DISTILLED_LORA_PATH = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-distilled-lora-384.safetensors"
)
SPATIAL_UPSAMPLER_PATH = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
)
GEMMA_ROOT = "/home/scratch.omniml_data_2/jingyux/models/LTX-2/gemma-3-12b-it-qat-q4_0-unquantized"

DEFAULT_NUM_FRAMES = 121


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LTX-2 video generation with skip-softmax sparse attention (diffusion)"
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
        help="Directory to save videos when using --prompt-dir (one .mp4 per prompt file)",
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
        default=2,
        help="Number of first/last transformer layers to exclude from sparsity",
    )

    # Calibration options
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate threshold via percentile method (recommended)",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.2,
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
        help="Number of prompts to use for calibration (from OpenVid-1M dataset)",
    )
    return parser.parse_args()


def _patch_vae_requires_grad(pipeline: TI2VidTwoStagesPipeline):
    """Ensure VAE decoder weights have requires_grad=False.

    The pipeline runs under @torch.inference_mode(), but the VAE's conv3d
    tries to save inputs for backward when weights have requires_grad=True,
    causing "Inference tensors cannot be saved for backward".  We wrap the
    lazy decoder loader to disable gradients as soon as it's loaded.
    """
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

    Uses flash_skip_softmax_diffusion which requires percentile calibration.
    """
    attn_cfg = {
        "method": "flash_skip_softmax_diffusion",
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
        "*.attn2": {"enable": False},  # Text cross-attention
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    # Keep first/last N layers dense
    for i in range(args.skip_first_last):
        sparse_cfg[f"*transformer_blocks.{i}.attn*"] = {"enable": False}
        sparse_cfg[f"*transformer_blocks.{47 - i}.attn*"] = {"enable": False}

    config: dict = {"sparse_cfg": sparse_cfg}

    # Percentile calibration (required for diffusion method)
    if args.calibrate:
        sparse_cfg["calibration"] = {
            "target_sparse_ratio": {"prefill": args.target_sparsity},
        }

    return config


def load_calib_prompts(calib_size: int) -> list[str]:
    """Load calibration prompts from OpenVid-1M dataset.

    Args:
        calib_size: Number of prompts to load.

    Returns:
        List of prompt strings.
    """
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
    """Build a forward loop for percentile calibration.

    Runs inference on prompts from OpenVid-1M to collect normalized gap
    statistics across all attention layers and timesteps.

    Args:
        pipeline: LTX-2 pipeline.
        num_steps: Denoising steps per calibration sample.
        num_frames: Frames per calibration sample (shorter = faster).
        calib_size: Number of prompts to calibrate on.
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
        stats = module.get_stats()
        sparsity = stats.get("average_sparsity", "N/A")
        print(f"  {name}: sparsity={sparsity}")


def main() -> None:
    args = parse_args()

    # ---- Build pipeline ----
    print("Building LTX-2 pipeline...")
    pipeline = build_pipeline()

    # ---- Get and sparsify the stage-1 transformer ----
    transformer = pipeline.stage_1_model_ledger.transformer()
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
    import os

    prompts_and_outputs: list[tuple[str, str]] = []
    if args.prompt_dir:
        os.makedirs(args.output_dir or "output_videos", exist_ok=True)
        output_dir = args.output_dir or "output_videos"
        prompt_files = sorted(
            f for f in os.listdir(args.prompt_dir) if f.endswith(".txt")
        )
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
