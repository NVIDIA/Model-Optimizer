#!/usr/bin/env python3
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

"""
Test VSA integration with LTX-2 video generation.

This script tests Video Sparse Attention (VSA) on the full LTX-2 pipeline,
measuring performance improvements and validating output quality.

Usage:
    # Test with VSA enabled (default)
    python test_ltx2_vsa_integration.py \
        --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat playing with a ball"

    # Test without VSA (baseline)
    python test_ltx2_vsa_integration.py \
        --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat playing with a ball" \
        --no-vsa

    # Compare both (recommended)
    python test_ltx2_vsa_integration.py \
        --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat playing with a ball" \
        --compare

    # Custom VSA parameters
    python test_ltx2_vsa_integration.py \
        --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat playing with a ball" \
        --top-k-ratio 0.5 \
        --num-frames 121 --height 512 --width 768

VSA improves attention performance by using 3D tile-based sparsity:
- Automatically adapts to LTX-2's compressed token sequence
"""

import argparse
import copy
import time
from pathlib import Path

import torch
from ltx_trainer.model_loader import load_model
from ltx_trainer.progress import StandaloneSamplingProgress
from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import save_video

from modelopt.torch.sparsity.attention_sparsity import sparsify
from modelopt.torch.sparsity.attention_sparsity.config import VSA_DEFAULT


def calculate_expected_tokens(num_frames: int, height: int, width: int) -> int:
    """Calculate expected token count for LTX-2.

    LTX-2 uses 1:8192 pixels-to-tokens compression ratio.
    """
    pixels = num_frames * height * width
    tokens = pixels // 8192
    return tokens


def is_vsa_compatible(num_frames: int, height: int, width: int) -> tuple[bool, str]:
    """Check if input size is compatible with VSA.

    Args:
        num_frames: Number of video frames.
        height: Video height in pixels.
        width: Video width in pixels.

    Returns:
        Tuple of (is_compatible, reason_message).
    """
    tokens = calculate_expected_tokens(num_frames, height, width)
    tiles = tokens // 64  # VSA tile size: 4x4x4 = 64

    if tiles >= 90:
        return True, f"Excellent: {tokens} tokens ({tiles} tiles)"
    elif tiles >= 16:
        return True, f"Marginal: {tokens} tokens ({tiles} tiles)"
    else:
        return False, f"Too small: {tokens} tokens ({tiles} tiles, need 16+ for VSA)"


def apply_vsa_to_transformer(
    transformer: torch.nn.Module,
    num_frames: int,
    height: int,
    width: int,
    top_k_ratio: float = 0.5,
) -> torch.nn.Module:
    """Apply VSA to the LTX-2 transformer.

    Args:
        transformer: The transformer model.
        num_frames: Number of frames (for compatibility checking).
        height: Video height (for compatibility checking).
        width: Video width (for compatibility checking).
        top_k_ratio: Sparsity ratio (0.5 = 50% sparsity).

    Returns:
        Modified transformer with VSA enabled.
    """
    print("\nConfiguring VSA for LTX-2...")

    # Check compatibility
    tokens = calculate_expected_tokens(num_frames, height, width)
    tiles = tokens // 64
    compatible, reason = is_vsa_compatible(num_frames, height, width)

    print(f"  Expected sequence: {tokens} tokens ({tiles} tiles)")
    print(f"  VSA compatibility: {reason}")

    if not compatible:
        print("  [WARNING] Input size may be too small for VSA to provide significant benefit.")
        print("     Consider using larger inputs (121+ frames @ 512x768+) for best results.")

    # Configure VSA using the standard preset, overriding top_k_ratio if needed
    sparse_config = copy.deepcopy(VSA_DEFAULT)
    # Find the attn pattern key and override top_k_ratio
    for cfg in sparse_config["sparse_cfg"].values():
        if isinstance(cfg, dict) and cfg.get("method") == "vsa":
            cfg["top_k_ratio"] = top_k_ratio

    # Apply VSA to transformer
    print("  Applying VSA to attention modules...")
    transformer = sparsify(transformer, sparse_config)

    return transformer


def run_generation(
    sampler: ValidationSampler,
    config: GenerationConfig,
    device: str,
    num_inference_steps: int,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """Run video generation and return timing information.

    Args:
        sampler: ValidationSampler instance.
        config: Generation configuration.
        device: Device to run on.
        num_inference_steps: Number of denoising steps.
        label: Label for logging (e.g., "BASELINE", "WITH VSA").

    Returns:
        Tuple of (video, audio, elapsed_time)
    """
    if label:
        print(f"\n{label}")

    print(f"Generating video ({num_inference_steps} steps)...")
    start_time = time.time()

    with StandaloneSamplingProgress(num_steps=num_inference_steps) as progress:
        sampler.sampling_context = progress
        video, audio = sampler.generate(config=config, device=device)

    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f}s")

    return video, audio, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test VSA integration with LTX-2 video generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        required=True,
        help="Path to Gemma text encoder directory",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene mountain landscape with a flowing river, golden hour lighting",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (must be divisible by 32)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width (must be divisible by 32)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=121,
        help="Number of video frames (must be k*8 + 1)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Video frame rate",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (CFG)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # VSA arguments
    parser.add_argument(
        "--no-vsa",
        action="store_true",
        help="Disable VSA (for baseline comparison)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both with and without VSA for comparison",
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.5,
        help="VSA sparsity ratio (0.5 = 50%% sparsity)",
    )

    # Audio arguments
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio generation (faster testing)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="output_vsa.mp4",
        help="Output video path (.mp4)",
    )
    parser.add_argument(
        "--output-baseline",
        type=str,
        default="output_baseline.mp4",
        help="Baseline output path (used with --compare)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Validate arguments
    generate_audio = not args.skip_audio

    print("=" * 80)
    print("LTX-2 + VSA Integration Test")
    print("=" * 80)

    # Check VSA compatibility
    tokens = calculate_expected_tokens(args.num_frames, args.height, args.width)
    tiles = tokens // 64
    compatible, reason = is_vsa_compatible(args.num_frames, args.height, args.width)

    print("\nInput Configuration:")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames} @ {args.frame_rate} fps")
    print(f"  Expected tokens: {tokens} ({tiles} tiles)")
    print(f"  VSA compatibility: {reason}")

    if not compatible and not args.no_vsa and not args.compare:
        print("\n[WARNING] WARNING: Input size may be too small for VSA benefit")
        print("  Recommended: 121+ frames @ 512x768+ for optimal VSA performance")
        print("  Use --no-vsa to disable VSA for small inputs")

    # Load model components
    print("\nLoading LTX-2 model components...")
    components = load_model(
        checkpoint_path=args.checkpoint,
        device="cpu",  # Load to CPU first
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=generate_audio,
        with_vocoder=generate_audio,
        with_text_encoder=True,
        text_encoder_path=args.text_encoder_path,
    )
    print("Model components loaded")

    # Create generation config
    gen_config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        condition_image=None,
        reference_video=None,
        generate_audio=generate_audio,
        include_reference_in_output=False,
    )

    print("\n" + "=" * 80)
    print("Generation Parameters")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.frame_rate} fps")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"CFG scale: {args.guidance_scale}")
    print(f"Seed: {args.seed}")
    if generate_audio:
        video_duration = args.num_frames / args.frame_rate
        print(f"Audio: Enabled (duration: {video_duration:.2f}s)")
    else:
        print("Audio: Disabled (skip-audio mode)")
    print("=" * 80)

    # Test scenarios
    results = {}

    if args.compare:
        # ======================================================================
        # Run BASELINE (no VSA)
        # ======================================================================
        print("\n" + "=" * 80)
        print("TEST 1/2: BASELINE (no VSA)")
        print("=" * 80)

        # Create sampler without VSA
        sampler_baseline = ValidationSampler(
            transformer=components.transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=components.text_encoder,
            audio_decoder=components.audio_vae_decoder if generate_audio else None,
            vocoder=components.vocoder if generate_audio else None,
        )

        try:
            video_baseline, audio_baseline, time_baseline = run_generation(
                sampler_baseline,
                gen_config,
                args.device,
                args.num_inference_steps,
            )
            results["baseline"] = time_baseline

            # Save baseline video
            output_baseline_path = Path(args.output_baseline)
            output_baseline_path.parent.mkdir(parents=True, exist_ok=True)

            audio_sample_rate = None
            if audio_baseline is not None and components.vocoder is not None:
                audio_sample_rate = components.vocoder.output_sample_rate

            save_video(
                video_tensor=video_baseline,
                output_path=output_baseline_path,
                fps=args.frame_rate,
                audio=audio_baseline,
                audio_sample_rate=audio_sample_rate,
            )
            print(f"Baseline video saved: {args.output_baseline}")
        except Exception as e:
            print(f"Baseline generation failed: {e}")
            import traceback

            traceback.print_exc()
            return

        # ======================================================================
        # Run WITH VSA
        # ======================================================================
        print("\n" + "=" * 80)
        print("TEST 2/2: WITH VSA")
        print("=" * 80)

        # Reload transformer for VSA test
        print("\nReloading transformer for VSA test...")
        components_vsa = load_model(
            checkpoint_path=args.checkpoint,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=False,
            with_video_vae_decoder=True,
            with_audio_vae_decoder=generate_audio,
            with_vocoder=generate_audio,
            with_text_encoder=True,
            text_encoder_path=args.text_encoder_path,
        )

        # Apply VSA
        components_vsa.transformer = apply_vsa_to_transformer(
            components_vsa.transformer,
            args.num_frames,
            args.height,
            args.width,
            top_k_ratio=args.top_k_ratio,
        )

        # Create sampler with VSA
        sampler_vsa = ValidationSampler(
            transformer=components_vsa.transformer,
            vae_decoder=components_vsa.video_vae_decoder,
            vae_encoder=components_vsa.video_vae_encoder,
            text_encoder=components_vsa.text_encoder,
            audio_decoder=components_vsa.audio_vae_decoder if generate_audio else None,
            vocoder=components_vsa.vocoder if generate_audio else None,
        )

        try:
            video_vsa, audio_vsa, time_vsa = run_generation(
                sampler_vsa,
                gen_config,
                args.device,
                args.num_inference_steps,
            )
            results["vsa"] = time_vsa

            # Save VSA video
            output_vsa_path = Path(args.output)
            output_vsa_path.parent.mkdir(parents=True, exist_ok=True)

            audio_sample_rate = None
            if audio_vsa is not None and components_vsa.vocoder is not None:
                audio_sample_rate = components_vsa.vocoder.output_sample_rate

            save_video(
                video_tensor=video_vsa,
                output_path=output_vsa_path,
                fps=args.frame_rate,
                audio=audio_vsa,
                audio_sample_rate=audio_sample_rate,
            )
            print(f"VSA video saved: {args.output}")
        except Exception as e:
            print(f"VSA generation failed: {e}")
            import traceback

            traceback.print_exc()
            return

    else:
        # ======================================================================
        # Single test (with or without VSA)
        # ======================================================================
        print("\n" + "=" * 80)
        print(f"TEST: {'WITH VSA' if not args.no_vsa else 'WITHOUT VSA'}")
        print("=" * 80)

        transformer = components.transformer

        # Apply VSA if enabled
        if not args.no_vsa:
            transformer = apply_vsa_to_transformer(
                transformer,
                args.num_frames,
                args.height,
                args.width,
                top_k_ratio=args.top_k_ratio,
            )

        # Create sampler
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=components.text_encoder,
            audio_decoder=components.audio_vae_decoder if generate_audio else None,
            vocoder=components.vocoder if generate_audio else None,
        )

        try:
            video, audio, elapsed = run_generation(
                sampler,
                gen_config,
                args.device,
                args.num_inference_steps,
            )
            results["single"] = elapsed

            # Save video
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            audio_sample_rate = None
            if audio is not None and components.vocoder is not None:
                audio_sample_rate = components.vocoder.output_sample_rate

            save_video(
                video_tensor=video,
                output_path=output_path,
                fps=args.frame_rate,
                audio=audio,
                audio_sample_rate=audio_sample_rate,
            )
            print(f"Video saved: {args.output}")
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback

            traceback.print_exc()
            return

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    if args.compare:
        speedup = results["baseline"] / results["vsa"]
        print("\nPerformance Comparison:")
        print(f"  Baseline (no VSA):  {results['baseline']:.2f}s")
        print(f"  With VSA:           {results['vsa']:.2f}s")
        print(f"  Speedup:            {speedup:.2f}x")
        print()
        print(f"  Baseline video: {args.output_baseline}")
        print(f"  VSA video:      {args.output}")
        print()
    else:
        print(f"\nGeneration time: {results['single']:.2f}s")
        print(f"Output: {args.output}")

    print("\nVSA integration test successful!")
    print("=" * 80)


if __name__ == "__main__":
    main()
