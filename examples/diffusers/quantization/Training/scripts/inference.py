#!/usr/bin/env python3
# ruff: noqa: T201
"""
CLI script for running LTX video/audio generation inference.

Usage:
    # Video-only
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --prompt "A cat playing with a ball" --output output.mp4

    # Video + Audio
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --prompt "A cat meowing loudly" --generate-audio \
        --output output.mp4 --audio-output output.wav
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ltx_core.model_loader import load_model
from ltxv_trainer.pipeline import LTXConditionPipeline
from ltxv_trainer.utils import save_video


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="LTX Video/Audio Generation",
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
        default=None,
        help="Path to text encoder (for Gemma/LTX-2 models)",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
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
        default=768,
        help="Video height (must be divisible by 32)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Video width (must be divisible by 32)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=97,
        help="Number of video frames",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=25,
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
        default=6.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Audio arguments
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate audio alongside video (audio duration will match video duration)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path (.mp4)",
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default=None,
        help="Output audio path (.wav, required if audio-duration is set)",
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
    generate_audio = args.generate_audio
    if generate_audio and args.audio_output is None:
        parser.error("--audio-output is required when --generate-audio is specified")

    print("=" * 80)
    print("LTX Video/Audio Generation")
    print("=" * 80)
    print(f"\nLoading model from {args.checkpoint}...")

    components = load_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=torch.bfloat16,
        with_video_vae=True,
        with_audio_vae=generate_audio,
        with_vocoder=generate_audio,
        with_text_encoder=True,
        with_connector=True,
        text_encoder_path=args.text_encoder_path,
    )

    print(f"Model type: {components.model_type.value}")
    print("\nCreating pipeline...")

    pipeline = LTXConditionPipeline(
        scheduler=components.scheduler,
        vae=components.video_vae,
        text_encoder=components.text_encoder,
        transformer=components.transformer,
        emb_connector=components.connector,
        audio_vae=components.audio_vae if generate_audio else None,
    )

    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    print("\n" + "=" * 80)
    print("Generation Parameters")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.frame_rate} fps")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    if generate_audio:
        video_duration = args.num_frames / args.frame_rate
        print(f"Audio: Enabled (duration will match video: {video_duration:.2f}s)")
    print("=" * 80)

    print(f"\nGenerating {'video + audio' if generate_audio else 'video'}...")

    videos, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        generate_audio=generate_audio,
    )

    # Save video
    print(f"\nSaving video to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert PIL images to tensor for save_video
    video_frames = videos[0]  # First batch item
    # Convert PIL images to numpy array
    frames_np = np.stack([np.array(frame) for frame in video_frames])
    # Convert to torch tensor [F, H, W, C] -> [F, C, H, W]
    video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

    save_video(video_tensor, output_path, fps=args.frame_rate)
    print(f"✓ Video saved: {len(video_frames)} frames @ {args.frame_rate} fps")

    # Save audio if generated
    if generate_audio and audio is not None:
        print(f"\nSaving audio to {args.audio_output}...")
        audio_output_path = Path(args.audio_output)
        audio_output_path.parent.mkdir(parents=True, exist_ok=True)

        # audio shape: [batch, channels, samples]
        audio_sample_rate = components.audio_vae.output_sample_rate
        torchaudio.save(
            str(audio_output_path),
            audio[0].cpu(),  # First batch item
            sample_rate=audio_sample_rate,
        )
        duration = audio.shape[2] / audio_sample_rate
        print(f"✓ Audio saved: {duration:.2f}s at {audio_sample_rate}Hz, {audio.shape[1]} channels")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
