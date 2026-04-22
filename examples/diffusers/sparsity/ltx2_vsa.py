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

"""LTX-2 inference with Video Sparse Attention (VSA).

Applies VSA to LTX-2's self-attention modules.  VSA is calibration-free —
sparsity is controlled via ``top_k_ratio`` (fraction of 3D blocks kept in
the sparse branch).

The LTX-2 plugin under ``modelopt.torch.sparsity.attention_sparsity.plugins.ltx2``
handles the specifics:

- Detects ``LTXSelfAttention`` modules by class name.
- Computes ``(T, H, W)`` from ``Modality.positions`` at each forward.
- Wraps each attention module in ``_LTX2SparseAttention``, which computes
  Q/K/V, RoPE, and the optional (zero-initialised, trainable)
  ``gate_compress`` before calling ``VSA.forward_attention``.

Requirements:
- ``fastvideo_kernel`` (Triton VSA kernel).
- ``ltx_core``, ``ltx_trainer``, ``ltx_pipelines`` (third-party LTX-2 packages
  from Lightricks — see the LICENSE notice in the top-level sparsity README).

Example::

    # VSA at 50% top-K ratio
    python ltx2_vsa.py --checkpoint path/to/model.safetensors \\
        --text-encoder-path path/to/gemma --top-k-ratio 0.5 \\
        --prompt "A cat playing piano" --output vsa.mp4

    # Baseline (no VSA)
    python ltx2_vsa.py --checkpoint path/to/model.safetensors \\
        --text-encoder-path path/to/gemma --no-vsa --output baseline.mp4
"""

import argparse
import copy
import time
from pathlib import Path

import torch

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import VSA_DEFAULT

# LTX-2 is optional; import lazily so --help works even without it.
try:
    from ltx_trainer.model_loader import load_model
    from ltx_trainer.progress import StandaloneSamplingProgress
    from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler
    from ltx_trainer.video_utils import save_video

    _LTX_AVAILABLE = True
except ImportError as _exc:
    _LTX_IMPORT_ERROR = _exc
    _LTX_AVAILABLE = False


# LTX-2 uses a 1:8192 pixels-to-tokens compression ratio
LTX2_PIXEL_TO_TOKEN_RATIO = 8192

# VSA 3D block size: 4x4x4 = 64 tokens per block
VSA_BLOCK_ELEMENTS = 64


def calculate_expected_tokens(num_frames: int, height: int, width: int) -> int:
    return num_frames * height * width // LTX2_PIXEL_TO_TOKEN_RATIO


def is_vsa_compatible(num_frames: int, height: int, width: int) -> tuple[bool, str]:
    """Check whether the requested input size is large enough for VSA to help."""
    tokens = calculate_expected_tokens(num_frames, height, width)
    tiles = tokens // VSA_BLOCK_ELEMENTS
    if tiles >= 90:
        return True, f"Excellent: {tokens} tokens ({tiles} tiles)"
    if tiles >= 16:
        return True, f"Marginal: {tokens} tokens ({tiles} tiles)"
    return False, f"Too small: {tokens} tokens ({tiles} tiles, need 16+ for VSA)"


def apply_vsa(
    transformer: torch.nn.Module,
    num_frames: int,
    height: int,
    width: int,
    top_k_ratio: float,
) -> torch.nn.Module:
    """Apply VSA to the LTX-2 transformer."""
    compatible, reason = is_vsa_compatible(num_frames, height, width)
    print(f"  VSA compatibility: {reason}")
    if not compatible:
        print("  [WARNING] Input size may be too small for VSA to help.")

    config = copy.deepcopy(VSA_DEFAULT)
    # Override top_k_ratio on the attention pattern
    for cfg in config["sparse_cfg"].values():
        if isinstance(cfg, dict) and cfg.get("method") == "vsa":
            cfg["top_k_ratio"] = top_k_ratio

    print("  Applying VSA to attention modules...")
    return mtsa.sparsify(transformer, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LTX-2 video generation with Video Sparse Attention (VSA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="LTX-2 model checkpoint")
    parser.add_argument(
        "--text-encoder-path", type=str, required=True, help="Gemma text encoder directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene mountain landscape with a flowing river, golden hour lighting",
    )
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=512, help="Video height (multiple of 32)")
    parser.add_argument("--width", type=int, default=768, help="Video width (multiple of 32)")
    parser.add_argument("--num-frames", type=int, default=121, help="Must be k*8 + 1")
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no-vsa",
        action="store_true",
        help="Disable VSA (baseline run, for timing comparison)",
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.5,
        help="VSA sparsity ratio (0.5 ⇒ 50%% sparsity, 0.3 ⇒ 70%%)",
    )

    parser.add_argument("--skip-audio", action="store_true", help="Skip audio generation")
    parser.add_argument("--output", type=str, default="output_vsa.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def run_generation(
    sampler,
    config,
    device: str,
    num_inference_steps: int,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    if label:
        print(f"\n{label}")
    print(f"Generating video ({num_inference_steps} steps)...")
    t0 = time.time()
    with StandaloneSamplingProgress(num_steps=num_inference_steps) as progress:
        sampler.sampling_context = progress
        video, audio = sampler.generate(config=config, device=device)
    elapsed = time.time() - t0
    print(f"Generation completed in {elapsed:.2f}s")
    return video, audio, elapsed


def main() -> None:
    if not _LTX_AVAILABLE:
        raise ImportError(
            "LTX-2 packages are required for this example. Install with: "
            "pip install ltx-core ltx-trainer ltx-pipelines. "
            f"(original error: {_LTX_IMPORT_ERROR})"
        )

    args = parse_args()
    generate_audio = not args.skip_audio

    print("=" * 72)
    print("LTX-2 + VSA")
    print("=" * 72)

    tokens = calculate_expected_tokens(args.num_frames, args.height, args.width)
    tiles = tokens // VSA_BLOCK_ELEMENTS
    _, reason = is_vsa_compatible(args.num_frames, args.height, args.width)
    print("\nInput Configuration:")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames:     {args.num_frames} @ {args.frame_rate} fps")
    print(f"  Tokens:     {tokens} ({tiles} tiles)")
    print(f"  VSA:        {reason}")

    print("\nLoading LTX-2 model components...")
    components = load_model(
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
    print("Model loaded")

    transformer = components.transformer

    if not args.no_vsa:
        transformer = apply_vsa(
            transformer,
            args.num_frames,
            args.height,
            args.width,
            top_k_ratio=args.top_k_ratio,
        )
        components.transformer = transformer

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

    sampler = ValidationSampler(
        transformer=components.transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=components.text_encoder,
        audio_decoder=components.audio_vae_decoder if generate_audio else None,
        vocoder=components.vocoder if generate_audio else None,
    )

    label = "BASELINE (no VSA)" if args.no_vsa else f"WITH VSA (top_k_ratio={args.top_k_ratio})"
    video, audio, elapsed = run_generation(
        sampler, gen_config, args.device, args.num_inference_steps, label=label
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_sample_rate = None
    if audio is not None and components.vocoder is not None:
        audio_sample_rate = components.vocoder.output_sample_rate
    save_video(
        video_tensor=video,
        output_path=out_path,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
    )
    print(f"Saved: {args.output}")

    print("\n" + "=" * 72)
    print(f"Done in {elapsed:.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
