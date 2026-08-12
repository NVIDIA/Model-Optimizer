# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generate an image with a trained Qwen-Image PDD transformer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

_THIS_DIR = Path(__file__).resolve().parent
_FASTGEN_DIR = _THIS_DIR.parent
_REPO_ROOT = _FASTGEN_DIR.parents[2]
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modelopt.torch.fastgen import PDDConfig, PDDPipeline  # noqa: E402
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (  # noqa: E402
    QwenImagePDDAdapter,
    enable_qwen_image_pdd_forward,
    restore_qwen_image_pdd_projection,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/diffusers/fastgen/pdd/configs/qwen_image.yaml"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Prepared full Diffusers pipeline created by prepare_qwen_image.py.",
    )
    parser.add_argument(
        "--transformer-dir",
        type=Path,
        help="Trained Diffusers transformer; defaults to MODEL_DIR/transformer.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--blocks",
        default="32,32,32,32",
        help="Comma-separated PDD block sizes; values must sum to grid_size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_blocks(value: str) -> list[int]:
    try:
        blocks = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--blocks must be a comma-separated list of integers.") from error
    if not blocks or any(block <= 0 for block in blocks):
        raise ValueError("--blocks must contain positive integers.")
    return blocks


def _load_config(path: Path, blocks: list[int]) -> PDDConfig:
    raw = yaml.safe_load(path.read_text())
    values = dict(raw["pdd"])
    values.update(inference_blocks=blocks, student_sample_steps=len(blocks))
    return PDDConfig.model_validate(values)


def _latent_shape(pipe: QwenImagePipeline, height: int, width: int) -> tuple[int, ...]:
    quantum = int(pipe.vae_scale_factor) * 2
    if height <= 0 or width <= 0 or height % quantum or width % quantum:
        raise ValueError(f"height and width must be positive multiples of {quantum}.")
    in_channels = int(pipe.transformer.config.in_channels)
    if in_channels <= 0 or in_channels % 4:
        raise ValueError("Qwen transformer in_channels must be a positive multiple of four.")
    return (1, in_channels // 4, 2 * (height // quantum), 2 * (width // quantum))


def _decode(pipe: QwenImagePipeline, latents: torch.Tensor):
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    decoded_input = latents.unsqueeze(2) * std.view(1, -1, 1, 1, 1)
    decoded_input = decoded_input + mean.view(1, -1, 1, 1, 1)
    decoded = pipe.vae.decode(decoded_input, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded[:, :, 0], output_type="pil")


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    blocks = _parse_blocks(args.blocks)
    config = _load_config(args.config, blocks)
    device = torch.device(args.device)
    transformer_dir = args.transformer_dir or args.model_dir / "transformer"

    transformer = QwenImageTransformer2DModel.from_pretrained(
        transformer_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    restore_qwen_image_pdd_projection(transformer, config)
    enable_qwen_image_pdd_forward(transformer)
    transformer.eval()

    pipe = QwenImagePipeline.from_pretrained(
        args.model_dir,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(device)
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    if prompt_mask is None:
        prompt_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.long)
    condition = (
        prompt_embeds.to(device=device, dtype=torch.bfloat16),
        prompt_mask.to(device=device, dtype=torch.long),
    )
    noise = torch.randn(
        _latent_shape(pipe, args.height, args.width),
        generator=torch.Generator(device=device).manual_seed(args.seed),
        device=device,
        dtype=torch.float32,
    )
    sampler = PDDPipeline(
        transformer,
        None,
        config,
        QwenImagePDDAdapter(config, compute_dtype=torch.bfloat16),
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    latents = sampler.sample(noise, condition=condition, blocks=blocks)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    images = _decode(pipe, latents.to(torch.bfloat16))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output)
    print(f"saved {args.output} with {len(blocks)} transformer calls in {latency:.3f}s")


if __name__ == "__main__":
    main()
