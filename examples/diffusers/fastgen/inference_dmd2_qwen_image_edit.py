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

"""Few-step inference for a DMD2-trained Qwen-Image-Edit-2511 student.

The stock EditPlus pipeline is reused for multimodal prompt encoding, reference-image
preprocessing, VAE encode/decode, and output postprocessing. Only its denoising loop is
replaced with the exact rectified-flow schedule used by DMD2 training. Target tokens are
followed by the fixed reference-image tokens on every transformer call; only the target
prediction prefix is stepped.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

logger = logging.getLogger(__name__)

_CONDITION_IMAGE_AREA = 384 * 384
_VAE_IMAGE_AREA = 1024 * 1024


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Match Diffusers EditPlus' area-preserving, 32-pixel-quantized resize."""
    raw_width = math.sqrt(target_area * ratio)
    raw_height = raw_width / ratio
    return round(raw_width / 32) * 32, round(raw_height / 32) * 32


def _overlay_ema(student: torch.nn.Module, ema_path: str | os.PathLike[str]) -> None:
    payload = torch.load(str(ema_path), map_location="cpu", weights_only=True)
    shadow = payload.get("shadow", payload) if isinstance(payload, dict) else payload
    if not isinstance(shadow, dict):
        raise TypeError(f"EMA payload must be a state dict, got {type(shadow).__name__}.")
    missing, unexpected = student.load_state_dict(shadow, strict=False)
    if missing or unexpected:
        logger.warning("EMA overlay: %d missing, %d unexpected keys", len(missing), len(unexpected))


@dataclass
class QwenImageEditDMDOutput:
    images: list[Any]


class QwenImageEditDMDInferencePipeline:
    """DMD sampler around a stock :class:`QwenImageEditPlusPipeline`."""

    def __init__(self, pipeline: QwenImageEditPlusPipeline, max_t: float = 0.999) -> None:
        self._pipe = pipeline
        self.max_t = float(max_t)

    @classmethod
    def from_pretrained(
        cls,
        student_path: str | os.PathLike[str],
        base_pipeline_path: str | os.PathLike[str] = "Qwen/Qwen-Image-Edit-2511",
        *,
        ema_path: str | os.PathLike[str] | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_t: float = 0.999,
    ) -> QwenImageEditDMDInferencePipeline:
        student_path = str(student_path)
        if not os.path.isdir(student_path):
            raise FileNotFoundError(f"student_path is not a directory: {student_path}")
        student = QwenImageTransformer2DModel.from_pretrained(student_path, torch_dtype=torch_dtype)
        if ema_path is not None:
            _overlay_ema(student, ema_path)
        student.eval()
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(base_pipeline_path), transformer=student, torch_dtype=torch_dtype
        )
        return cls(pipeline, max_t=max_t)

    def to(self, device: str | torch.device) -> QwenImageEditDMDInferencePipeline:
        self._pipe.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._pipe.transformer.device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._pipe.transformer.parameters()).dtype

    @staticmethod
    def _resolve_schedule(
        num_inference_steps: int,
        max_t: float,
        t_list: list[float] | None,
    ) -> list[float]:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        if t_list is None:
            return torch.linspace(max_t, 0.0, num_inference_steps + 1).tolist()
        if len(t_list) != num_inference_steps + 1:
            raise ValueError("t_list must contain num_inference_steps + 1 entries.")
        schedule = [float(value) for value in t_list]
        if abs(schedule[-1]) > 1e-6:
            raise ValueError("t_list must end at 0.0.")
        if any(left <= right for left, right in itertools.pairwise(schedule)):
            raise ValueError("t_list must be strictly decreasing.")
        return schedule

    @torch.no_grad()
    def __call__(
        self,
        image: Any | list[Any],
        prompt: str,
        *,
        negative_prompt: str | None = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        generator: torch.Generator | None = None,
        max_t: float | None = None,
        t_list: list[float] | None = None,
        sample_type: str = "ode",
        output_type: str = "pil",
    ) -> QwenImageEditDMDOutput:
        """Edit one image from one or more ordered references.

        A CFG-trained DMD2 student has already internalized teacher guidance, so the
        default ``guidance_scale=1`` performs a single transformer call per step.
        """
        if sample_type not in {"ode", "sde"}:
            raise ValueError("sample_type must be 'ode' or 'sde'.")
        references = list(image) if isinstance(image, (list, tuple)) else [image]
        if not references:
            raise ValueError("At least one reference image is required.")

        pipe = self._pipe
        device, dtype = self.device, self.dtype
        max_t = self.max_t if max_t is None else float(max_t)
        schedule = self._resolve_schedule(num_inference_steps, max_t, t_list)

        # Match QwenImageEditPlusPipeline.__call__: the last reference determines the
        # default target aspect ratio; each reference gets separate vision/VAE resolutions.
        last_width, last_height = references[-1].size
        default_width, default_height = _calculate_dimensions(
            _VAE_IMAGE_AREA, last_width / last_height
        )
        width = int(width or default_width)
        height = int(height or default_height)
        multiple = pipe.vae_scale_factor * 2
        width, height = width // multiple * multiple, height // multiple * multiple

        condition_images: list[Any] = []
        vae_images: list[torch.Tensor] = []
        vae_sizes: list[tuple[int, int]] = []
        for reference in references:
            ref_width, ref_height = reference.size
            ratio = ref_width / ref_height
            cond_width, cond_height = _calculate_dimensions(_CONDITION_IMAGE_AREA, ratio)
            vae_width, vae_height = _calculate_dimensions(_VAE_IMAGE_AREA, ratio)
            condition_images.append(pipe.image_processor.resize(reference, cond_height, cond_width))
            vae_images.append(
                pipe.image_processor.preprocess(reference, vae_height, vae_width).unsqueeze(2)
            )
            vae_sizes.append((vae_width, vae_height))

        prompt_embeds, prompt_mask = pipe.encode_prompt(
            image=condition_images, prompt=prompt, device=device, num_images_per_prompt=1
        )
        do_cfg = guidance_scale != 1.0
        negative_embeds = negative_mask = None
        if do_cfg:
            negative_prompt = " " if negative_prompt is None else negative_prompt
            negative_embeds, negative_mask = pipe.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                device=device,
                num_images_per_prompt=1,
            )

        channels = pipe.transformer.config.in_channels // 4
        h_lat = 2 * (height // (pipe.vae_scale_factor * 2))
        w_lat = 2 * (width // (pipe.vae_scale_factor * 2))
        noise = randn_tensor(
            (1, 1, channels, h_lat, w_lat), generator=generator, device=device, dtype=dtype
        )
        target = pipe._pack_latents(noise * schedule[0], 1, channels, h_lat, w_lat)
        target_tokens = target.shape[1]

        packed_references: list[torch.Tensor] = []
        for vae_image in vae_images:
            ref_latent = pipe._encode_vae_image(
                vae_image.to(device=device, dtype=dtype), generator=generator
            )
            ref_h, ref_w = ref_latent.shape[3:]
            packed_references.append(pipe._pack_latents(ref_latent, 1, channels, ref_h, ref_w))
        img_shapes = [
            [
                (1, h_lat // 2, w_lat // 2),
                *[
                    (
                        1,
                        vae_height // pipe.vae_scale_factor // 2,
                        vae_width // pipe.vae_scale_factor // 2,
                    )
                    for vae_width, vae_height in vae_sizes
                ],
            ]
        ]

        x = target
        fixed_references = torch.cat(packed_references, dim=1)
        for t_cur, t_next in itertools.pairwise(schedule):
            model_input = torch.cat([x, fixed_references], dim=1)
            timestep = torch.full((1,), float(t_cur), device=device, dtype=dtype)
            flow = pipe.transformer(
                hidden_states=model_input,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states_mask=prompt_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                return_dict=False,
            )[0][:, :target_tokens]
            if do_cfg:
                negative_flow = pipe.transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states_mask=negative_mask,
                    encoder_hidden_states=negative_embeds,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0][:, :target_tokens]
                flow = (
                    negative_flow.double()
                    + float(guidance_scale) * (flow.double() - negative_flow.double())
                ).to(dtype)

            x0 = (x.double() - float(t_cur) * flow.double()).to(dtype)
            if t_next <= 1e-6:
                x = x0
                continue
            if sample_type == "ode":
                eps = (
                    (x.double() - (1.0 - float(t_cur)) * x0.double()) / max(float(t_cur), 1e-6)
                ).to(dtype)
            else:
                eps = torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
            x = ((1.0 - float(t_next)) * x0.double() + float(t_next) * eps.double()).to(dtype)

        decoded_latents = pipe._unpack_latents(x, height, width, pipe.vae_scale_factor)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        decoded_latents = decoded_latents * latents_std + latents_mean
        decoded = pipe.vae.decode(decoded_latents, return_dict=False)[0][:, :, 0]
        return QwenImageEditDMDOutput(
            images=pipe.image_processor.postprocess(decoded, output_type=output_type)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--base-pipeline-path", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--image", nargs="+", required=True, help="Ordered reference image(s).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-path")
    parser.add_argument("--output", default="qwen_image_edit_dmd2.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline = QwenImageEditDMDInferencePipeline.from_pretrained(
        args.student_path,
        args.base_pipeline_path,
        ema_path=args.ema_path,
    ).to(device)
    references = [load_image(path).convert("RGB") for path in args.image]
    output = pipeline(
        references,
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.images[0].save(args.output)


if __name__ == "__main__":
    main()
