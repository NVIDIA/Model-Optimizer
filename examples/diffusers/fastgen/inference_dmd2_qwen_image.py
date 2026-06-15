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

"""Inference pipeline for DMD2-trained Qwen-Image students.

Loads the consolidated safetensors transformer (from a §10 checkpoint with
``model_save_format=safetensors`` + ``save_consolidated=true``) plus the base
Qwen-Image VAE / text-encoder / tokenizer, and exposes a diffusers-style
``pipe(prompt=...).images[0]`` call that runs the DMD few-step sampler.

Math is bit-aligned with the training-time ``_build_student_input`` in
``modelopt/torch/fastgen/methods/dmd.py``:

  Single-step (Phase 1 default):
    x_T   = noise * max_t                     # initial latent
    v     = student(x_T, t=max_t, text_emb)   # one forward
    x_0   = x_T - max_t * v                   # RF identity
    image = vae.decode(x_0)

  Multi-step (Phase 2, ``num_inference_steps > 1``):
    x   = noise * t_list[0]                            # initial latent at t_max
    for (t_cur, t_next) in zip(t_list[:-1], t_list[1:]):
        v   = student(x, t=t_cur, text_emb)            # flow at t_cur
        x_0 = x - t_cur * v                            # RF identity → x_0 estimate
        if t_next > 0:
            eps = (x - (1 - t_cur) * x_0) / t_cur      # ODE: invert RF forward
            x   = (1 - t_next) * x_0 + t_next * eps    # re-noise to t_next
        else:
            x   = x_0                                  # final step
    image = vae.decode(x)

Usage::

    from inference_dmd2_qwen_image import QwenImageDMDInferencePipeline
    import torch

    pipe = QwenImageDMDInferencePipeline.from_pretrained(
        student_path="/path/to/checkpoint/epoch_0_step_500/model/consolidated",
        base_pipeline_path="Qwen/Qwen-Image",
        ema_path=None,  # or "…/epoch_0_step_5/ema_shadow.pt"
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    image = pipe(
        prompt="a small red cube on a white table",
        num_inference_steps=1,
        height=512,
        width=512,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    image.save("dmd2_smoke.png")
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QwenImageDMDOutput:
    """Container for inference outputs — mirrors diffusers' pipeline outputs."""

    images: list


class QwenImageDMDInferencePipeline:
    """Thin inference wrapper around a DMD2-trained Qwen-Image student.

    Wraps a stock ``diffusers.QwenImagePipeline`` whose ``transformer`` field
    has been swapped for our trained student. Re-uses the pipeline's VAE,
    text encoder, tokenizer, and image-processor for everything *except* the
    denoising loop, which is replaced by the DMD few-step sampler.
    """

    def __init__(
        self,
        base_pipeline: QwenImagePipeline,
        max_t: float = 0.999,
    ) -> None:
        self._pipe = base_pipeline
        self.max_t = max_t

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        student_path: str | Path,
        base_pipeline_path: str | Path,
        ema_path: str | Path | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_t: float = 0.999,
    ) -> QwenImageDMDInferencePipeline:
        """Load the student + base Qwen-Image components.

        Args:
            student_path: A consolidated dir from §10's safetensors save — must
                contain ``config.json``, ``diffusion_pytorch_model.safetensors.index.json``,
                and one or more ``*.safetensors`` shards. Loadable directly via
                ``QwenImageTransformer2DModel.from_pretrained``.
            base_pipeline_path: The base Qwen-Image checkpoint (e.g.
                ``Qwen/Qwen-Image`` or a local snapshot). Used only for the
                ``vae`` / ``text_encoder`` / ``tokenizer`` / ``image_processor``;
                the transformer is replaced.
            ema_path: Optional ``ema_shadow.pt`` produced by ``_save_dmd_extras``.
                If provided, the EMA shadow weights are overlaid onto the
                student after the safetensors load. EMA usually yields cleaner
                samples than the live student.
            torch_dtype: dtype for the student + VAE. ``bfloat16`` is what we
                trained with.
            max_t: Initial timestep for the 1-step sampler. Must match the
                recipe's ``sample_t_cfg.max_t`` (default 0.999).
        """
        student_path = str(student_path)
        base_pipeline_path = str(base_pipeline_path)
        if not os.path.isdir(student_path):
            raise FileNotFoundError(f"student_path is not a directory: {student_path}")
        if not os.path.isdir(base_pipeline_path):
            raise FileNotFoundError(f"base_pipeline_path is not a directory: {base_pipeline_path}")

        logger.info("[DMD2-Inference] Loading trained student from %s", student_path)
        student = QwenImageTransformer2DModel.from_pretrained(student_path, torch_dtype=torch_dtype)

        if ema_path is not None:
            logger.info("[DMD2-Inference] Overlaying EMA shadow from %s", ema_path)
            ema_state = torch.load(str(ema_path), map_location="cpu")
            shadow = (
                ema_state.get("shadow", ema_state) if isinstance(ema_state, dict) else ema_state
            )
            if not isinstance(shadow, dict):
                raise ValueError(
                    f"ema_shadow.pt content has unexpected type {type(shadow).__name__}; "
                    "expected dict[str, Tensor]."
                )
            missing, unexpected = student.load_state_dict(shadow, strict=False)
            if unexpected:
                logger.warning(
                    "[DMD2-Inference] EMA overlay had %d unexpected keys (first: %s)",
                    len(unexpected),
                    unexpected[:3],
                )
            if missing:
                logger.warning(
                    "[DMD2-Inference] EMA overlay missed %d student keys (first: %s)",
                    len(missing),
                    missing[:3],
                )

        student.eval()

        logger.info(
            "[DMD2-Inference] Loading base Qwen-Image pipeline from %s (transformer replaced)",
            base_pipeline_path,
        )
        # Passing transformer= bypasses loading the original transformer from disk;
        # the rest (vae, text_encoder, tokenizer, scheduler, image_processor) loads
        # normally.
        pipe = QwenImagePipeline.from_pretrained(
            base_pipeline_path,
            transformer=student,
            torch_dtype=torch_dtype,
        )

        return cls(base_pipeline=pipe, max_t=max_t)

    def to(self, device: str | torch.device) -> QwenImageDMDInferencePipeline:
        self._pipe.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._pipe.transformer.device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._pipe.transformer.parameters()).dtype

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list,
        negative_prompt: str | list | None = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        max_t: float | None = None,
        t_list: list | None = None,
        sample_type: str = "ode",
        output_type: str = "pil",
        max_sequence_length: int = 512,
    ) -> QwenImageDMDOutput:
        """Generate image(s) from the trained DMD2 student.

        ``num_inference_steps == 1`` runs the canonical Phase 1 single-step
        sampler (``x_0 = x_T - max_t * v``). ``num_inference_steps > 1`` runs
        the multi-step DMD unroll using ``t_list`` (or
        ``linspace(max_t, 0, num_inference_steps + 1)`` if ``t_list`` is None).

        ``sample_type`` selects between deterministic (``"ode"``, recovers eps
        from x_0 via RF identity) and stochastic (``"sde"``, fresh noise per
        step). Matches FastGen's ``student_sample_type``.

        ``guidance_scale != 1.0`` activates inference-time classifier-free
        guidance: the student is called twice (positive + negative prompt) per
        step and the two flows are blended as ``v = v_neg + s*(v_pos - v_neg)``.
        ``negative_prompt`` defaults to ``""`` (the canonical empty prompt) when
        unset and CFG is engaged.

        **Note on DMD2 students trained with CFG**: a student trained with
        ``dmd2.guidance_scale=4.0`` has *already* internalised CFG (its
        single-pass output is the CFG-augmented teacher target). Pass
        ``guidance_scale=1.0`` at inference to avoid double-CFG. Use
        ``guidance_scale > 1.0`` only for students that were trained without
        CFG (``dmd2.guidance_scale=null``).
        """
        if sample_type not in ("ode", "sde"):
            raise ValueError(f"sample_type must be 'ode' or 'sde', got {sample_type!r}")
        do_cfg = guidance_scale != 1.0
        if do_cfg and negative_prompt is None:
            # Default to the canonical empty unconditional prompt.
            negative_prompt = ""

        max_t = float(max_t) if max_t is not None else float(self.max_t)
        pipe = self._pipe
        device = self.device
        dtype = self.dtype

        # ---- 1. Resolve t_list -----------------------------------------------
        if num_inference_steps == 1:
            schedule = [max_t, 0.0]
        elif t_list is not None:
            if len(t_list) != num_inference_steps + 1:
                raise ValueError(
                    f"t_list must have num_inference_steps+1 entries "
                    f"(got {len(t_list)} for num_inference_steps={num_inference_steps})"
                )
            schedule = [float(t) for t in t_list]
            if abs(schedule[-1]) > 1e-6:
                raise ValueError(
                    f"t_list must end at 0.0 (got {schedule[-1]}); the final step lands on x_0."
                )
        else:
            # Default: linear schedule from max_t to 0 (matches FastGen's
            # torch.linspace(max_t, 0, sample_steps + 1) fallback).
            schedule = torch.linspace(max_t, 0.0, num_inference_steps + 1).tolist()

        # ---- 2. Encode prompt(s) ---------------------------------------------
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        neg_prompt_embeds = None
        neg_prompt_embeds_mask = None
        if do_cfg:
            neg_prompt_embeds, neg_prompt_embeds_mask = pipe.encode_prompt(
                prompt=negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).int().tolist() if prompt_embeds_mask is not None else None
        )
        neg_txt_seq_lens = (
            neg_prompt_embeds_mask.sum(dim=1).int().tolist()
            if neg_prompt_embeds_mask is not None
            else None
        )

        # ---- 3. Build initial noisy latents at t = schedule[0] ---------------
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        batch_size = batch_size * num_images_per_prompt

        num_channels_latents = pipe.transformer.config.in_channels // 4  # 64 // 4 = 16
        h_lat = 2 * (height // (pipe.vae_scale_factor * 2))
        w_lat = 2 * (width // (pipe.vae_scale_factor * 2))
        latent_shape = (batch_size, 1, num_channels_latents, h_lat, w_lat)

        # DMD initial latents: noise * schedule[0] (RF: sigma(t0) = t0).
        noise = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
        latents_5d = noise * schedule[0]

        latents_packed = pipe._pack_latents(
            latents_5d, batch_size, num_channels_latents, h_lat, w_lat
        )
        img_shapes = [[(1, h_lat // 2, w_lat // 2)]] * batch_size

        # ---- 4. DMD few-step unroll -----------------------------------------
        x_packed = latents_packed
        for t_cur, t_next in itertools.pairwise(schedule):
            timestep = torch.tensor([t_cur], device=device, dtype=dtype).expand(batch_size)
            flow_packed = pipe.transformer(
                hidden_states=x_packed,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                guidance=None,
                return_dict=False,
            )[0]
            if do_cfg:
                # CFG two-pass: ``v_cfg = v_neg + s*(v_pos - v_neg)``. Equivalent
                # to the FastGen training-time formula
                # ``teacher_pos + (s-1) * (teacher_pos - teacher_neg)`` once
                # expanded. Engaged only when the caller passes
                # ``guidance_scale != 1.0`` — DMD2 students trained with a
                # non-null ``dmd2.guidance_scale`` already internalise CFG, so
                # leave guidance_scale=1.0 for those.
                neg_flow_packed = pipe.transformer(
                    hidden_states=x_packed,
                    encoder_hidden_states=neg_prompt_embeds,
                    encoder_hidden_states_mask=neg_prompt_embeds_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=neg_txt_seq_lens,
                    guidance=None,
                    return_dict=False,
                )[0]
                flow_packed = (
                    neg_flow_packed.to(torch.float64)
                    + float(guidance_scale)
                    * (flow_packed.to(torch.float64) - neg_flow_packed.to(torch.float64))
                ).to(dtype)
            # RF identity: x_0 = x_t - t_cur * v (computed in fp64 for stability).
            x0_packed = (
                x_packed.to(torch.float64) - float(t_cur) * flow_packed.to(torch.float64)
            ).to(dtype)

            if t_next > 1e-6:
                # Re-noise x_0 forward to t_next.
                if sample_type == "ode":
                    # Deterministic: invert the RF forward to recover the implied eps,
                    # then re-noise. eps = (x_t - (1 - t_cur) * x_0) / t_cur
                    alpha_cur = 1.0 - float(t_cur)
                    eps_packed = (
                        (x_packed.to(torch.float64) - alpha_cur * x0_packed.to(torch.float64))
                        / max(float(t_cur), 1e-6)
                    ).to(dtype)
                else:
                    # Stochastic: fresh Gaussian noise.
                    eps_packed = torch.randn(
                        x_packed.shape, generator=generator, device=device, dtype=dtype
                    )
                # RF forward: x_{t_next} = (1 - t_next) * x_0 + t_next * eps.
                alpha_next = 1.0 - float(t_next)
                x_packed = (
                    alpha_next * x0_packed.to(torch.float64)
                    + float(t_next) * eps_packed.to(torch.float64)
                ).to(dtype)
            else:
                # Last step: x_0 is the output.
                x_packed = x0_packed

        # Unpack to 5D for VAE.
        x0_5d = pipe._unpack_latents(x_packed, height, width, pipe.vae_scale_factor)

        # ---- 5. VAE decode ---------------------------------------------------
        # Reverse the VAE-side scaling that diffusers applied at encoding time.
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(device=device, dtype=dtype)
        x0_scaled = x0_5d / latents_std + latents_mean

        # vae.decode returns 5D; the trailing [:, :, 0] drops the temporal dim
        # since Qwen-Image treats images as 1-frame videos.
        image_5d = pipe.vae.decode(x0_scaled, return_dict=False)[0]
        image_4d = image_5d[:, :, 0]  # [B, C, H, W]

        images = pipe.image_processor.postprocess(image_4d, output_type=output_type)
        return QwenImageDMDOutput(images=images)


# ---------------------------------------------------------------------------- #
# Standalone smoke-test driver. Validates end-to-end wiring against the §10    #
# safetensors checkpoint. Mock-data training means the image won't be          #
# coherent — pass criterion is just "the pipeline produces a finite image      #
# tensor and the file writes successfully".                                    #
# ---------------------------------------------------------------------------- #


def _smoke_test(
    student_path: str,
    base_pipeline_path: str,
    output_png: str,
    ema_path: str | None = None,
    prompt: str = "a small red cube on a white table",
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> None:
    """Run a one-shot inference and dump a PNG.

    Writes a small JSON sidecar next to the PNG with shape/dtype/range stats
    so the success criteria can be machine-checked.
    """
    import json

    logging.basicConfig(level=logging.INFO)
    pipe = QwenImageDMDInferencePipeline.from_pretrained(
        student_path=student_path,
        base_pipeline_path=base_pipeline_path,
        ema_path=ema_path,
        torch_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        num_inference_steps=1,
        height=height,
        width=width,
        generator=gen,
    )
    image = out.images[0]

    # PIL image; sanity-check shape + range.
    import numpy as np

    arr = np.array(image)
    stats = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "seed": seed,
        "ema_overlay": ema_path is not None,
        "image_shape": list(arr.shape),
        "image_dtype": str(arr.dtype),
        "image_min": int(arr.min()),
        "image_max": int(arr.max()),
        "image_mean": float(arr.mean()),
        "image_std": float(arr.std()),
        "is_finite": bool(np.isfinite(arr).all()),
        "is_not_constant": bool(arr.std() > 0),
    }

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    image.save(output_png)
    sidecar = output_png.replace(".png", "_stats.json")
    with open(sidecar, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nImage saved to: {output_png}")
    print(f"Stats sidecar: {sidecar}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student_path",
        required=True,
        help="Path to the consolidated safetensors student checkpoint "
        "(e.g. .../epoch_0_step_500/model/consolidated).",
    )
    parser.add_argument(
        "--base_pipeline_path",
        default="Qwen/Qwen-Image",
        help="Base Qwen-Image pipeline (HF id or local snapshot) for the VAE / text-encoder / tokenizer.",
    )
    parser.add_argument(
        "--output_png",
        default="./outputs/dmd2_sample.png",
    )
    parser.add_argument("--ema_path", default=None)
    parser.add_argument("--prompt", default="a small red cube on a white table")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _smoke_test(
        student_path=args.student_path,
        base_pipeline_path=args.base_pipeline_path,
        output_png=args.output_png,
        ema_path=args.ema_path,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
