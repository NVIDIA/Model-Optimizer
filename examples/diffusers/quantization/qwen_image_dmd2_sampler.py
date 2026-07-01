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

"""Compact DMD2 few-step sampler for Qwen-Image students.

This is a vendored, calibration-friendly version of the few-step unroll in
``examples/diffusers/fastgen/inference_dmd2_qwen_image.py``. It is kept here so
the quantization example is self-contained (no cross-example ``sys.path``
imports) and so calibration can run the **same forward logic the student was
trained/served with** — which is what makes the collected ``amax`` statistics
representative.

The single :func:`dmd2_sample` entry point serves two callers:

* **Calibration** (``decode=False``): runs only the transformer forwards of the
  DMD unroll and returns ``None``. The VAE / image post-processing is skipped
  because quantization only needs the transformer's activation statistics, and
  skipping the VAE saves substantial time and memory on the 60-layer student.
* **Sanity inference** (``decode=True``): additionally runs the VAE decode and
  returns a list of images, used to confirm a restored (quantized) student
  still produces a finite image.

The math is bit-aligned with the training-time ``_build_student_input`` in
``modelopt/torch/fastgen/methods/dmd.py`` and with the inference reference:

  for (t_cur, t_next) in pairwise(t_list):
      v   = student(x, t=t_cur, text_emb)        # flow at t_cur
      x_0 = x - t_cur * v                         # RF identity -> x_0 estimate
      if t_next > 0:
          eps = (x - (1 - t_cur) * x_0) / t_cur   # ODE: invert RF forward
          x   = (1 - t_next) * x_0 + t_next * eps # re-noise to t_next
      else:
          x   = x_0                               # final step

``t_list`` MUST match the student's training schedule (e.g. the LightX2V
"shift=3" 4-step shape ``[1.0, 0.9, 0.75, 0.5, 0.0]``); a mismatch produces a
train/inference gap and therefore misleading calibration statistics.
"""

from __future__ import annotations

import itertools

import torch
from diffusers.utils.torch_utils import randn_tensor

# Canonical 4-step "shift=3" student schedule (LightX2V-Qwen-Image-Lightning
# shape). t_list has student_sample_steps + 1 entries: the first N are the
# timesteps the student is evaluated at, the trailing 0.0 is the terminal the
# final Euler step lands on (NOT an extra evaluation).
DEFAULT_T_LIST: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.0)
DEFAULT_MAX_T: float = 0.999


def resolve_schedule(
    t_list: list[float] | tuple[float, ...] | None,
    sample_steps: int | None,
    max_t: float = DEFAULT_MAX_T,
) -> list[float]:
    """Resolve the sampling schedule (timesteps + terminal 0.0).

    Priority:
      1. An explicit ``t_list`` (must end at 0.0 and have ``sample_steps + 1``
         entries when ``sample_steps`` is given).
      2. ``sample_steps == 1`` -> ``[max_t, 0.0]`` (canonical single-step).
      3. ``sample_steps == 4`` (or None) with no ``t_list`` -> ``DEFAULT_T_LIST``.
      4. Otherwise a linear ``linspace(max_t, 0, sample_steps + 1)`` fallback.
    """
    if t_list is not None:
        schedule = [float(t) for t in t_list]
        if abs(schedule[-1]) > 1e-6:
            raise ValueError(
                f"t_list must end at 0.0 (got {schedule[-1]}); the final step lands on x_0."
            )
        if sample_steps is not None and len(schedule) != sample_steps + 1:
            raise ValueError(
                f"t_list must have sample_steps+1 entries "
                f"(got {len(schedule)} for sample_steps={sample_steps})."
            )
        return schedule

    if sample_steps == 1:
        return [float(max_t), 0.0]
    if sample_steps in (None, 4):
        return list(DEFAULT_T_LIST)
    return torch.linspace(float(max_t), 0.0, sample_steps + 1).tolist()


@torch.no_grad()
def dmd2_sample(
    pipe,
    prompt: str | list[str],
    *,
    schedule: list[float],
    sample_type: str = "ode",
    guidance_scale: float = 1.0,
    negative_prompt: str | list[str] | None = None,
    height: int = 1024,
    width: int = 1024,
    num_images_per_prompt: int = 1,
    generator: torch.Generator | None = None,
    max_sequence_length: int = 512,
    decode: bool = False,
    output_type: str = "pil",
) -> list | None:
    """Run the DMD few-step unroll on ``pipe.transformer``.

    Args:
        pipe: A ``QwenImagePipeline`` whose ``transformer`` is the DMD2 student.
        prompt: A prompt or list of prompts (one calibration batch).
        schedule: Full timestep schedule incl. trailing 0.0 (see
            :func:`resolve_schedule`).
        sample_type: ``"ode"`` (deterministic, recover eps via RF identity) or
            ``"sde"`` (fresh Gaussian noise between steps). Must match training.
        guidance_scale: Inference-time CFG. Leave at ``1.0`` for students trained
            with an internalised (non-null) ``dmd2.guidance_scale`` — passing
            ``> 1.0`` there would double-apply CFG.
        negative_prompt: Negative prompt for CFG; defaults to ``""`` when CFG is
            engaged and none is given.
        height/width: Output spatial size (must be VAE-compatible).
        num_images_per_prompt: Images per prompt.
        generator: Optional RNG for reproducible noise.
        max_sequence_length: Text-encoder max sequence length.
        decode: If ``True`` run VAE decode + post-process and return images. If
            ``False`` (calibration) skip the VAE and return ``None``.
        output_type: Passed to the image processor when ``decode=True``.

    Returns:
        A list of images when ``decode=True``, else ``None``.
    """
    if sample_type not in ("ode", "sde"):
        raise ValueError(f"sample_type must be 'ode' or 'sde', got {sample_type!r}")

    do_cfg = guidance_scale != 1.0
    if do_cfg and negative_prompt is None:
        negative_prompt = ""

    device = pipe.transformer.device
    dtype = next(pipe.transformer.parameters()).dtype

    # ---- Encode prompt(s) ------------------------------------------------
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    neg_prompt_embeds = neg_prompt_embeds_mask = None
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

    # ---- Build initial noisy latents at t = schedule[0] ------------------
    batch_size = (1 if isinstance(prompt, str) else len(prompt)) * num_images_per_prompt
    num_channels_latents = pipe.transformer.config.in_channels // 4  # 64 // 4 = 16
    h_lat = 2 * (height // (pipe.vae_scale_factor * 2))
    w_lat = 2 * (width // (pipe.vae_scale_factor * 2))
    latent_shape = (batch_size, 1, num_channels_latents, h_lat, w_lat)

    noise = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
    latents_5d = noise * schedule[0]  # RF: sigma(t0) = t0
    x_packed = pipe._pack_latents(latents_5d, batch_size, num_channels_latents, h_lat, w_lat)
    img_shapes = [[(1, h_lat // 2, w_lat // 2)]] * batch_size

    # ---- DMD few-step unroll (transformer forwards) ----------------------
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

        # RF identity: x_0 = x_t - t_cur * v (fp64 for stability).
        x0_packed = (x_packed.to(torch.float64) - float(t_cur) * flow_packed.to(torch.float64)).to(
            dtype
        )

        if t_next > 1e-6:
            if sample_type == "ode":
                alpha_cur = 1.0 - float(t_cur)
                eps_packed = (
                    (x_packed.to(torch.float64) - alpha_cur * x0_packed.to(torch.float64))
                    / max(float(t_cur), 1e-6)
                ).to(dtype)
            else:
                eps_packed = torch.randn(
                    x_packed.shape, generator=generator, device=device, dtype=dtype
                )
            alpha_next = 1.0 - float(t_next)
            x_packed = (
                alpha_next * x0_packed.to(torch.float64)
                + float(t_next) * eps_packed.to(torch.float64)
            ).to(dtype)
        else:
            x_packed = x0_packed

    if not decode:
        # Calibration path: transformer forwards already ran; nothing to decode.
        return None

    # ---- VAE decode (sanity-inference path only) -------------------------
    x0_5d = pipe._unpack_latents(x_packed, height, width, pipe.vae_scale_factor)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device=device, dtype=dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(device=device, dtype=dtype)
    x0_scaled = x0_5d / latents_std + latents_mean
    image_5d = pipe.vae.decode(x0_scaled, return_dict=False)[0]
    image_4d = image_5d[:, :, 0]  # Qwen-Image treats images as 1-frame videos
    return pipe.image_processor.postprocess(image_4d, output_type=output_type)
