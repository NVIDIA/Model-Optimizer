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

"""Wan2.2 inference pipeline for validation video generation and data preprocessing.

Wraps the official Wan2.2 T5 encoder, VAE, and denoising loop to work with
the unified trainer's cached-embeddings protocol.
"""

from __future__ import annotations

import importlib
import logging
import os

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import CachedEmbeddings, TextEmbeddings, free_gpu_memory
from .._deps import WAN_AVAILABLE

if WAN_AVAILABLE:
    from wan.modules.t5 import T5EncoderModel
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


class WanInferencePipeline:
    """Inference pipeline for Wan2.2 models.

    Manages T5 text encoder and VAE lifecycles:
    - T5 is loaded, used for embedding, then permanently deleted.
    - VAE stays on CPU during training, briefly moves to GPU for decode/encode.

    Adapts to different Wan variants (ti2v-5B, t2v-A14B, etc.) via the
    variant metadata from the loader module.
    """

    def __init__(self, variant: str | None = None) -> None:
        from .loader import get_variant_config

        self._text_encoder = None
        self._vae = None
        self._var = get_variant_config(variant)
        self._config = None

    def load_components(self, model_config, device: str, dtype: torch.dtype) -> None:
        if not WAN_AVAILABLE:
            raise ImportError("The 'wan' package is required for the Wan model backend.")

        path = str(getattr(model_config, "model_path", model_config))
        self._config = self._var["config"]()

        t5_path = os.path.join(path, self._config.t5_checkpoint)
        # Prefer local tokenizer dir (avoids HuggingFace network calls).
        # Wan ships tokenizer files under <model_root>/google/umt5-xxl/.
        tokenizer_path = self._config.t5_tokenizer
        local_tokenizer = os.path.join(path, tokenizer_path)
        if os.path.isdir(local_tokenizer):
            tokenizer_path = local_tokenizer
        self._text_encoder = T5EncoderModel(
            text_len=self._config.text_len,
            dtype=dtype,
            device=torch.device("cpu"),
            checkpoint_path=t5_path,
            tokenizer_path=tokenizer_path,
        )

        vae_mod = importlib.import_module(self._var["vae_module"])
        vae_cls = getattr(vae_mod, self._var["vae_class"])
        vae_path = os.path.join(path, self._config.vae_checkpoint)
        self._vae = vae_cls(vae_pth=vae_path, device=device)

        logger.info(f"Wan inference components loaded (T5 + {self._var['vae_class']})")

    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompt: str,
        device: str,
    ) -> list[CachedEmbeddings]:
        assert self._text_encoder is not None, "Call load_components() first"

        self._text_encoder.model.to(device)
        cached = []
        with torch.no_grad():
            for prompt in prompts:
                ctx_pos = self._text_encoder([prompt], torch.device(device))
                ctx_neg = self._text_encoder([negative_prompt], torch.device(device))
                cached.append(
                    CachedEmbeddings(
                        positive={"context": ctx_pos[0].cpu()},
                        negative={"context": ctx_neg[0].cpu()},
                    )
                )
        self._text_encoder.model.cpu()
        return cached

    def process_text_embeddings(
        self,
        raw_embeds: Tensor,
        attention_mask: Tensor,
    ) -> TextEmbeddings:
        """Identity: Wan T5 embeddings are used directly, no connector needed."""
        return TextEmbeddings(video_context=raw_embeds, audio_context=None)

    def unload_text_encoder(self) -> None:
        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None
        free_gpu_memory()
        logger.info("T5 text encoder unloaded")

    def offload_to_cpu(self) -> None:
        if self._vae is not None:
            self._vae.model.cpu()
        free_gpu_memory()

    def encode_videos(
        self,
        videos: list[Tensor],
        device: str,
    ) -> list[Tensor]:
        assert self._vae is not None, "Call load_components() first"
        self._vae.model.to(device)
        with torch.no_grad():
            latents = self._vae.encode([v.to(device) for v in videos])
        self._vae.model.cpu()
        free_gpu_memory()
        return [z.cpu() for z in latents]

    def generate(
        self,
        model: nn.Module,
        cached_embeds: list[CachedEmbeddings],
        config: dict,
        device: str,
    ) -> list[Tensor]:
        assert self._vae is not None
        assert self._config is not None

        width = config.get("width", 512)
        height = config.get("height", 320)
        num_frames = config.get("num_frames", 33)
        num_steps = config.get("num_inference_steps", 30)
        guidance_scale = config.get("guidance_scale", 5.0)
        seed = config.get("seed", 42)
        shift = self._config.sample_shift

        vae_stride = self._config.vae_stride
        n_f = (num_frames - 1) // vae_stride[0] + 1
        n_h = height // vae_stride[1]
        n_w = width // vae_stride[2]
        z_dim = self._vae.model.z_dim

        patch_size = self._config.patch_size
        seq_len = n_f * (n_h // patch_size[1]) * (n_w // patch_size[2])

        videos = []
        generator = torch.Generator(device=device).manual_seed(seed)

        for emb in cached_embeds:
            # Scheduler must be re-created per sample because its internal
            # step_index is not reset between runs.
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            scheduler.set_timesteps(num_steps, device=device, shift=shift)
            timesteps = scheduler.timesteps

            context = [emb.positive["context"].to(device)]
            context_null = [emb.negative["context"].to(device)]

            latent = torch.randn(
                z_dim, n_f, n_h, n_w, dtype=torch.float32, device=device, generator=generator
            )
            latents = [latent]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                for t in timesteps:
                    timestep = torch.stack([t])
                    timestep_expanded = timestep.expand(1, seq_len)

                    noise_pred_cond = model(
                        latents,
                        t=timestep_expanded,
                        context=context,
                        seq_len=seq_len,
                    )[0]
                    noise_pred_uncond = model(
                        latents,
                        t=timestep_expanded,
                        context=context_null,
                        seq_len=seq_len,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0]
                    latent = temp_x0.squeeze(0)
                    latents = [latent]

            self._vae.model.to(device)
            with torch.no_grad():
                decoded = self._vae.decode([latent])
            self._vae.model.cpu()

            video = ((decoded[0] + 1.0) / 2.0).clamp(0, 1).float().cpu()
            videos.append(video)

        free_gpu_memory()
        return videos
