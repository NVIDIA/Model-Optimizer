"""LTX-2 inference pipeline for validation video generation and data preprocessing.

Uses ltx-core components (VideoLatentTools, LTX2Scheduler, X0Model, etc.) and
ltx-trainer's model_loader functions for checkpoint loading.

Text encoder lifecycle:
    load_components -> encode_prompts -> unload_text_encoder

The full text encoder (including embedding connectors) is used during
encode_prompts, so the returned CachedEmbeddings contain post-connector
embeddings ready for direct use by the model.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import CachedEmbeddings, free_gpu_memory
from .._deps import LTX_CORE_AVAILABLE, LTX_TRAINER_AVAILABLE

if LTX_TRAINER_AVAILABLE:
    from ltx_trainer.model_loader import (
        load_text_encoder,
        load_video_vae_decoder,
        load_video_vae_encoder,
    )

if LTX_CORE_AVAILABLE:
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

logger = logging.getLogger(__name__)

_LTX2_LATENT_CHANNELS = 128


class LTX2InferencePipeline:
    """Inference pipeline for LTX-2 models.

    Manages Gemma text encoder, video VAE encoder/decoder lifecycles.
    """

    def __init__(self) -> None:
        self._text_encoder = None
        self._vae_decoder = None
        self._vae_encoder = None

    def load_components(
        self, model_config, device: str, dtype: torch.dtype
    ) -> None:
        if not LTX_TRAINER_AVAILABLE:
            raise ImportError("The 'ltx_trainer' package is required for LTX-2 inference components.")

        checkpoint_path = model_config.model_path
        text_encoder_path = model_config.text_encoder_path

        if text_encoder_path is not None:
            logger.info("Loading Gemma text encoder ...")
            self._text_encoder = load_text_encoder(
                checkpoint_path=checkpoint_path,
                gemma_model_path=text_encoder_path,
                device="cpu",
                dtype=dtype,
            )
        else:
            logger.warning(
                "No text_encoder_path set -- skipping text encoder. "
                "encode_prompts() will not be available."
            )

        logger.info("Loading video VAE decoder ...")
        self._vae_decoder = load_video_vae_decoder(
            checkpoint_path, device="cpu", dtype=dtype
        )

        logger.info("Loading video VAE encoder ...")
        self._vae_encoder = load_video_vae_encoder(
            checkpoint_path, device="cpu", dtype=dtype
        )

        logger.info("LTX-2 inference components loaded")

    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompt: str,
        device: str,
    ) -> list[CachedEmbeddings]:
        assert self._text_encoder is not None, (
            "Text encoder not loaded. Provide text_encoder_path in model config."
        )

        self._text_encoder.to(device)
        cached = []

        with torch.no_grad():
            for prompt in prompts:
                # __call__ returns (video_embeds, audio_embeds, mask) -- post-connector
                v_pos, a_pos, _ = self._text_encoder(prompt)
                v_neg, a_neg, _ = self._text_encoder(negative_prompt)
                cached.append(CachedEmbeddings(
                    positive={
                        "video_context": v_pos.cpu(),
                        "audio_context": a_pos.cpu(),
                    },
                    negative={
                        "video_context": v_neg.cpu(),
                        "audio_context": a_neg.cpu(),
                    },
                ))

        # Keep connectors but offload heavy backbone
        self._text_encoder.model.to("cpu")
        self._text_encoder.feature_extractor_linear.to("cpu")

        return cached

    def unload_text_encoder(self) -> None:
        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None
        free_gpu_memory()
        logger.info("Gemma text encoder unloaded")

    def offload_to_cpu(self) -> None:
        if self._vae_decoder is not None:
            self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder.to("cpu")
        free_gpu_memory()

    def encode_videos(
        self,
        videos: list[Tensor],
        device: str,
    ) -> list[Tensor]:
        assert self._vae_encoder is not None, "VAE encoder not loaded"
        self._vae_encoder.to(device)
        latents = []
        with torch.no_grad():
            for video in videos:
                # video: [C, F, H, W] -> [1, C, F, H, W]
                inp = video.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z = self._vae_encoder(inp)  # [1, C, F', H', W']
                latents.append(z.squeeze(0).cpu())
        self._vae_encoder.to("cpu")
        free_gpu_memory()
        return latents

    def generate(
        self,
        model: nn.Module,
        cached_embeds: list[CachedEmbeddings],
        config: dict,
        device: str,
    ) -> list[Tensor]:
        if not LTX_CORE_AVAILABLE:
            raise ImportError("The 'ltx_core' package is required for LTX-2 inference.")

        height = config.get("height", 544)
        width = config.get("width", 960)
        num_frames = config.get("num_frames", 97)
        num_steps = config.get("num_inference_steps", 30)
        guidance_scale = config.get("guidance_scale", 4.0)
        seed = config.get("seed", 42)
        fps = config.get("frame_rate", 25.0)

        patchifier = VideoLatentPatchifier(patch_size=1)
        scale_factors = SpatioTemporalScaleFactors.default()

        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=fps,
        )
        video_tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape),
            fps=fps,
            scale_factors=scale_factors,
            causal_fix=True,
        )

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=num_steps).to(device).float()
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(guidance_scale)

        x0_model = X0Model(model)
        videos = []

        for emb in cached_embeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)

            v_ctx_pos = emb.positive["video_context"].to(device)
            v_ctx_neg = emb.negative["video_context"].to(device)

            video_state = video_tools.create_initial_state(
                device=device, dtype=torch.bfloat16
            )
            video_clean_state = video_state
            video_state = noiser(latent_state=video_state, noise_scale=1.0)

            video_mod = Modality(
                enabled=True,
                latent=video_state.latent,
                timesteps=video_state.denoise_mask,
                positions=video_state.positions,
                context=v_ctx_pos,
                context_mask=None,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for step_idx, sigma in enumerate(sigmas[:-1]):
                    video_mod = replace(
                        video_mod,
                        latent=video_state.latent,
                        timesteps=sigma * video_state.denoise_mask,
                        positions=video_state.positions,
                    )

                    # Positive pass
                    pos_video, _pos_audio = x0_model(
                        video=video_mod, audio=None, perturbations=None
                    )
                    denoised_video = pos_video

                    # CFG
                    if cfg_guider.enabled() and v_ctx_neg is not None:
                        video_neg = replace(video_mod, context=v_ctx_neg)
                        neg_video, _ = x0_model(
                            video=video_neg, audio=None, perturbations=None
                        )
                        denoised_video = denoised_video + cfg_guider.delta(
                            pos_video, neg_video
                        )

                    # Conditioning mask
                    denoised_video = (
                        denoised_video * video_state.denoise_mask
                        + video_clean_state.latent.float()
                        * (1 - video_state.denoise_mask)
                    )

                    # Euler step
                    video_state = replace(
                        video_state,
                        latent=stepper.step(
                            sample=video_mod.latent,
                            denoised_sample=denoised_video,
                            sigmas=sigmas,
                            step_index=step_idx,
                        ),
                    )

            # Decode
            video_state_clean = video_tools.clear_conditioning(video_state)
            video_state_unpatch = video_tools.unpatchify(video_state_clean)

            self._vae_decoder.to(device)
            latent = video_state_unpatch.latent.to(dtype=torch.bfloat16)
            with torch.no_grad():
                decoded = self._vae_decoder(latent)
            self._vae_decoder.to("cpu")

            video = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
            videos.append(video[0].float().cpu())  # [C, F, H, W]

        free_gpu_memory()
        return videos
