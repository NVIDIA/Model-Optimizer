from dataclasses import dataclass

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL.Image import Image as PilImage
from torch import Tensor
from torchvision.transforms.functional import center_crop, resize

from ltx_core.models.av_model import LTXAVModel
from ltx_core.models.model import LTXBaseModel
from ltx_core.models.text_encoders.gemma_emb_connector import GemmaConnector
from ltx_core.models.text_encoders.text_encoder_interface import BaseTextEncoder
from ltxv_trainer.utils import logger


@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        image (`PilImage`):
            The image to condition the video on.
        video (`list[PilImage]`):
            The video to condition the video on.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    image: PilImage | None = None
    video: list[PilImage] | None = None
    frame_index: int = 0
    strength: float = 1.0


# from LTX-Video/ltx_video/schedulers/rf.py
def linear_quadratic_schedule(
    num_steps: int, threshold_noise: float = 0.025, linear_steps: int | None = None
) -> Tensor:
    if linear_steps is None:
        linear_steps = num_steps // 2

    if num_steps < 2:
        return torch.tensor([1.0])

    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]

    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]

    return torch.tensor(sigma_schedule[:-1])


class LTXConditionPipeline:
    """
    Pipeline for text/image/video-to-video generation.
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: torch.nn.Module,
        text_encoder: BaseTextEncoder,
        transformer: LTXBaseModel,
        emb_connector: GemmaConnector,
        audio_vae: torch.nn.Module | None = None,
    ):
        """
        Args:
            transformer ([`LTXBaseModel`]):
                LTX transformer model (LTXV or LTX-2) to denoise the encoded video latents.
            scheduler ([`FlowMatchEulerDiscreteScheduler`]):
                A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
            vae ([`torch.nn.Module`]):
                Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
                Uses ltx_core VideoVAE implementation.
            text_encoder ([`BaseTextEncoder`]):
                Unified text encoder (T5 or Gemma) that handles both tokenization and encoding.
            emb_connector ([`GemmaConnector`]):
                Gemma connector to connect the text encoder to the transformer.
            audio_vae ([`torch.nn.Module`], *optional*):
                Audio VAE Model to encode and decode audio to and from latent representations.
                Required for audio generation with LTX-2 models.
        """

        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.emb_connector = emb_connector
        self.audio_vae = audio_vae

        # LTX VAE compression ratios
        self.vae_spatial_compression_ratio = 32
        self.vae_temporal_compression_ratio = 8
        self.num_channels_latents = 128

        # Audio constants (from LTXAVModel)
        self.num_audio_channels = 8
        self.audio_frequency_bins = 16

        # LTX transformers don't use patchification (patch_size=1)
        self.transformer_spatial_patch_size = 1
        self.transformer_temporal_patch_size = 1

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    @torch.inference_mode()
    def __call__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        conditions: LTXVideoCondition | list[LTXVideoCondition] | None = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 30,
        guidance_scale: float = 3,
        generator: torch.Generator | list[torch.Generator] | None = None,
        reference_video: Tensor | None = None,
        output_reference_comparison: bool = False,
        prompt_embeds: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        generate_audio: bool = False,
        encoder_tile_threshold: int = 768 * 768 * 33,
        decoder_tile_threshold: int = 24 * 24 * 5,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Pipeline's invocation method for text/image/video-to-video generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, defaults to empty string.
            conditions (`LTXVideoCondition` or `list[LTXVideoCondition]`, *optional*):
                Frame-conditioning for the video generation. Each condition specifies an image or video to condition
                on, along with the frame_index (where to apply it) and strength (how strongly to apply it).
                If None, generates video from text prompt only.
            height (`int`, defaults to `512`):
                The height in pixels of the generated video.
            width (`int`, defaults to `704`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate.
            frame_rate (`int`, defaults to `25`):
                The frame rate for the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to higher quality at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `3`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                Higher guidance scale encourages generation closely linked to the text `prompt`.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            reference_video (`Tensor`, *optional*):
                An optional reference video to guide the generation process. Should be a tensor with shape
                [F, C, H, W] in range [0, 1] as returned by `read_video()` from video_utils. The reference video
                will be encoded and concatenated to the latent sequence, providing global guidance while remaining
                unchanged during denoising. The reference video can be of any size and will be automatically
                resized and cropped to match the target dimensions.
            output_reference_comparison (`bool`, defaults to `False`):
                Whether to output a side-by-side comparison showing both the reference video (if provided) and the
                generated video. If `False`, only the generated video is returned. Only applies when `reference_video`
                is provided.
            prompt_embeds (`Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            generate_audio (`bool`, defaults to `False`):
                Whether to generate audio alongside video. Audio duration will automatically match video duration
                (calculated from num_frames and frame_rate). Only supported for LTX-2 models with audio VAE loaded.

        Returns:
            `tuple[Tensor, Tensor | None]`: A tuple containing:
                - video: Video tensor [B, C, F, H, W], float32 in range [0, 1]
                - audio: Audio waveform tensor [B, C, samples], float32 in range [-1, 1], or None if audio not generated

        Example:
            ```python
            from ltxv_trainer.pipeline import LTXVideoCondition
            from ltxv_trainer.utils import save_video
            from PIL import Image

            # Text-to-video (no conditioning)
            video, audio = pipeline(prompt="A cat playing with a ball")

            # Save video (first batch item) - save_video handles format conversion
            save_video(video[0], "output.mp4", fps=25)

            # Save with audio
            video, audio = pipeline(prompt="A dog barking", generate_audio=True)
            save_video(video[0], "output_with_audio.mp4", fps=25, audio=audio[0], audio_sample_rate=48000)
            ```
        """
        # 1. Check inputs. Raise error if not correct
        self._check_inputs(
            prompt=prompt,
            conditions=conditions,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            reference_video=reference_video,
            generate_audio=generate_audio,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Process conditions into separate lists
        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]

            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]
        else:
            # No frame conditioning
            strength = []
            frame_index = []
            image = []
            video = []

        device = next(self.transformer.parameters()).device

        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        vae_dtype = next(self.vae.parameters()).dtype

        conditioning_tensors = []
        is_conditioning_image_or_video = len(image) > 0 or len(video) > 0
        if is_conditioning_image_or_video:
            for condition_image, condition_video, condition_frame_index, _condition_strength in zip(
                image, video, frame_index, strength, strict=False
            ):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width)
                        .unsqueeze(2)
                        .to(device, dtype=vae_dtype)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(condition_video, height, width)
                    num_frames_input = condition_tensor.size(2)
                    num_frames_output = self._trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(device, dtype=vae_dtype)
                else:
                    raise ValueError("Either `image` or `video` must be provided for conditioning.")

                if condition_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form "
                        f"(k * {self.vae_temporal_compression_ratio} + 1) "
                        f"but got {condition_tensor.size(2)} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        # 4. Prepare latent variables
        latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self._prepare_latents(
            conditioning_tensors,
            strength,
            frame_index,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        # 4.5. Process reference video (if provided) and concatenate at the beginning
        reference_latents = None
        reference_num_latents = 0
        if reference_video is not None:
            # Work with the original tensor format [F, C, H, W]
            ref_frames = reference_video  # [F, C, H, W]

            # Resize maintaining aspect ratio (resize all frames)
            current_height, current_width = ref_frames.shape[2:]
            aspect_ratio = current_width / current_height
            target_aspect_ratio = width / height

            if aspect_ratio > target_aspect_ratio:
                # Width is relatively larger, resize based on height
                resize_height = height
                resize_width = int(resize_height * aspect_ratio)
            else:
                # Height is relatively larger, resize based on width
                resize_width = width
                resize_height = int(resize_width / aspect_ratio)

            ref_frames = resize(ref_frames, [resize_height, resize_width], antialias=True)

            # Center crop to target dimensions
            ref_frames = center_crop(ref_frames, [height, width])

            # Convert to VAE input format: [1, C, F, H, W] and proper range [-1, 1]
            reference_tensor = ref_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, F, C, H, W] -> [1, C, F, H, W]
            reference_tensor = reference_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

            # Trim reference video to proper frame count for temporal compression
            ref_num_frames_input = reference_tensor.size(2)
            ref_num_frames_output = self._trim_conditioning_sequence(0, ref_num_frames_input, num_frames)
            reference_tensor = reference_tensor[:, :, :ref_num_frames_output]
            reference_tensor = reference_tensor.to(device, dtype=vae_dtype)

            # Ensure proper frame count for VAE temporal compression
            if reference_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                # Trim to make it compatible with temporal compression
                ref_frames_to_keep = (
                    (reference_tensor.size(2) - 1) // self.vae_temporal_compression_ratio
                ) * self.vae_temporal_compression_ratio + 1
                reference_tensor = reference_tensor[:, :, :ref_frames_to_keep]

            # Expand reference tensor for batch size
            reference_tensor = reference_tensor.repeat(batch_size, 1, 1, 1, 1)

            # Encode reference video to latents
            # ltx_core VAE encode returns normalized latents directly
            reference_latents = self.vae.tiled_encode(reference_tensor, tiling_threshold=encoder_tile_threshold).to(
                device, dtype=torch.bfloat16
            )

            # Create "clean" coordinates for reference video (as if no frame conditioning applied)
            ref_latent_frames = reference_latents.size(2)
            ref_latent_height = reference_latents.size(3)
            ref_latent_width = reference_latents.size(4)

            reference_video_coords = self._prepare_video_ids(
                batch_size,
                ref_latent_frames,
                ref_latent_height,
                ref_latent_width,
                patch_size_t=self.transformer_temporal_patch_size,
                patch_size=self.transformer_spatial_patch_size,
                device=device,
            )
            reference_video_coords = self._scale_video_ids(
                reference_video_coords,
                scale_factor=self.vae_spatial_compression_ratio,
                scale_factor_t=self.vae_temporal_compression_ratio,
                frame_index=0,  # Reference video starts at frame 0
            )

            # Pack reference latents
            reference_latents = self._pack_latents(reference_latents)
            reference_num_latents = reference_latents.size(1)
            # Store reference dimensions for later unpacking
            reference_latent_frames = ref_latent_frames
            reference_latent_height = ref_latent_height
            reference_latent_width = ref_latent_width

            # Concatenate reference latents at the beginning: [reference_latents, frame_conditions, target_latents]
            latents = torch.cat([reference_latents, latents], dim=1)

            # Update video coordinates: [reference_coords, existing_coords]
            video_coords = torch.cat([reference_video_coords, video_coords], dim=2)

            # Update conditioning mask to include reference (frozen = strength 1.0)
            if conditioning_mask is not None:
                reference_conditioning_mask = torch.ones(
                    (batch_size, reference_num_latents), device=device, dtype=torch.bfloat16
                )
                conditioning_mask = torch.cat([reference_conditioning_mask, conditioning_mask], dim=1)
            else:
                # If no frame conditioning, still create mask for reference
                conditioning_mask = torch.ones((batch_size, reference_num_latents), device=device, dtype=torch.bfloat16)
                # Add zeros for target latents
                target_conditioning_mask = torch.zeros(
                    (batch_size, latents.size(1) - reference_num_latents),
                    device=device,
                    dtype=torch.bfloat16,
                )
                conditioning_mask = torch.cat([conditioning_mask, target_conditioning_mask], dim=1)

        # Calculate latent dimensions (needed for audio combination and timesteps)
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        # 4.6. Prepare audio latents and combine with video (if audio generation enabled)
        audio_length = 0
        if generate_audio:
            # Calculate audio duration from video parameters (audio matches video length)
            audio_duration = num_frames / frame_rate

            # Prepare random audio latents
            audio_latents = self._prepare_audio_latents(
                batch_size=batch_size,
                audio_duration=audio_duration,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )

            # Combine audio and video latents
            latents, audio_length = self._combine_audio_video_latents(
                video_latents=latents,
                audio_latents=audio_latents,
                latent_num_frames=latent_num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
            )

        if do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 5. Prepare timesteps
        sigmas = linear_quadratic_schedule(num_inference_steps)
        timesteps = sigmas * 1000
        self.scheduler.set_timesteps(timesteps=timesteps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Denoising loop
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # Handle conditioning mask for frame conditioning or reference video
            has_conditioning = is_conditioning_image_or_video or reference_video is not None
            if has_conditioning:
                conditioning_mask_model_input = (
                    torch.cat([conditioning_mask, conditioning_mask])
                    if do_classifier_free_guidance
                    else conditioning_mask
                )

            latent_model_input = latent_model_input.to(prompt_embeds.dtype)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
            if has_conditioning:
                timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

            # Convert timestep from per-token [B, seq_len] to single [B,]
            # Take the maximum timestep value (non-conditioning tokens have the sampled value)
            if timestep.ndim > 1:
                timestep_single = timestep.max(dim=1).values
                # Ensure it's [B,] shape
                while timestep_single.ndim > 1 and timestep_single.shape[-1] == 1:
                    timestep_single = timestep_single.squeeze(-1)
            else:
                timestep_single = timestep

            # Store the scalar timestep for audio (before per-token expansion)
            audio_timestep_scalar = t.expand(latent_model_input.shape[0]).float() if generate_audio else None

            # Prepare transformer input and kwargs
            # When reference video is present, we need to handle the concatenated sequence specially
            transformer_kwargs = {
                "frame_rate": frame_rate,
                "transformer_options": {
                    "run_vx": True,  # Always run video branch
                    "run_ax": generate_audio,  # Run audio branch if audio generation enabled
                    "a2v_cross_attn": generate_audio,  # Audio→video cross-attention
                    "v2a_cross_attn": generate_audio,  # Video→audio cross-attention
                },
                "audio_length": audio_length,  # Number of audio latent time steps (0 for video-only)
            }

            # Determine latent_shape for unpacking packed latents
            if reference_video is not None:
                # Total frames = reference frames + target frames
                total_frames = reference_latent_frames + latent_num_frames
                # Note: Assumes reference and target have same spatial dimensions (enforced by resize/crop)
                transformer_kwargs["latent_shape"] = (total_frames, latent_height, latent_width)
            else:
                # Just target frames (may include extra conditioning frames)
                transformer_kwargs["latent_shape"] = (latent_num_frames, latent_height, latent_width)

            # Create per-token timesteps [B, seq_len]
            # Conditioning tokens get timestep=0, denoising tokens get timestep=timestep_single
            batch_size_with_cfg = latent_model_input.shape[0]
            total_seq_len = latent_model_input.shape[1]
            timestep_per_token = timestep_single.unsqueeze(1).expand(batch_size_with_cfg, total_seq_len)

            # Set conditioning tokens to timestep 0
            if has_conditioning:
                # conditioning_mask_model_input: 1.0 = frozen (conditioning), 0.0 = denoise
                # Multiply by (1 - mask) to zero out conditioning tokens
                timestep_per_token = timestep_per_token * (1 - conditioning_mask_model_input)

            # Pass video_coords as pixel_coords if available (for reference video conditioning)
            if has_conditioning:
                # video_coords contains the positional information for the latent sequence
                # For reference video: includes both reference and target coordinates
                # For frame conditioning: includes conditioning and target coordinates

                # Convert video_coords from [B, 3, seq_len] to [B, 3, seq_len, 2] format
                # ltx_core expects start/end coordinates for each patch
                coords_start = video_coords  # [B, 3, seq_len]

                # Create end coordinates (start + scale_factors for patch_size=1)
                coords_end = coords_start.clone()
                coords_end[:, 0, :] += self.vae_temporal_compression_ratio  # +8 for temporal
                coords_end[:, 1, :] += self.vae_spatial_compression_ratio  # +32 for height
                coords_end[:, 2, :] += self.vae_spatial_compression_ratio  # +32 for width

                # Stack to [B, 3, seq_len, 2]
                pixel_coords = torch.stack([coords_start, coords_end], dim=-1)
                transformer_kwargs["pixel_coords"] = pixel_coords

            # Set denoise mask for conditioning
            # Invert: conditioning_mask=1.0 (frozen) -> denoise_mask=False (don't denoise)
            denoise_mask = 1 - conditioning_mask_model_input if has_conditioning else None
            transformer_kwargs["denoise_mask"] = denoise_mask

            # Call ltx_core transformer with new interface
            # When generating audio, pass timestep as tuple (v_timestep, a_timestep)
            if generate_audio:
                transformer_timestep = (
                    timestep_per_token / 1000,  # Video timestep (per-token)
                    audio_timestep_scalar.unsqueeze(-1) / 1000,  # Audio timestep (scalar, will be expanded)
                )
            else:
                transformer_timestep = timestep_per_token / 1000

            noise_pred = self.transformer(
                x=latent_model_input,
                timestep=transformer_timestep,
                context=prompt_embeds,
                attention_mask=prompt_attention_mask,
                **transformer_kwargs,
            )

            # Model returns packed format [B, seq_len, C]
            # No need to pack - already in the correct format for scheduler
            # noise_pred is already [B, total_seq_len, C] where total_seq_len includes
            # reference (if present) + conditioning frames (if present) + target frames

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                timestep, _ = timestep.chunk(2)

            denoised_latents = self.scheduler.step(
                -noise_pred,
                t,
                latents,
                per_token_timesteps=timestep,
                return_dict=False,
            )[0]
            if has_conditioning:
                tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
            else:
                latents = denoised_latents

        # Handle reference video output processing
        if reference_video is not None and output_reference_comparison:
            # Split latents: [reference_latents, frame_conditions, target_latents]
            reference_latents_out = latents[:, :reference_num_latents]
            remaining_latents = latents[:, reference_num_latents:]

            # Remove frame conditioning from remaining latents if needed
            if is_conditioning_image_or_video:
                target_latents_out = remaining_latents[:, extra_conditioning_num_latents:]
            else:
                target_latents_out = remaining_latents

            # Process both reference and target latents
            video_tensors = []
            latent_sets = [
                (reference_latents_out, reference_latent_frames, reference_latent_height, reference_latent_width),
                (target_latents_out, latent_num_frames, latent_height, latent_width),
            ]
            for curr_latents, curr_frames, curr_height, curr_width in latent_sets:
                # ltx_core VAE expects unpacked latents [B, C, F, H, W]
                # ltx_core VAE handles denormalization and timestep conditioning internally
                unpacked_latents = self._unpack_latents(
                    curr_latents,
                    curr_frames,
                    curr_height,
                    curr_width,
                )

                curr_video = self.vae.tiled_decode(
                    unpacked_latents.to(vae_dtype), tiling_threshold=decoder_tile_threshold
                )
                curr_video = self.video_processor.postprocess_video(curr_video, output_type="pt")
                video_tensors.append(curr_video)

            # Concatenate videos side-by-side (along width dimension)
            # Both tensors have shape [B, C, F, H, W]
            video = torch.cat(video_tensors, dim=4)  # Concatenate along width (dim=4)

            # Note: Audio generation not supported with reference video comparison mode
            if generate_audio:
                logger.warning("Audio generation not supported with reference video comparison mode")
            audio = None
        else:
            # Regular processing - just remove conditioning parts and output generated video
            if reference_video is not None:
                # Remove reference latents
                latents = latents[:, reference_num_latents:]

            if is_conditioning_image_or_video:
                latents = latents[:, extra_conditioning_num_latents:]

            # Decode video and audio (if audio generation enabled)
            if generate_audio:
                video, audio = self._decode_audio_video(
                    latents=latents,
                    audio_length=audio_length,
                    latent_num_frames=latent_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    device=device,
                    decoder_tile_threshold=decoder_tile_threshold,
                )
            else:
                # Video-only decoding
                latents = self._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                )

                # ltx_core VAE expects unpacked latents [B, C, F, H, W]
                # ltx_core VAE handles denormalization and timestep conditioning internally
                latents = latents.to(vae_dtype)
                video = self.vae.tiled_decode(latents, tiling_threshold=decoder_tile_threshold)
                video = self.video_processor.postprocess_video(video, output_type="pt")
                audio = None

        return video, audio

    def _check_inputs(  # noqa: PLR0912
        self,
        prompt: str | list[str] | None,
        conditions: LTXVideoCondition | list[LTXVideoCondition] | None,
        height: int,
        width: int,
        prompt_embeds: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        reference_video: Tensor | None = None,
        generate_audio: bool = False,
    ) -> None:
        """Validate input parameters for the pipeline call."""
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if conditions is not None:
            # Normalize to list for validation
            conditions_list = conditions if isinstance(conditions, list) else [conditions]
            for i, condition in enumerate(conditions_list):
                if not isinstance(condition, LTXVideoCondition):
                    raise ValueError(
                        f"All conditions must be LTXVideoCondition objects, but condition {i} is {type(condition)}"
                    )
                if condition.image is None and condition.video is None:
                    raise ValueError(f"Condition {i} must have either `image` or `video` set, but both are None.")
                if condition.image is not None and condition.video is not None:
                    raise ValueError(f"Condition {i} cannot have both `image` and `video` set. Choose one.")

        if reference_video is not None:
            if not isinstance(reference_video, Tensor):
                raise ValueError(
                    "`reference_video` must be a Tensor with shape [F, C, H, W] as returned by read_video()."
                )
            if reference_video.ndim != 4:
                raise ValueError(
                    f"`reference_video` must be a 4D tensor with shape [F, C, H, W], "
                    f"but got shape {reference_video.shape}."
                )

        # Audio-specific validations
        if generate_audio:
            if self.audio_vae is None:
                raise ValueError(
                    "Audio generation requires `audio_vae` to be loaded in the pipeline. "
                    "Please initialize the pipeline with an AudioVAE instance."
                )
            # Check if transformer supports audio (is LTXAVModel)
            if not isinstance(self.transformer, LTXAVModel):
                raise ValueError(
                    "Audio generation requires an LTX-2 model (LTXAVModel), "
                    f"but got {type(self.transformer).__name__}. "
                    "Please use an LTX-2 checkpoint that supports audio-video generation."
                )

    def _encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode prompts into text embeddings for both positive and negative guidance."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_prompt_embeds(prompt=prompt)

        if do_classifier_free_guidance:
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_prompt_embeds(prompt=negative_prompt)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def _get_prompt_embeds(self, prompt: str | list[str] | None = None) -> tuple[Tensor, Tensor]:
        """Generate text embeddings from prompts using the text encoder and optional emb_connector."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        self.text_encoder.to("cuda")
        # Get projected embeddings from text encoder (before emb_connector)
        projected_embeds, prompt_attention_mask = self.text_encoder.encode_text(prompt)
        self.text_encoder.to("cpu")

        # Apply emb_connector if available (for Gemma models)
        if self.emb_connector is not None:
            self.emb_connector.to("cuda")
            # Apply emb_connector to get final prompt embeddings
            prompt_embeds_v, prompt_attention_mask_v = self.emb_connector.preprocess_prompt_embeds(
                projected_embeds, prompt_attention_mask, is_audio=False
            )
            prompt_embeds_a, _ = self.emb_connector.preprocess_prompt_embeds(
                projected_embeds, prompt_attention_mask, is_audio=True
            )
            self.emb_connector.to("cpu")

            prompt_embeds = torch.cat([prompt_embeds_v, prompt_embeds_a], dim=-1)
            prompt_attention_mask = prompt_attention_mask_v
        else:
            # For T5 models, projected_embeds are already the final embeddings
            prompt_embeds = projected_embeds

        return prompt_embeds, prompt_attention_mask

    def _prepare_latents(  # noqa: PLR0913
        self,
        conditions: list[Tensor] | None = None,
        condition_strength: list[float] | None = None,
        condition_frame_index: list[int] | None = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        num_prefix_latent_frames: int = 2,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Initialize random latents and apply frame conditioning with proper coordinate handling."""
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if len(conditions) > 0:
            condition_latent_frames_mask = torch.zeros(
                (batch_size, num_latent_frames), device=device, dtype=torch.bfloat16
            )

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0
            for data, strength, frame_index in zip(conditions, condition_strength, condition_frame_index, strict=False):
                # ltx_core VAE encode returns normalized latents directly
                condition_latents = self.vae.encode(data).to(device, dtype=dtype)

                num_data_frames = data.size(2)
                num_cond_frames = condition_latents.size(2)

                if frame_index == 0:
                    latents[:, :, :num_cond_frames] = torch.lerp(
                        latents[:, :, :num_cond_frames], condition_latents, strength
                    )
                    condition_latent_frames_mask[:, :num_cond_frames] = strength

                else:
                    if num_data_frames > 1:
                        if num_cond_frames < num_prefix_latent_frames:
                            raise ValueError(
                                f"Number of latent frames must be at least {num_prefix_latent_frames} "
                                f"but got {num_data_frames}."
                            )

                        if num_cond_frames > num_prefix_latent_frames:
                            start_frame = frame_index // self.vae_temporal_compression_ratio + num_prefix_latent_frames
                            end_frame = start_frame + num_cond_frames - num_prefix_latent_frames
                            latents[:, :, start_frame:end_frame] = torch.lerp(
                                latents[:, :, start_frame:end_frame],
                                condition_latents[:, :, num_prefix_latent_frames:],
                                strength,
                            )
                            condition_latent_frames_mask[:, start_frame:end_frame] = strength
                            condition_latents = condition_latents[:, :, :num_prefix_latent_frames]

                    noise = randn_tensor(condition_latents.shape, generator=generator, device=device, dtype=dtype)
                    condition_latents = torch.lerp(noise, condition_latents, strength)

                    condition_video_ids = self._prepare_video_ids(
                        batch_size,
                        condition_latents.size(2),
                        latent_height,
                        latent_width,
                        patch_size=self.transformer_spatial_patch_size,
                        patch_size_t=self.transformer_temporal_patch_size,
                        device=device,
                    )
                    condition_video_ids = self._scale_video_ids(
                        condition_video_ids,
                        scale_factor=self.vae_spatial_compression_ratio,
                        scale_factor_t=self.vae_temporal_compression_ratio,
                        frame_index=frame_index,
                    )
                    condition_latents = self._pack_latents(condition_latents)
                    condition_conditioning_mask = torch.full(
                        condition_latents.shape[:2], strength, device=device, dtype=dtype
                    )

                    extra_conditioning_latents.append(condition_latents)
                    extra_conditioning_video_ids.append(condition_video_ids)
                    extra_conditioning_mask.append(condition_conditioning_mask)
                    extra_conditioning_num_latents += condition_latents.size(1)

        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )
        if len(conditions) > 0:
            conditioning_mask = condition_latent_frames_mask.gather(1, video_ids[:, 0])
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0
        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
        )
        latents = self._pack_latents(latents)

        if len(conditions) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat([*extra_conditioning_mask, conditioning_mask], dim=1)

        return latents, conditioning_mask, video_ids, extra_conditioning_num_latents

    @staticmethod
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: torch.device | None = None,
    ) -> Tensor:
        """Generate 3D coordinate grid for video latents in [B, 3, seq_len] format."""
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, num_frames, patch_size_t, device=device),
            torch.arange(0, height, patch_size, device=device),
            torch.arange(0, width, patch_size, device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

        return latent_coords

    @staticmethod
    def _scale_video_ids(
        video_ids: Tensor,
        scale_factor: int = 32,
        scale_factor_t: int = 8,
        frame_index: int = 0,
    ) -> Tensor:
        """Scale latent coordinates to pixel space and apply frame offset."""
        scaled_latent_coords = (
            video_ids
            * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
        )
        scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
        scaled_latent_coords[:, 0] += frame_index

        return scaled_latent_coords

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        """Pack latents from [B, C, F, H, W] to [B, seq_len, C] for transformer processing."""
        B, C, F, H, W = latents.shape  # noqa: N806
        # Permute to [B, F, H, W, C] and reshape to [B, F*H*W, C]
        latents = latents.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
        return latents

    @staticmethod
    def _unpack_latents(latents: Tensor, num_frames: int, height: int, width: int) -> Tensor:
        """Unpack latents from [B, seq_len, C] to [B, C, F, H, W] for VAE decoding."""
        B, seq_len, C = latents.shape  # noqa: N806
        assert seq_len == num_frames * height * width, f"Expected seq_len={num_frames * height * width}, got {seq_len}"
        # Reshape to [B, F, H, W, C] and permute to [B, C, F, H, W]
        latents = latents.reshape(B, num_frames, height, width, C).permute(0, 4, 1, 2, 3)
        return latents

    def _trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int) -> int:
        """Trim conditioning sequence to fit within target video length and align with VAE temporal compression."""
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    def _prepare_audio_latents(
        self,
        batch_size: int,
        audio_duration: float,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Initialize random audio latents for generation.

        Args:
            batch_size: Number of samples in batch
            audio_duration: Duration of audio in seconds
            generator: Random generator for reproducibility
            device: Device to create tensor on
            dtype: Data type for tensor

        Returns:
            Audio latent tensor with shape [batch, 8, time, 16]
        """
        # Calculate number of audio latent frames based on duration
        audio_latent_frames = int(audio_duration * self.audio_vae.latents_per_second)

        # Shape is [batch, channels, time, frequency_bins]  # noqa: ERA001
        shape = (
            batch_size,
            self.num_audio_channels,
            audio_latent_frames,
            self.audio_frequency_bins,
        )

        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _combine_audio_video_latents(
        self,
        video_latents: Tensor,
        audio_latents: Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> tuple[Tensor, int]:
        """Combine audio and video latents into a single tensor for processing.

        Args:
            video_latents: Packed video latents [batch, seq_len, channels]
            audio_latents: Audio latents [batch, 8, time, 16]
            latent_num_frames: Number of video latent frames
            latent_height: Video latent height
            latent_width: Video latent width

        Returns:
            Tuple of (combined_packed_latents, audio_length)
        """
        # Unpack video latents to [batch, 128, F, H, W]
        video_latents_unpacked = self._unpack_latents(
            video_latents,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Combine using transformer's method (handles padding)
        latents_combined = self.transformer._recombine_audio_and_video_latents(
            video_latents_unpacked,
            audio_latents,
        )

        # Pack back for transformer processing
        latents_packed = self._pack_latents(latents_combined)

        # Store audio length for later separation
        audio_length = audio_latents.size(2)

        return latents_packed, audio_length

    def _decode_audio_video(
        self,
        latents: Tensor,
        audio_length: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        decoder_tile_threshold: int,
    ) -> tuple[Tensor, Tensor]:
        """Decode combined audio-video latents to video tensor and audio waveform.

        Args:
            latents: Combined packed latents [batch, seq_len, channels]
            audio_length: Number of audio latent time steps
            latent_num_frames: Number of video latent frames
            latent_height: Video latent height
            latent_width: Video latent width
            device: Device for processing
            decoder_tile_threshold: Threshold for tiled decoding in VAE

        Returns:
            Tuple of (video_tensor, audio_waveform) where:
                - video_tensor: [B, C, F, H, W] float32 in range [0, 1]
                - audio_waveform: [B, C, samples] float32 in range [-1, 1]
        """
        # Unpack combined latents
        latents_unpacked = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Separate audio and video using transformer's method
        video_latents_final, audio_latents_final = self.transformer._separate_audio_and_video_latents(
            latents_unpacked,
            audio_length,
        )

        # Decode video
        video_vae_dtype = next(self.vae.parameters()).dtype
        video_latents_final = video_latents_final.to(video_vae_dtype)
        video = self.vae.tiled_decode(video_latents_final, tiling_threshold=decoder_tile_threshold)
        video = self.video_processor.postprocess_video(video, output_type="pt")

        # Decode audio (latents → mel spectrogram → waveform)
        # AudioVAE.decode() handles the complete pipeline internally:
        # 1. Denormalize latents
        # 2. Decode to mel spectrogram
        # 3. Convert mel to waveform via vocoder
        self.audio_vae.to(device)
        audio_vae_dtype = next(self.audio_vae.parameters()).dtype
        audio_latents_final = audio_latents_final.to(audio_vae_dtype)
        audio_waveform = self.audio_vae.decode(audio_latents_final)
        self.audio_vae.to("cpu")  # Offload to save memory

        return video, audio_waveform
