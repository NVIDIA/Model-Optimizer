"""Training strategies for different conditioning modes.

This module implements the Strategy Pattern to handle different training modes:
- Standard training (no conditioning)
- Reference video training (IC-LoRA mode)

Each strategy encapsulates the specific logic for preparing batches, model inputs, and loss computation.
"""

import random
from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel, computed_field, model_validator
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import pack_latents, prepare_video_coordinates, unpack_latents
from ltxv_trainer.timestep_samplers import TimestepSampler

DEFAULT_FPS = 24  # Default frames per second for video missing in the FPS metadata


class TrainingBatch(BaseModel):
    """Container for prepared training data.

    This model holds all the prepared data needed for a training step,
    organized in a way that's agnostic to the specific training strategy.
    """

    # Video latent data (packed format)
    video_latents: Tensor  # Packed video latents [B, seq_len, 128] where seq_len = F*H*W
    video_targets: Tensor  # Packed video targets [B, seq_len, 128] for loss computation
    video_coords: Tensor | None = None  # Optional explicit video coordinates for `video_latents`

    # Audio data (packed format, for audio-video training)
    audio_latents: Tensor | None = None  # Packed audio latents [B, T*16, 8]
    audio_targets: Tensor | None = None  # Packed audio targets [B, T*16, 8] for loss computation
    audio_length: int | None = None  # Number of audio time steps (T)

    # Text conditioning
    prompt_embeds: Tensor  # Text embeddings
    prompt_attention_mask: Tensor  # Attention mask for text

    # Timestep information
    timesteps: Tensor  # Timestep values for the video latents
    audio_timesteps: Tensor | None = None  # Timestep values for the audio latents
    sigmas: Tensor  # Noise schedule values

    # Conditioning information
    conditioning_mask: Tensor  # Boolean mask: True = conditioning token, False = target token

    # Video metadata
    num_frames: int  # Number of latent frames (target video)
    height: int  # Height of the video latents (target video)
    width: int  # Width of the video latents (target video)
    fps: float  # Frames per second

    # Reference video metadata (for IC-LoRA training)
    ref_num_frames: int | None = None  # Number of reference latent frames
    ref_height: int | None = None  # Height of reference latents
    ref_width: int | None = None  # Width of reference latents
    ref_seq_len: int | None = None  # Sequence length of reference latents

    @computed_field
    @property
    def batch_size(self) -> int:
        """Compute batch size from video latents tensor."""
        return self.video_latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        """Compute sequence length from video latents tensor."""
        seq_length = self.video_latents.shape[1]

        if self.audio_latents is not None:
            seq_length += self.audio_latents.shape[1]

        return seq_length

    @model_validator(mode="after")
    def validate_ic_lora_requirements(self) -> "TrainingBatch":
        """Validate IC-LoRA specific requirements."""
        # For IC-LoRA training, video_coords and reference metadata must be set
        if self.video_coords is not None and (
            self.ref_num_frames is None or self.ref_height is None or self.ref_width is None
        ):
            raise ValueError(
                "When `video_coords` is set (IC-LoRA mode), ref_num_frames, ref_height, and ref_width must also be set"
            )
        return self

    @model_validator(mode="after")
    def validate_audio_requirements(self) -> "TrainingBatch":
        """Validate audio training requirements."""
        # If any audio field is set, all must be set
        audio_fields = [self.audio_latents, self.audio_targets, self.audio_length]
        has_any = any(f is not None for f in audio_fields)
        has_all = all(f is not None for f in audio_fields)
        if has_any and not has_all:
            raise ValueError(
                "When audio training is enabled, audio_latents, audio_targets, and audio_length must all be set"
            )
        return self

    model_config = {"arbitrary_types_allowed": True}  # Allow torch.Tensor type


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.

    Each strategy encapsulates the logic for a specific training mode,
    handling batch preparation, model input preparation, and loss computation.
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize strategy with conditioning configuration.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        self.conditioning_config = conditioning_config

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Get the required data sources for this training strategy.

        Returns:
            Either a list of data directory names (where output keys match directory names)
            or a dictionary mapping data directory names to custom output keys for the dataset
        """

    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare a raw data batch for training.

        Args:
            batch: Raw batch data from the dataset
            timestep_sampler: Sampler for generating timesteps and noise

        Returns:
            Prepared training batch with all necessary data
        """

    @staticmethod
    def _create_timesteps_from_conditioning_mask(conditioning_mask: Tensor, sampled_timestep_values: Tensor) -> Tensor:
        """Create timesteps based on conditioning mask.

        Args:
            conditioning_mask: Boolean mask of shape (batch_size, sequence_length),
            where True = conditioning, False = target.
            sampled_timestep_values: Sampled timestep values for target tokens of shape (batch_size,)

        Returns:
            Timesteps tensor with 0 for conditioning tokens, sampled values for target tokens
        """
        # Expand sampled values to match conditioning mask shape
        expanded_timesteps = sampled_timestep_values.view(-1, 1).expand_as(conditioning_mask)

        # Use conditioning mask to select between 0 (conditioning) and sampled values (target)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    def _create_first_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Create conditioning mask for first frame conditioning.

        Returns:
            Boolean mask where True indicates first frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if (
            self.conditioning_config.first_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.first_frame_conditioning_p
        ):
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True

        return conditioning_mask

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model.

        Args:
            batch: Prepared training data

        Returns:
            Dictionary of keyword arguments for the transformer forward call
        """
        timestep = batch.timesteps / 1000

        # Create denoise_mask from conditioning_mask
        # denoise_mask: True = denoise this token, False = keep frozen
        # conditioning_mask: True = conditioning token (keep frozen), False = target token (denoise)
        denoise_mask = ~batch.conditioning_mask

        # Determine audio_length (0 for video-only training)
        audio_length = batch.audio_length or 0

        model_inputs = {
            "x": batch.video_latents,
            "timestep": timestep,
            "context": batch.prompt_embeds,
            "attention_mask": batch.prompt_attention_mask,
            "frame_rate": batch.fps,
            "denoise_mask": denoise_mask,
            "latent_shape": (batch.num_frames, batch.height, batch.width),
            "transformer_options": {},
            "audio_length": audio_length,
        }

        # Add pixel_coords if available (for IC-LoRA)
        if batch.video_coords is not None:
            model_inputs["pixel_coords"] = batch.video_coords

        return model_inputs

    @abstractmethod
    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute the training loss.

        Args:
            model_pred: Output from the transformer model
            batch: The prepared training data containing targets

        Returns:
            Scalar loss tensor
        """


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training strategy without conditioning.

    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Single video sequence length
    - Supports first frame conditioning
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize standard training strategy.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> list[str]:
        """Standard training requires latents and text conditions."""
        return ["latents", "conditions"]

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for standard training."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Create conditioning mask (only first frame conditioning for standard training)
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=target_latents.shape[0],
            sequence_length=target_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )

        # Create noise for the target latents
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)

        # Apply noise only to non-conditioning tokens
        sigmas = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

        # For conditioning tokens, use clean latents instead of noisy ones
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)  # (B, seq_len, 1)
        noisy_latents = torch.where(conditioning_mask_expanded, target_latents, noisy_latents)

        targets = noise - target_latents

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        return TrainingBatch(
            video_latents=noisy_latents,
            video_targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked MSE loss using conditioning mask."""
        loss = (model_pred - batch.video_targets).pow(2)

        # Create loss mask: exclude conditioning tokens
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()

        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()


class ReferenceVideoTrainingStrategy(TrainingStrategy):
    """Reference video training strategy for IC-LoRA.

    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates are doubled to handle concatenated sequence
    - Loss is computed only on the target portion (masked loss)
    - Supports first frame conditioning on the target sequence
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize with configurable reference latents directory.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents."""
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.conditioning_config.reference_latents_dir: "ref_latents",
        }

    def prepare_batch(self, batch: dict[str, dict[str, Tensor]], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for IC-LoRA training with reference videos."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Create noise only for the target part
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)
        sigmas = sigmas.view(-1, 1, 1)

        # Create conditioning mask
        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]

        # Reference tokens are always conditioning
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=target_latents.device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )

        # Combine reference and target conditioning masks
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Apply noise only to target part
        noisy_target = (1 - sigmas) * target_latents + sigmas * noise

        # For first frame conditioning in target, use clean latents instead of noisy ones
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)  # (B, target_seq_len, 1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        targets = noise - target_latents

        # Concatenate reference and noisy target in the sequence dimension
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        # Prepare video coordinates for reference and target sequences separately
        batch_size = combined_latents.shape[0]

        # Get reference video dimensions from the actual reference latents
        ref_latents_info = batch["ref_latents"]
        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

        # Generate coordinates for reference sequence
        ref_coords = prepare_video_coordinates(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            device=target_latents.device,
            start_end=True,  # ltx_core expects [B, 3, seq_len, 2] format
        )

        # Generate coordinates for target sequence
        target_coords = prepare_video_coordinates(
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            batch_size=batch_size,
            device=target_latents.device,
            start_end=True,  # ltx_core expects [B, 3, seq_len, 2] format
        )

        # Concatenate reference and target coordinates along sequence dimension
        video_coords = torch.cat([ref_coords, target_coords], dim=2)

        return TrainingBatch(
            video_latents=combined_latents,
            video_targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            video_coords=video_coords,
            ref_num_frames=ref_frames,
            ref_height=ref_height,
            ref_width=ref_width,
            ref_seq_len=ref_seq_len,
        )

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model with reference video handling.

        For IC-LoRA training, the latents are already concatenated in packed format
        [ref_seq + target_seq]. We pass them as-is with custom pixel_coords.
        """
        timestep = batch.timesteps / 1000

        # Create denoise_mask from conditioning_mask
        # For IC-LoRA:
        #   - Reference portion: all False (don't denoise, keep frozen)
        #   - Target portion: based on conditioning_mask (denoise non-conditioning tokens)
        # denoise_mask: True = denoise this token, False = keep frozen
        # conditioning_mask: True = conditioning token (keep frozen), False = target token (denoise)
        denoise_mask = ~batch.conditioning_mask

        # For IC-LoRA, we need to compute the total number of frames (ref + target)
        # to properly pass latent_shape to the model
        total_frames = batch.ref_num_frames + batch.num_frames

        # Note: We assume ref and target have the same spatial dimensions (H, W)
        # This is enforced during preprocessing (resize + center crop)
        return {
            "x": batch.video_latents,
            "timestep": timestep,
            "context": batch.prompt_embeds,
            "attention_mask": batch.prompt_attention_mask,
            "frame_rate": batch.fps,
            "denoise_mask": denoise_mask,
            "latent_shape": (total_frames, batch.height, batch.width),
            "pixel_coords": batch.video_coords,  # Custom coords for IC-LoRA
            "transformer_options": {},
            "audio_length": 0,  # Video-only training (no audio)
        }

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked loss only on target portion, excluding conditioning tokens.

        The model_pred is in packed format [B, seq_len, C] where seq_len includes both
        reference and target tokens. We extract the target portion and compute loss.
        """
        # Split model prediction into reference and target portions
        # model_pred shape: [B, ref_seq_len + target_seq_len, C]
        ref_seq_len = batch.ref_seq_len
        target_pred = model_pred[:, ref_seq_len:, :]

        # Get target conditioning mask
        target_conditioning_mask = batch.conditioning_mask[:, ref_seq_len:]

        # Compute loss
        loss = (target_pred - batch.video_targets).pow(2)

        # Create loss mask: exclude conditioning tokens
        loss_mask = (~target_conditioning_mask.unsqueeze(-1)).float()

        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()


class AudioVideoTrainingStrategy(TrainingStrategy):
    """Audio-video training strategy for LTX-2 models.

    This strategy handles joint audio-video training where:
    - Both video and audio latents are loaded and processed
    - Latents are combined for the transformer forward pass
    - Loss is computed on both video and audio portions
    - Supports first-frame conditioning on the video portion

    Note: This strategy only works with LTX-2 models (LTXAVModel) that support audio.
    For video-only models (LTXV), use StandardTrainingStrategy.
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize audio-video training strategy.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """Audio-video training requires latents, conditions, and audio latents."""
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.conditioning_config.audio_latents_dir: "audio_latents",
        }

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch with video and audio latents."""
        # Get pre-encoded video latents
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, seq_len, 128]

        # Get video dimensions
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Get audio latents
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]  # [B, T*16, 8]
        audio_time_steps = audio_data["num_time_steps"][0].item()

        # Create conditioning mask for video (first-frame conditioning)
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=video_latents.shape[0],
            sequence_length=video_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=video_latents.device,
        )

        # Sample timesteps and create noise
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to video
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens (first frame), use clean latents
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Apply noise to audio (same sigma as video)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Compute targets (velocity prediction)
        video_targets = video_noise - video_latents
        audio_targets = audio_noise - audio_latents

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Create audio timesteps - all audio tokens use the sampled timestep (no conditioning mask for audio)
        # Shape: [batch_size, audio_time_steps] - expanded to match number of audio tokens after patchification
        audio_timesteps = sampled_timestep_values.view(-1, 1).expand(-1, audio_time_steps)

        return TrainingBatch(
            video_latents=noisy_video,
            video_targets=video_targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            audio_latents=noisy_audio,
            audio_targets=audio_targets,
            audio_length=audio_time_steps,
            audio_timesteps=audio_timesteps,
        )

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model with combined audio-video latents.

        This method combines video and audio latents into the format expected by LTXAVModel,
        where audio is embedded in extra channels of the latent tensor.

        Args:
            batch: Prepared training data with separate video and audio latents

        Returns:
            Dictionary of keyword arguments for the transformer forward call
        """
        # Video timestep: per-token timesteps with 0 for conditioning tokens
        video_timestep = batch.timesteps / 1000
        # Audio timestep: per-token timesteps (all use sampled value, no conditioning mask)
        audio_timestep = batch.audio_timesteps / 1000

        # LTXAVModel expects timestep as [video_timestep, audio_timestep] for joint training
        timestep = [video_timestep, audio_timestep]

        # Create denoise_mask from conditioning_mask (only for video tokens)
        # Note: denoise_mask is only used for video; audio doesn't have conditioning tokens
        denoise_mask = ~batch.conditioning_mask

        # Combine video and audio latents for transformer input
        combined_latents = AudioVideoTrainingStrategy._combine_video_audio_latents(
            video_latents=batch.video_latents,
            audio_latents=batch.audio_latents,
            num_frames=batch.num_frames,
            height=batch.height,
            width=batch.width,
            audio_time_steps=batch.audio_length,
        )

        model_inputs = {
            "x": combined_latents,
            "timestep": timestep,
            "context": batch.prompt_embeds,
            "attention_mask": batch.prompt_attention_mask,
            "frame_rate": batch.fps,
            "denoise_mask": denoise_mask,
            "latent_shape": (batch.num_frames, batch.height, batch.width),
            "transformer_options": {
                "run_vx": True,  # Always run video branch
                "run_ax": True,  # Run audio branch if audio generation enabled
                "a2v_cross_attn": True,  # Audio→video cross-attention
                "v2a_cross_attn": True,  # Video→audio cross-attention
            },
            "audio_length": batch.audio_length,
        }

        return model_inputs

    @staticmethod
    def _combine_video_audio_latents(
        video_latents: Tensor,
        audio_latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        audio_time_steps: int,
    ) -> Tensor:
        """Combine video and audio latents for transformer input.

        This replicates the logic from LTXAVModel._recombine_audio_and_video_latents
        to properly combine video and audio latents in the format expected by the model.

        Args:
            video_latents: Packed video latents [B, seq_len, 128] where seq_len = F*H*W
            audio_latents: Packed audio latents [B, T*freq, 8] where freq=16
            num_frames: Number of video latent frames (F)
            height: Video latent height (H)
            width: Video latent width (W)
            audio_time_steps: Number of audio time steps (T)

        Returns:
            Combined packed latents [B, seq_len, C] where C = 128 + extra_audio_channels
        """
        batch_size = video_latents.shape[0]
        audio_seq_len = audio_latents.shape[1]  # T * freq
        audio_channels = audio_latents.shape[2]  # 8
        audio_freq_bins = audio_seq_len // audio_time_steps  # freq = seq_len / T

        # Unpack video latents: [B, seq_len, 128] -> [B, 128, F, H, W]
        video_unpacked = unpack_latents(video_latents, num_frames, height, width)

        # Unpack audio latents: [B, T*freq, 8] -> [B, 8, T, freq]
        audio_unpacked = audio_latents.reshape(batch_size, audio_time_steps, audio_freq_bins, audio_channels)
        audio_unpacked = audio_unpacked.permute(0, 3, 1, 2)  # [B, 8, T, freq]

        # Flatten audio: [B, 8, T, freq] -> [B, 8*T*freq]
        audio_flat = audio_unpacked.reshape(batch_size, -1)

        # Pad audio to be divisible by F*H*W (same logic as _recombine_audio_and_video_latents)
        divisor = num_frames * height * width
        padded_len = ((audio_flat.shape[-1] + divisor - 1) // divisor) * divisor
        audio_padded = torch.nn.functional.pad(audio_flat, (0, padded_len - audio_flat.shape[-1]))

        # Reshape to [B, extra_channels, F, H, W]
        extra_channels = padded_len // divisor
        audio_reshaped = audio_padded.reshape(batch_size, extra_channels, num_frames, height, width)

        # Concatenate on channel dimension: [B, 128 + extra_channels, F, H, W]
        combined_unpacked = torch.cat([video_unpacked, audio_reshaped], dim=1)

        # Pack back to [B, seq_len, C]
        combined_packed = pack_latents(combined_unpacked)

        return combined_packed

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute combined loss for video and audio.

        For audio-video training:
        - Model output is in combined format [B, seq_len, C] where C = 128 + extra_audio_channels
        - We separate video (first 128 channels) and audio (remaining channels)
        - Video loss is computed on the video portion with conditioning mask
        - Audio loss is computed on the audio portion (extracted and reshaped)
        - Total loss = video_loss + audio_loss (equal weighting)
        """
        if batch.audio_latents is None:
            # Video-only: use standard loss computation
            # model_pred should be [B, seq_len, 128]
            loss = (model_pred - batch.video_targets).pow(2)
            loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()
            loss = loss.mul(loss_mask).div(loss_mask.mean())
            return loss.mean()

        # Separate video and audio from model prediction
        # model_pred: [B, seq_len, C] where C = 128 + extra_audio_channels, seq_len = F*H*W
        video_pred = model_pred[:, :, :128]  # [B, seq_len, 128]

        # Video loss (with conditioning mask)
        video_loss = (video_pred - batch.video_targets).pow(2)
        video_loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # Extract audio prediction from extra channels
        # audio_pred_packed: [B, seq_len, extra_channels] where seq_len = F*H*W
        audio_pred_packed = model_pred[:, :, 128:]

        batch_size = model_pred.shape[0]
        audio_length = batch.audio_length  # T (time steps)

        # Derive audio shape from targets: [B, T*freq, 8]
        audio_seq_len = batch.audio_targets.shape[1]  # T * freq
        audio_channels = batch.audio_targets.shape[2]  # 8
        audio_freq_bins = audio_seq_len // audio_length  # freq

        # Unpack audio: [B, seq_len, extra_channels] -> [B, extra_channels, F, H, W]
        # This reverses pack_latents which does [B, C, F, H, W] -> [B, F*H*W, C]
        extra_channels = audio_pred_packed.shape[2]
        audio_unpacked = audio_pred_packed.reshape(
            batch_size, batch.num_frames, batch.height, batch.width, extra_channels
        )
        audio_unpacked = audio_unpacked.permute(0, 4, 1, 2, 3)  # [B, extra_channels, F, H, W]

        # Flatten to extract valid audio: [B, extra_channels * F * H * W]
        audio_flat = audio_unpacked.reshape(batch_size, -1)

        # Extract valid audio portion: first 8 * T * freq elements
        valid_audio_len = audio_channels * audio_length * audio_freq_bins
        audio_valid = audio_flat[:, :valid_audio_len]

        # Reshape to unpacked audio format: [B, 8, T, freq]
        audio_pred_unpacked = audio_valid.reshape(batch_size, audio_channels, audio_length, audio_freq_bins)

        # Convert to packed format [B, T*freq, 8] to match targets
        audio_pred_final = audio_pred_unpacked.permute(0, 2, 3, 1)  # [B, T, freq, 8]
        audio_pred_final = audio_pred_final.reshape(batch_size, -1, audio_channels)  # [B, T*freq, 8]

        # Audio loss (no conditioning mask for audio)
        audio_loss = (audio_pred_final - batch.audio_targets).pow(2).mean()

        # Combined loss (equal weighting)
        return (video_loss + audio_loss) / 2


def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.

    Args:
        conditioning_config: Configuration for conditioning behavior

    Returns:
        The appropriate training strategy instance

    Raises:
        ValueError: If conditioning mode is not supported
    """
    conditioning_mode = conditioning_config.mode

    if conditioning_mode == "none":
        strategy = StandardTrainingStrategy(conditioning_config)
    elif conditioning_mode == "reference_video":
        strategy = ReferenceVideoTrainingStrategy(conditioning_config)
    elif conditioning_mode == "audio_video":
        strategy = AudioVideoTrainingStrategy(conditioning_config)
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")

    logger.debug(f"🎯 Using {strategy.__class__.__name__}")
    return strategy
