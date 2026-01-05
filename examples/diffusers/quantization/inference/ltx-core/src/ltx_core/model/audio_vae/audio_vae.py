# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Ivan Zorin


from typing import Set, Tuple

import torch
import torch.nn.functional as F

from ltx_core.model.audio_vae.attention import AttentionType, AttnBlock
from ltx_core.model.audio_vae.causal_conv_2d import make_conv2d
from ltx_core.model.audio_vae.causality_axis import CausalityAxis
from ltx_core.model.audio_vae.ops import PerChannelStatistics
from ltx_core.model.audio_vae.resnet import ResnetBlock
from ltx_core.model.audio_vae.upsample import Upsample
from ltx_core.model.video_vae.normalization import build_normalization_layer
from ltx_core.pipeline.components.patchifiers import AudioPatchifier
from ltx_core.pipeline.components.protocols import AudioLatentShape

LATENT_DOWNSAMPLE_FACTOR = 4


def make_attn(
    in_channels: int, attn_type: AttentionType = AttentionType.VANILLA, norm_type: str = "group"
) -> torch.nn.Module:
    match attn_type:
        case AttentionType.VANILLA:
            return AttnBlock(in_channels, norm_type=norm_type)
        case AttentionType.NONE:
            return torch.nn.Identity(in_channels)
        case AttentionType.LINEAR:
            raise NotImplementedError(f"Attention type {attn_type.value} is not supported yet.")
        case _:
            raise ValueError(f"Unknown attention type: {attn_type}")


class Decoder(torch.nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.

    The decoder mirrors the encoder structure with configurable channel multipliers,
    attention resolutions, and causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        in_channels: int,
        resolution: int,
        z_channels: int,
        norm_type: str = "group",
        causality_axis: CausalityAxis | str = CausalityAxis.WIDTH,
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = None,
    ) -> None:
        """
        Initialize the Decoder.

        Args:
            Arguments are configuration parameters, loaded from the audio VAE checkpoint config
            (audio_vae.model.params.ddconfig):
            - ch, out_ch, ch_mult, num_res_blocks, attn_resolutions
            - in_channels, resolution, z_channels
            - norm_type, causality_axis
        """
        super().__init__()

        # Internal behavioural defaults that are not driven by the checkpoint.
        resamp_with_conv = True
        give_pre_end = False
        tanh_out = False
        attn_type = AttentionType.VANILLA

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )
        self.non_linearity = torch.nn.SiLU()
        self.mid = self._build_mid_layers(base_block_channels, dropout, mid_block_add_attention)
        self.up, final_block_channels = self._build_up_path(
            initial_block_channels=base_block_channels,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """
        Adjust output shape to match target dimensions for variable-length audio.

        This function handles the common case where decoded audio spectrograms need to be
        resized to match a specific target shape.

        Args:
            decoded_output: Tensor of shape (batch, channels, time, frequency)
            target_shape: AudioLatentShape describing (batch, channels, time, mel bins)

        Returns:
            Tensor adjusted to match target_shape exactly
        """
        # Current output shape: (batch, channels, time, frequency)
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Step 1: Crop first to avoid exceeding target dimensions
        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        # Step 2: Calculate padding needed for time and frequency dimensions
        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        # Step 3: Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
            # For audio: pad_left/right = frequency, pad_top/bottom = time
            padding = (
                0,
                max(freq_padding_needed, 0),  # frequency padding (left, right)
                0,
                max(time_padding_needed, 0),  # time padding (top, bottom)
            )
            decoded_output = F.pad(decoded_output, padding)

        # Step 4: Final safety crop to ensure exact target shape
        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def forward(
        self,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent features back to audio spectrograms.

        Args:
            sample: Encoded latent representation of shape (batch, channels, height, width)

        Returns:
            Reconstructed audio spectrogram of shape (batch, channels, time, frequency)
        """
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        # Denormalize latents via per-channel statistics operated in patch space.
        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR

        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        hidden_features = self.conv_in(sample)
        hidden_features = self._run_mid_layers(hidden_features)
        hidden_features = self._run_upsampling_path(hidden_features)
        decoded_output = self._finalize_output(hidden_features)

        # Adjust shape for audio data
        decoded_output = self._adjust_output_shape(decoded_output, target_shape)

        return decoded_output

    def _build_mid_layers(self, channels: int, dropout: float, add_attention: bool) -> torch.nn.Module:
        mid = torch.nn.Module()
        mid.block_1 = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        mid.attn_1 = (
            make_attn(channels, attn_type=self.attn_type, norm_type=self.norm_type)
            if add_attention
            else torch.nn.Identity()
        )
        mid.block_2 = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        return mid

    def _build_up_path(
        self,
        *,
        initial_block_channels: int,
        dropout: float,
        resamp_with_conv: bool,
    ) -> tuple[torch.nn.ModuleList, int]:
        up_modules = torch.nn.ModuleList()
        block_in = initial_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage = torch.nn.Module()
            stage.block = torch.nn.ModuleList()
            stage.attn = torch.nn.ModuleList()
            block_out = self.ch * self.channel_multipliers[level]

            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    stage.attn.append(make_attn(block_in, attn_type=self.attn_type, norm_type=self.norm_type))

            if level != 0:
                stage.upsample = Upsample(block_in, resamp_with_conv, causality_axis=self.causality_axis)
                curr_res *= 2

            up_modules.insert(0, stage)

        return up_modules, block_in

    def _run_mid_layers(self, features: torch.Tensor) -> torch.Tensor:
        features = self.mid.block_1(features, temb=None)
        features = self.mid.attn_1(features)
        return self.mid.block_2(features, temb=None)

    def _run_upsampling_path(self, features: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                features = block(features, temb=None)
                if stage.attn:
                    features = stage.attn[block_idx](features)

            if level != 0 and hasattr(stage, "upsample"):
                features = stage.upsample(features)

        return features

    def _finalize_output(self, features: torch.Tensor) -> torch.Tensor:
        if self.give_pre_end:
            return features

        hidden = self.norm_out(features)
        hidden = self.non_linearity(hidden)
        decoded = self.conv_out(hidden)
        return torch.tanh(decoded) if self.tanh_out else decoded
