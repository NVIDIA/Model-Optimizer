# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Ivan Zorin

from ltx_core.loader.sd_keys_ops import CompulsoryContent, ContentReplacement, SDKeyOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from ltx_core.model.video_vae.video_vae import Decoder, Encoder


class VAEEncoderConfigurator(ModelConfigurator[Encoder]):
    @classmethod
    def from_config(cls: type[Encoder], config: dict) -> Encoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        in_channels = config.get("in_channels", 3)
        latent_channels = config.get("latent_channels", 128)
        encoder_spatial_padding_mode = PaddingModeType(config.get("encoder_spatial_padding_mode", "zeros"))
        encoder_blocks = config.get("encoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        latent_log_var_str = config.get("latent_log_var", "uniform")

        return Encoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=latent_channels,
            encoder_blocks=encoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            latent_log_var=LogVarianceType(latent_log_var_str),
            encoder_spatial_padding_mode=encoder_spatial_padding_mode,
        )


class VAEDecoderConfigurator(ModelConfigurator[Decoder]):
    @classmethod
    def from_config(cls: type[Decoder], config: dict) -> Decoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        latent_channels = config.get("latent_channels", 128)
        decoder_spatial_padding_mode = PaddingModeType(config.get("decoder_spatial_padding_mode", "reflect"))
        out_channels = config.get("out_channels", 3)
        decoder_blocks = config.get("decoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        causal = config.get("causal_decoder", False)
        timestep_conditioning = config.get("timestep_conditioning", True)

        return Decoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=latent_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=decoder_spatial_padding_mode,
        )


VAE_DECODER_COMFY_KEYS_FILTER = SDKeyOps(
    name="VAE_DECODER_COMFY_KEYS_FILTER",
    mapping=(
        CompulsoryContent(prefix="vae.decoder."),
        CompulsoryContent(prefix="vae.per_channel_statistics."),
        ContentReplacement("vae.decoder.", ""),
        ContentReplacement("vae.per_channel_statistics.", "per_channel_statistics."),
    ),
)

VAE_ENCODER_COMFY_KEYS_FILTER = SDKeyOps(
    name="VAE_ENCODER_COMFY_KEYS_FILTER",
    mapping=(
        CompulsoryContent(prefix="vae.encoder."),
        CompulsoryContent(prefix="vae.per_channel_statistics."),
        ContentReplacement("vae.encoder.", ""),
        ContentReplacement("vae.per_channel_statistics.", "per_channel_statistics."),
    ),
)
