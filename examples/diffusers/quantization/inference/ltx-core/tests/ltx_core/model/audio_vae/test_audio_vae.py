from tests.ltx_core.utils import resolve_model_path

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae.model_configurator import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, VAEDecoderConfigurator


def test_audio_vae_decoder_instantiation() -> None:
    vae_encoder = Builder(
        model_path=resolve_model_path(),
        model_class_configurator=VAEDecoderConfigurator,
        model_sd_key_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build()
    assert vae_encoder is not None
    assert not any(param.device.type == "meta" for param in vae_encoder.parameters())
