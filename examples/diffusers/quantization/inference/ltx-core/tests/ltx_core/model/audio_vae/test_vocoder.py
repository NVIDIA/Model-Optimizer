# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Ivan Zorin


from tests.ltx_core.utils import resolve_model_path

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae.model_configurator import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator


def test_vocoder() -> None:
    builder = Builder(
        model_path=resolve_model_path(),
        model_class_configurator=VocoderConfigurator,
        model_sd_key_ops=VOCODER_COMFY_KEYS_FILTER,
    )
    model = builder.build()
    assert model is not None
