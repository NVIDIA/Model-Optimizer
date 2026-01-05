# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import pytest
import torch

from ltx_core.model.transformer.model_configurator import LTXModelConfigurator


def test_model() -> None:
    transformer_config = {
        "transformer": {
            "dropout": 0.0,
            "norm_num_groups": 32,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "causal_temporal_positioning": True,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "use_middle_indices_grid": True,
        }
    }
    with torch.device("meta"):
        with pytest.raises(ValueError, match="Config value"):
            LTXModelConfigurator.from_config({})
        model = LTXModelConfigurator.from_config(transformer_config)
    assert model is not None
