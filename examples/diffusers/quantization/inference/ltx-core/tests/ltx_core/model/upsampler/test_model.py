# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import torch

from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator


def test_model() -> None:
    model = LatentUpsamplerConfigurator.from_config(
        {
            "in_channels": 1,
            "mid_channels": 32,
            "num_blocks_per_stage": 1,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": True,
            "spatial_scale": 2.0,
            "rational_resampler": False,
        }
    )
    assert model is not None
    latent = torch.randn(1, 1, 2, 2, 2)
    with torch.inference_mode():
        output = model(latent)
    assert output.shape == (1, 1, 3, 4, 4)
