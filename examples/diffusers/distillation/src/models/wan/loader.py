"""Wan2.2 model loader.

Supports multiple Wan model variants (ti2v-5B, t2v-A14B, i2v-A14B) via the
VARIANTS dict. Each variant maps to its wan config object and VAE metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path

from .._deps import WAN_AVAILABLE

if WAN_AVAILABLE:
    from wan.configs.wan_i2v_A14B import i2v_A14B
    from wan.configs.wan_t2v_A14B import t2v_A14B
    from wan.configs.wan_ti2v_5B import ti2v_5B
    from wan.modules.model import WanModel

logger = logging.getLogger(__name__)

_VARIANTS = {
    "ti2v-5B": {
        "config": lambda: ti2v_5B,
        "z_dim": 48,
        "vae_stride": (4, 16, 16),
        "vae_module": "wan.modules.vae2_2",
        "vae_class": "Wan2_2_VAE",
    },
    "t2v-A14B": {
        "config": lambda: t2v_A14B,
        "z_dim": 16,
        "vae_stride": (4, 8, 8),
        "vae_module": "wan.modules.vae2_1",
        "vae_class": "Wan2_1_VAE",
    },
    "i2v-A14B": {
        "config": lambda: i2v_A14B,
        "z_dim": 16,
        "vae_stride": (4, 8, 8),
        "vae_module": "wan.modules.vae2_1",
        "vae_class": "Wan2_1_VAE",
    },
}


def get_variant_config(variant: str | None) -> dict:
    """Return the variant metadata dict, defaulting to ti2v-5B."""
    variant = variant or "ti2v-5B"
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown Wan variant '{variant}'. Available: {list(_VARIANTS)}")
    return _VARIANTS[variant]


class WanModelLoader:
    def __init__(self, variant: str | None = None) -> None:
        self.variant = variant or "ti2v-5B"
        self._var = get_variant_config(self.variant)

    def load_transformer(
        self,
        path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        if not WAN_AVAILABLE:
            raise ImportError("The 'wan' package is required for the Wan model backend.")

        logger.info(f"Loading WanModel ({self.variant}) from {path}")
        model = WanModel.from_pretrained(str(path))
        model = model.to(device=device, dtype=dtype)
        return model
