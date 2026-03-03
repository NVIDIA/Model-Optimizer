"""LTX-2 model loader.

Uses SingleGPUModelBuilder from ltx-core with the LTXModelConfigurator
and COMFY key renaming map for checkpoint loading.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .._deps import LTX_CORE_AVAILABLE

if TYPE_CHECKING:
    from pathlib import Path

if LTX_CORE_AVAILABLE:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

logger = logging.getLogger(__name__)


class LTX2ModelLoader:
    def load_transformer(
        self,
        path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        if not LTX_CORE_AVAILABLE:
            raise ImportError("The 'ltx_core' package is required for the LTX-2 model backend.")

        logger.info(f"Loading LTX-2 transformer from {path}")
        model = SingleGPUModelBuilder(
            model_path=str(path),
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        ).build(
            device=torch.device(device) if isinstance(device, str) else device,
            dtype=dtype,
        )
        return model
