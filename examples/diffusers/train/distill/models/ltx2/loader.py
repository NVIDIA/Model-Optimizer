# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
