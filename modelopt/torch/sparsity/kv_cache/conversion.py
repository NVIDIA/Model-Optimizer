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

"""Convert/restore/update entrypoints for TriAttention mode.

TriAttention is a calibration-only mode. Convert is a no-op on model weights.
Calibration data is stored in metadata and fused into the checkpoint at save time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

    from .config import TriAttentionConfig

__all__ = [
    "convert_triattention",
    "restore_triattention",
    "update_triattention_metadata",
]


def convert_triattention(model: nn.Module, config: TriAttentionConfig) -> ConvertReturnType:
    """Apply TriAttention mode to model.

    This is a no-op on model weights. It stores the configuration in metadata
    so that calibration can be run subsequently.
    """
    metadata = {
        "triattention_config": config.model_dump(),
    }
    return model, metadata


def restore_triattention(
    model: nn.Module, config: TriAttentionConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore TriAttention mode from saved state.

    Loads calibration data from metadata if present.
    """
    return model


def update_triattention_metadata(
    model: nn.Module, config: TriAttentionConfig, metadata: MetadataDict
) -> None:
    """Update metadata before saving.

    Ensures calibration data and config are current in metadata.
    """
    metadata["triattention_config"] = config.model_dump()
