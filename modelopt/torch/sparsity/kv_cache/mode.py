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

"""Mode registration for KV cache sparsity."""

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeRegistryCls,
)

from .config import TriAttentionConfig
from .conversion import convert_triattention, restore_triattention, update_triattention_metadata

KVCacheSparsityRegistry = _ModeRegistryCls("kv_cache_sparsity")


@KVCacheSparsityRegistry.register_mode
class TriAttentionModeDescriptor(ModeDescriptor):
    """Mode descriptor for TriAttention KV cache sparsity.

    TriAttention is a calibration-only mode: convert is a no-op on model weights,
    calibration computes per-head frequency statistics, and the results are stored
    in metadata for export to serving engines.
    """

    @property
    def name(self) -> str:
        """Return the mode name."""
        return "triattention"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Return the configuration class."""
        return TriAttentionConfig

    @property
    def convert(self) -> ConvertEntrypoint:
        """Return the convert entrypoint."""
        return convert_triattention

    @property
    def restore(self) -> RestoreEntrypoint:
        """Return the restore entrypoint."""
        return restore_triattention

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """Return the update-for-save entrypoint."""
        return update_triattention_metadata
