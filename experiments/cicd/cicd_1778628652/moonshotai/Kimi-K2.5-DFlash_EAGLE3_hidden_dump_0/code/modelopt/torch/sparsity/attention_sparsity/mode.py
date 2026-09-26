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

"""Sparse attention mode descriptor for ModelOpt."""

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeRegistryCls,
)

from .config import SparseAttentionConfig
from .conversion import (
    convert_to_sparse_attention_model,
    restore_sparse_attention_model,
    update_sparse_attention_metadata,
)

# Create registry for sparse attention modes
SparseAttentionModeRegistry = _ModeRegistryCls("sparse_attention")


@SparseAttentionModeRegistry.register_mode
class SparseAttentionModeDescriptor(ModeDescriptor):
    """Mode descriptor for sparse attention optimization.

    This mode enables various sparse attention methods to reduce
    computational complexity and memory usage in transformer models.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "sparse_attention"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return SparseAttentionConfig

    @property
    def next_prohibited_modes(self) -> set[str] | None:
        """Modes that should not be applied after this mode."""
        # Can work with quantization but not with weight sparsity
        return {"sparsity"}

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export_sparse_attention"

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_to_sparse_attention_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_sparse_attention_model

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the model's state before saving."""
        return update_sparse_attention_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the model's state before new mode."""
        return update_sparse_attention_metadata
