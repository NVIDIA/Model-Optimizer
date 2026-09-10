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

"""PEFT mode descriptors for model optimization."""

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeRegistryCls,
)

from .config import ExportPEFTConfig, PEFTConfig
from .conversion import (
    convert_to_peft_model,
    export_peft_model,
    restore_export_peft_model,
    restore_peft_model,
    update_peft_metadata,
)

PEFTModeRegistry = _ModeRegistryCls("PEFT")


@PEFTModeRegistry.register_mode
class PEFTModeDescriptor(ModeDescriptor):
    """Mode descriptor for PEFT (Parameter-Efficient Fine-Tuning) mode."""

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "peft"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return PEFTConfig

    @property
    def export_mode(self) -> str | None:
        """Specifies the export mode name for this mode."""
        return "export_peft"

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_to_peft_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_peft_model

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the model's state before saving."""
        return update_peft_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_peft_metadata


@PEFTModeRegistry.register_mode
class ExportPEFTModeDescriptor(ModeDescriptor):
    """Mode descriptor for exporting PEFT models."""

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "export_peft"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return ExportPEFTConfig

    @property
    def is_export_mode(self) -> bool:
        """Specifies whether the mode is an export mode."""
        return True

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return export_peft_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_export_peft_model
