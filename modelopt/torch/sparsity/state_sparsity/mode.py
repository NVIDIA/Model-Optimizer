# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DASC state-sparsity mode descriptor."""

from typing import cast

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeRegistryCls,
)

from .config import DASCConfig
from .conversion import convert_dasc_model, restore_dasc_model, update_dasc_metadata

__all__ = ["DASCModeRegistry"]

DASCModeRegistry = _ModeRegistryCls("state_sparsity")


@DASCModeRegistry.register_mode
class DASCModeDescriptor(ModeDescriptor):
    """Describe checkpoint-specific GDN DASC policy calibration."""

    @property
    def name(self) -> str:
        """Return the mode name."""
        return "dasc"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Return the validated DASC configuration class."""
        return DASCConfig

    @property
    def next_prohibited_modes(self) -> set[str]:
        """Route repeat calibration through the replacing public API."""
        return {"dasc"}

    @property
    def convert(self) -> ConvertEntrypoint:
        """Return the DASC calibration entrypoint."""
        return cast("ConvertEntrypoint", convert_dasc_model)

    @property
    def restore(self) -> RestoreEntrypoint:
        """Return the DASC restore entrypoint."""
        return restore_dasc_model

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """Return the metadata refresh entrypoint."""
        return update_dasc_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """Return the metadata refresh entrypoint."""
        return update_dasc_metadata
