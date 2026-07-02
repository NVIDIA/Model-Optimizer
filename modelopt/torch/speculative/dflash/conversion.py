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

"""DFlash conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from ..config import DFlashConfig

DFlashDMRegistry = _DMRegistryCls(prefix="DFlash")  # global instance for the registry
# Domino reuses the dflash mode/config/recipe but converts the base model to a
# DFlash module augmented with a causal correction head. It is selected via
# ``dflash_architecture_config.projector_type == "domino"`` and lives in its own
# registry so its wrapper (HFDominoModel) does not overwrite HFDFlashModel.
DominoDMRegistry = _DMRegistryCls(prefix="Domino")


def convert_to_dflash_model(model: nn.Module, config: DFlashConfig) -> ConvertReturnType:
    """Convert the model to a DFlash (or Domino) model as per `config`."""
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # merge custom config with default config (lazy import to avoid circular)
    from .default_config import default_dflash_config

    custom_config = config.dflash_architecture_config
    config.dflash_architecture_config = {**default_dflash_config, **custom_config}

    # Route to the Domino registry when the architecture asks for the Domino head.
    projector_type = config.dflash_architecture_config.get("projector_type")
    if projector_type == "domino":
        registry = DominoDMRegistry
    elif projector_type in (None, "dflash"):
        registry = DFlashDMRegistry
    else:
        raise ValueError(
            f"Unsupported dflash_architecture_config.projector_type: {projector_type!r}. "
            "Expected 'dflash' (default) or 'domino'."
        )

    original_cls = type(model)
    if original_cls not in registry:
        for cls in registry._registry:
            if issubclass(original_cls, cls):
                registry.register({original_cls: "base_model_class"})(registry[cls])
                break

    dflash_model = registry.convert(model)
    dflash_model.modify(config)

    metadata = {}
    return dflash_model, metadata


def restore_dflash_model(
    model: nn.Module, config: DFlashConfig, metadata: MetadataDict
) -> nn.Module:
    """Function for restoring a previously converted model to a DFlash model."""
    assert not metadata, "No metadata expected!"
    return convert_to_dflash_model(model, config)[0]
