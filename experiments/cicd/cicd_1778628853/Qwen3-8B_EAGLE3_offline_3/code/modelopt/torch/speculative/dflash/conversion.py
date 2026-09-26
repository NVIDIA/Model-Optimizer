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


def convert_to_dflash_model(model: nn.Module, config: DFlashConfig) -> ConvertReturnType:
    """Convert the model to a DFlash model as per `config`."""
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in DFlashDMRegistry:
        for cls in DFlashDMRegistry._registry:
            if issubclass(original_cls, cls):
                DFlashDMRegistry.register({original_cls: "base_model_class"})(DFlashDMRegistry[cls])
                break

    # merge custom config with default config (lazy import to avoid circular)
    from .default_config import default_dflash_config

    custom_config = config.dflash_architecture_config
    config.dflash_architecture_config = {**default_dflash_config, **custom_config}

    dflash_model = DFlashDMRegistry.convert(model)
    dflash_model.modify(config)

    metadata = {}
    return dflash_model, metadata


def restore_dflash_model(
    model: nn.Module, config: DFlashConfig, metadata: MetadataDict
) -> nn.Module:
    """Function for restoring a previously converted model to a DFlash model."""
    assert not metadata, "No metadata expected!"
    return convert_to_dflash_model(model, config)[0]
