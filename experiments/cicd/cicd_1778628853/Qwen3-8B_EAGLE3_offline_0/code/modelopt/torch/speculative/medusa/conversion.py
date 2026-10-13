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

"""Medusa conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from ..config import MedusaConfig

MedusaDMRegistry = _DMRegistryCls(prefix="Medusa")  # global instance for the registry


def convert_to_medusa_model(model: nn.Module, config: MedusaConfig) -> ConvertReturnType:
    """Convert the model to a medusa model as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in MedusaDMRegistry:
        for cls in MedusaDMRegistry._registry:
            if issubclass(original_cls, cls):
                MedusaDMRegistry.register({original_cls: "base_model_class"})(MedusaDMRegistry[cls])
                break

    medusa_model = MedusaDMRegistry.convert(model)
    medusa_model.modify(
        medusa_num_heads=config.medusa_num_heads, medusa_num_layers=config.medusa_num_layers
    )

    # no metadata, all specified via config.
    metadata = {}

    return medusa_model, metadata


def restore_medusa_model(
    model: nn.Module, config: MedusaConfig, metadata: MetadataDict
) -> nn.Module:
    """Function for restoring a previously convert model to a medusa model."""
    # the metadata should be empty
    assert not metadata, "No metadata expected!"

    return convert_to_medusa_model(model, config)[0]
