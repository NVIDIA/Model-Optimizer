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

"""External draft conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from ..config import ExternalDraftConfig

ExternalDraftDMRegistry = _DMRegistryCls(prefix="ExternalDraft")  # global instance for the registry

__all__ = [
    "ExternalDraftDMRegistry",
    "convert_to_external_draft_model",
    "restore_external_draft_model",
]


def convert_to_external_draft_model(
    model: nn.Module, config: ExternalDraftConfig
) -> ConvertReturnType:
    """Convert a pretrained causal LM into a trainable external draft as per `config`.

    Note that, unlike the other speculative modes, ``model`` here is the *draft*
    rather than the base model. Nothing is grafted onto a target; the draft keeps
    its own embeddings and lm_head.
    """
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in ExternalDraftDMRegistry:
        for cls in ExternalDraftDMRegistry._registry:
            if issubclass(original_cls, cls):
                ExternalDraftDMRegistry.register({original_cls: "draft_model_class"})(
                    ExternalDraftDMRegistry[cls]
                )
                break

    external_draft_model = ExternalDraftDMRegistry.convert(model)
    external_draft_model.modify(config)

    # no metadata, all specified via config.
    metadata = {}

    return external_draft_model, metadata


def restore_external_draft_model(
    model: nn.Module, config: ExternalDraftConfig, metadata: MetadataDict
) -> nn.Module:
    """Function for restoring a previously converted model to an external draft model."""
    # the metadata should be empty
    assert not metadata, "No metadata expected!"

    return convert_to_external_draft_model(model, config)[0]
