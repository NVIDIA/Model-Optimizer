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

"""Eagle conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.speculative.config import eagle3_default_config, kimik2_eagle_default_config

from ..config import EagleConfig

EagleDMRegistry = _DMRegistryCls(prefix="Eagle")  # global instance for the registry


def convert_to_eagle_model(model: nn.Module, config: EagleConfig) -> ConvertReturnType:
    """Convert the model to a eagle model as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in EagleDMRegistry:
        for cls in EagleDMRegistry._registry:
            if issubclass(original_cls, cls):
                EagleDMRegistry.register({original_cls: "base_model_class"})(EagleDMRegistry[cls])
                break

    # merge custom config with default config
    default_arch_config = {
        "llama": eagle3_default_config,
        "kimik2": kimik2_eagle_default_config,
    }[config.eagle_decoder_type]
    custom_config = config.eagle_architecture_config
    config.eagle_architecture_config = {**default_arch_config, **custom_config}

    eagle_model = EagleDMRegistry.convert(model)
    eagle_model.modify(
        eagle_offline=config.eagle_offline,
        eagle_hidden_state_distillation=config.eagle_hidden_state_distillation,
        eagle_self_logit_distillation=config.eagle_self_logit_distillation,
        eagle_freeze_base_model=config.eagle_freeze_base_model,
        eagle_report_acc=config.eagle_report_acc,
        eagle_reuse_base_decoder=config.eagle_reuse_base_decoder,
        eagle_loss_decay_factor=config.eagle_loss_decay_factor,
        eagle_architecture_config=config.eagle_architecture_config,
        eagle_decoder_type=config.eagle_decoder_type,
    )

    # no metadata, all specified via config.
    metadata = {}

    return eagle_model, metadata


def restore_eagle_model(model: nn.Module, config: EagleConfig, metadata: MetadataDict) -> nn.Module:
    """Function for restoring a previously convert model to a eagle model."""
    # the metadata should be empty
    assert not metadata, "No metadata expected!"

    return convert_to_eagle_model(model, config)[0]
