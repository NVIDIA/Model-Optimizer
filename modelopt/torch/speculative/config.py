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

"""Configurations for speculative decoding modes."""

from copy import deepcopy

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

from .eagle.default_config import default_eagle_config, default_kimik2_eagle_config

kimik2_eagle_default_config = deepcopy(default_kimik2_eagle_config)

eagle3_default_config = deepcopy(default_eagle_config)
eagle_mtp_default_config = deepcopy(default_eagle_config)

eagle3_default_config.update({"use_aux_hidden_state": True, "use_last_layernorm": True})
eagle_mtp_default_config.update({"use_last_layernorm": True, "use_mtp_layernorm": True})

EAGLE1_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_architecture_config": deepcopy(default_eagle_config),
    },
}

EAGLE3_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_architecture_config": eagle3_default_config,
    },
}

EAGLE_MTP_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_reuse_base_decoder": True,
        "eagle_architecture_config": eagle_mtp_default_config,
    },
}


class MedusaConfig(ModeloptBaseConfig):
    """Medusa config."""

    medusa_num_heads: int = ModeloptField(
        default=2,
        description=("The number of medusa heads added to the model."),
    )

    medusa_num_layers: int = ModeloptField(
        default=1,
        description=("The number of ResBlocks used in medusa head."),
    )


class EagleConfig(ModeloptBaseConfig):
    """Eagle config."""

    eagle_offline: bool = ModeloptField(
        default=False, description=("Whether to use detached Eagle.")
    )

    eagle_hidden_state_distillation: bool = ModeloptField(
        default=False, description=("Whether to use feature hidden states distillation.")
    )

    eagle_self_logit_distillation: bool = ModeloptField(
        default=True, description=("Whether to use logit distillation.")
    )

    eagle_freeze_base_model: bool = ModeloptField(
        default=True, description=("Whether to freeze base model during eagle module training.")
    )

    eagle_report_acc: bool = ModeloptField(
        default=True, description=("Whether to report eval accuracy.")
    )

    eagle_reuse_base_decoder: bool = ModeloptField(
        default=False, description=("Whether to reuse base model decoder in eagle module.")
    )

    eagle_loss_decay_factor: float = ModeloptField(
        default=0.9, description=("The decay factor for multiple eagle_loss.")
    )

    eagle_architecture_config: dict = ModeloptField(
        default={}, description=("The config for eagle module architecture.")
    )

    eagle_decoder_type: str = ModeloptField(
        default="llama",
        description=("The class of eagle decoder to use. Available options: llama, kimik2"),
    )
