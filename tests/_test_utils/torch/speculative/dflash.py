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

"""Shared DFlash test config, used by the unit and gpu speculative-decoding tests."""

from copy import deepcopy

from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG

DFLASH_BLOCK_SIZE = 4
DFLASH_NUM_DRAFT_LAYERS = 2


def get_dflash_config(
    block_size: int = DFLASH_BLOCK_SIZE,
    num_layers: int = DFLASH_NUM_DRAFT_LAYERS,
    offline: bool | None = None,
):
    """DFlash config sized for a tiny model: no torch.compile, token 0 as the mask token.

    ``offline`` is only written when set, so callers that don't care keep the default.
    """
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0  # use token 0 as mask for tiny model
    config["dflash_architecture_config"] = {"num_hidden_layers": num_layers}
    if offline is not None:
        config["dflash_offline"] = offline
    return config
