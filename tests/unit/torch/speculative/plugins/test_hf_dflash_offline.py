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

"""CPU unit tests for DFlash offline training support."""

from copy import deepcopy

from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp
from modelopt.recipe.config import ModelOptDFlashRecipe
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG

NUM_BASE_LAYERS = 4
NUM_DRAFT_LAYERS = 2


def _get_dflash_config(offline=False):
    """Build a minimal DFlash config dict for mtsp.convert."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_offline"] = offline
    config["dflash_block_size"] = 4
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0
    config["dflash_architecture_config"] = {"num_hidden_layers": NUM_DRAFT_LAYERS}
    return config


def test_convert_online_keeps_base_layers():
    """Online DFlash (default) keeps the base model layers intact."""
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    mtsp.convert(model, [("dflash", _get_dflash_config(offline=False))])

    assert model.dflash_offline is False
    assert "layers" in model._base_model._modules
    assert len(model._base_model.layers) == NUM_BASE_LAYERS


def test_convert_offline_deletes_base_layers():
    """Offline DFlash drops the base model layers to save memory."""
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    # num_orig_hidden_layers records the pre-deletion layer count; users set it before convert.
    model.config.num_orig_hidden_layers = NUM_BASE_LAYERS
    mtsp.convert(model, [("dflash", _get_dflash_config(offline=True))])

    assert model.dflash_offline is True
    assert "layers" not in model._base_model._modules


def test_convert_offline_target_layer_ids_from_orig():
    """Offline path selects target layer IDs from num_orig_hidden_layers, not the live base."""
    num_orig = 8
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    model.config.num_orig_hidden_layers = num_orig
    mtsp.convert(model, [("dflash", _get_dflash_config(offline=True))])

    assert len(model.target_layer_ids) == NUM_DRAFT_LAYERS
    # With num_orig=8, build_target_layer_ids(8, 2) spans beyond the 4 live base layers —
    # proves offline path read num_orig_hidden_layers rather than num_hidden_layers.
    assert max(model.target_layer_ids) >= NUM_BASE_LAYERS
    assert all(0 <= lid < num_orig for lid in model.target_layer_ids)


def test_dflash_recipe_derives_offline_from_data():
    """ModelOptDFlashRecipe._derive_dflash_offline flips dflash_offline based on data.offline_data_path."""
    dflash_section = {"dflash_mask_token_id": 0}

    # offline_data_path set → offline=True
    recipe = ModelOptDFlashRecipe(data={"offline_data_path": "/fake/path"}, dflash=dflash_section)
    assert recipe.dflash.dflash_offline is True

    # offline_data_path absent → offline=False
    recipe = ModelOptDFlashRecipe(dflash=dflash_section)
    assert recipe.dflash.dflash_offline is False
