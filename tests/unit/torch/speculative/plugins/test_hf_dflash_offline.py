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

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp
from modelopt.recipe.config import ModelOptDFlashRecipe

NUM_BASE_LAYERS = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16


def test_convert_online_keeps_base_layers():
    """Online DFlash (default) keeps the base model layers intact."""
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    mtsp.convert(model, [("dflash", get_dflash_config(offline=False))])

    assert model.dflash_offline is False
    assert "layers" in model._base_model._modules
    assert len(model._base_model.layers) == NUM_BASE_LAYERS


def test_convert_offline_deletes_base_layers():
    """Offline DFlash drops the base model layers to save memory."""
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    # num_orig_hidden_layers records the pre-deletion layer count; users set it before convert.
    model.config.num_orig_hidden_layers = NUM_BASE_LAYERS
    mtsp.convert(model, [("dflash", get_dflash_config(offline=True))])

    assert model.dflash_offline is True
    assert "layers" not in model._base_model._modules


def test_convert_offline_target_layer_ids_from_orig():
    """Offline path selects target layer IDs from num_orig_hidden_layers, not the live base."""
    num_orig = 8
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    model.config.num_orig_hidden_layers = num_orig
    mtsp.convert(model, [("dflash", get_dflash_config(offline=True))])

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


def _loss_and_grads(model, input_ids, features, seed):
    model.zero_grad()
    torch.manual_seed(seed)  # seeds the anchor sampling
    loss = model(input_ids=input_ids, base_model_outputs=features).loss
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.dflash_module.named_parameters()}
    return loss.detach(), grads


def _context_parallel_parity_job(rank, size, gradient_checkpointing):
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS, dtype=torch.float32)
    model.config.num_orig_hidden_layers = NUM_BASE_LAYERS
    mtsp.convert(model, [("dflash", get_dflash_config(offline=True))])
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    generator = torch.Generator().manual_seed(0)
    hidden_size = model.config.hidden_size
    input_ids = torch.randint(1, model.config.vocab_size, (2, SEQ_LEN), generator=generator)
    features = {
        "aux_hidden_states": torch.randn(
            2, SEQ_LEN, NUM_DRAFT_LAYERS * hidden_size, generator=generator
        ),
        "base_model_hidden_states": torch.randn(2, SEQ_LEN, hidden_size, generator=generator),
    }
    expected_loss, expected_grads = _loss_and_grads(model, input_ids, dict(features), seed=0)

    model.enable_context_parallel(dist.group.WORLD)
    model.shard_context(features)
    assert features["aux_hidden_states"].shape[1] == SEQ_LEN // size
    # Ranks seeded apart must still draft the blocks rank 0 samples.
    loss, grads = _loss_and_grads(model, input_ids, features, seed=rank)

    torch.testing.assert_close(loss, expected_loss)
    for name, grad in grads.items():
        dist.all_reduce(grad)  # the trainer averages gradients over the ranks
        torch.testing.assert_close(
            grad / size, expected_grads[name], rtol=1e-4, atol=1e-5, msg=name
        )


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_context_parallel_matches_single_rank(cpu_dist_workers, gradient_checkpointing):
    """Two ranks with half the context each reproduce the one-rank loss and gradients."""
    cpu_dist_workers.run(_context_parallel_parity_job, gradient_checkpointing)


@pytest.mark.parametrize(
    ("setting", "value", "match"),
    [
        ("dflash_offline", False, "online training"),
        ("dflash_use_torch_compile", True, "torch_compile"),
        ("dflash_architecture_config", {"projector_type": "dspark"}, "dspark"),
        ("dflash_architecture_config", {"attention_dropout": 0.1}, "attention_dropout"),
    ],
)
def test_context_parallel_rejects_unsupported_setups(setting, value, match):
    config = get_dflash_config(offline=True)
    if setting == "dflash_architecture_config":
        value = {**config[setting], **value}
    config[setting] = value
    model = get_tiny_llama(num_hidden_layers=NUM_BASE_LAYERS)
    model.config.num_orig_hidden_layers = NUM_BASE_LAYERS
    mtsp.convert(model, [("dflash", config)])

    with pytest.raises(ValueError, match=match):
        model.enable_context_parallel(None)
    assert not model.context_parallel_enabled
