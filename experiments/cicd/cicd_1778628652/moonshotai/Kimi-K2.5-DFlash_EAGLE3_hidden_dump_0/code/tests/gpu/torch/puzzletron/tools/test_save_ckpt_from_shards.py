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

"""Tests for save_checkpoint_from_shards in checkpoint_utils_hf."""

import json
from functools import partial

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file as safe_load_file

from modelopt.torch.puzzletron.anymodel.models.llama.llama_model_descriptor import (
    LlamaModelDescriptor,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_SUBBLOCKS_DIR_NAME,
    save_checkpoint_from_shards,
)


class TestSaveCheckpointFromShardsSingleProcess:
    """Tests that run without torch.distributed (world_size=1 path)."""

    def test_creates_config_index_and_subblocks(self, tmp_path):
        model = get_tiny_llama()
        expected_keys = set(model.state_dict().keys())
        save_checkpoint_from_shards(model, tmp_path, LlamaModelDescriptor)

        # test safetensors index file exists and contains weight map
        index_path = tmp_path / SAFE_WEIGHTS_INDEX_NAME
        assert index_path.exists(), "safetensors index file was not written"
        index = json.loads(index_path.read_text())
        assert "weight_map" in index
        assert set(index["weight_map"].keys()) == expected_keys

        # test subblocks directory exists and contains shard files
        subblocks_dir = tmp_path / SAFETENSORS_SUBBLOCKS_DIR_NAME
        assert subblocks_dir.is_dir(), "subblocks directory was not created"
        assert len(list(subblocks_dir.glob("*.safetensors"))) > 0, (
            "no safetensors shard files were saved"
        )

        # test config.json saved
        config_path = tmp_path / "config.json"
        assert config_path.exists(), "config.json was not saved"
        cfg = json.loads(config_path.read_text())
        assert cfg["num_hidden_layers"] == get_tiny_llama().config.num_hidden_layers

        # test subblock filenames follow descriptor groups
        filenames = set(index["weight_map"].values())
        expected_substrings = {"embeddings", "lm_head", "block_0_ffn", "block_0_attention"}
        for substr in expected_substrings:
            assert any(substr in f for f in filenames), f"no shard filename contains '{substr}'"

    def test_tie_word_embeddings_excluded(self, tmp_path):
        model = get_tiny_llama(tie_word_embeddings=True)
        save_checkpoint_from_shards(model, tmp_path, LlamaModelDescriptor)

        index = json.loads((tmp_path / SAFE_WEIGHTS_INDEX_NAME).read_text())
        assert "lm_head.weight" not in index["weight_map"]

        reloaded_sd = {}
        for shard in (tmp_path / SAFETENSORS_SUBBLOCKS_DIR_NAME).glob("*.safetensors"):
            reloaded_sd.update(safe_load_file(str(shard)))
        assert "lm_head.weight" not in reloaded_sd

    def test_saved_weights_match_original(self, tmp_path):
        model = get_tiny_llama()
        original_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        save_checkpoint_from_shards(model, tmp_path, LlamaModelDescriptor)

        reloaded_sd = {}
        for shard in (tmp_path / SAFETENSORS_SUBBLOCKS_DIR_NAME).glob("*.safetensors"):
            reloaded_sd.update(safe_load_file(str(shard)))

        assert set(reloaded_sd.keys()) == set(original_sd.keys())
        for key in original_sd:
            torch.testing.assert_close(reloaded_sd[key], original_sd[key])


def _distributed_save_worker(rank, world_size, checkpoint_dir):
    """Worker that shards a model's state dict across ranks and saves."""
    model = get_tiny_llama()
    full_sd = model.state_dict()
    keys = sorted(full_sd.keys())
    per_rank = len(keys) // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else len(keys)
    shard_keys = keys[start:end]

    # Zero out keys not owned by this rank so gather reconstructs the full dict.
    for k in keys:
        if k not in shard_keys:
            full_sd[k] = torch.zeros_like(full_sd[k])

    model.load_state_dict(full_sd)
    save_checkpoint_from_shards(model, checkpoint_dir, LlamaModelDescriptor)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need >=2 GPUs for multi-rank test")
class TestSaveCheckpointFromShardsMultiProcess:
    """Tests that exercise the distributed gather path (world_size > 1)."""

    def test_distributed_save_creates_valid_checkpoint(self, tmp_path):
        spawn_multiprocess_job(2, partial(_distributed_save_worker, checkpoint_dir=tmp_path))

        index_path = tmp_path / SAFE_WEIGHTS_INDEX_NAME
        assert index_path.exists()
        index = json.loads(index_path.read_text())

        model = get_tiny_llama()
        expected_keys = set(model.state_dict().keys())
        assert set(index["weight_map"].keys()) == expected_keys

        shard_files = list((tmp_path / SAFETENSORS_SUBBLOCKS_DIR_NAME).glob("*.safetensors"))
        assert len(shard_files) > 0
