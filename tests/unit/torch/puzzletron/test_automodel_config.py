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

"""Tests for the puzzletron -> NeMo recipe config translation (no NeMo/GPU)."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.plugins.automodel.config import (
    _align_pipeline_batch_size,
    build_recipe_config,
    build_solution_recipe_config,
)


def _cfg(method="independent", eval_samples=200, micro_batch_size=2):
    return OmegaConf.create(
        {
            "puzzle_dir": "/puzzle",
            "descriptor": "qwen3",
            "nccl_timeout_minutes": 90,
            "pruning": {
                "activations_log_dir": "/puzzle/scores",
                "eval_samples": eval_samples,
                "micro_batch_size": micro_batch_size,
                "activation_hooks_kwargs": {"method": method, "optimize_for": "memory"},
                "automodel": {
                    "force_hf": True,
                    "eval_iters": 50,
                    "parallel": {
                        "tp": 2,
                        "cp": 1,
                        "pp": 1,
                        "ep": 1,
                        "dp_shard": 1,
                        "dp_replicate": 1,
                    },
                },
            },
        }
    )


def test_build_recipe_config_generates_recipe_from_stage_parallelism(monkeypatch):
    cfg = _cfg()
    cfg.pruning.automodel.parallel = {
        "tp": 1,
        "cp": 2,
        "pp": 2,
        "ep": 4,
        "dp_shard": 4,
        "dp_replicate": 2,
        "sequence_parallel": False,
        "pipeline_schedule": "1f1b",
    }
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config._inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_pipeline_config",
        lambda *args, **kwargs: None,
    )

    recipe = build_recipe_config(cfg)

    assert recipe["distributed"] == {
        "dp_size": 8,
        "dp_replicate_size": 2,
        "tp_size": 1,
        "cp_size": 2,
        "ep_size": 4,
        "pp_size": 2,
        "sequence_parallel": False,
        "pipeline": {
            "pp_schedule": "1f1b",
            "pp_microbatch_size": 1,
            "pp_batch_size": 2,
        },
    }
    assert recipe["dataset"]["_target_"].endswith("make_dummy_dataset")
    assert recipe["distributed_config"]["activation_checkpointing"] is True
    assert recipe["optimizer"]["lr"] == 0.0


def test_build_recipe_config_rejects_pure_ddp_replication():
    cfg = _cfg()
    cfg.pruning.automodel.parallel.dp_shard = 1
    cfg.pruning.automodel.parallel.dp_replicate = 2

    with pytest.raises(ValueError, match="dp_shard greater than one"):
        build_recipe_config(cfg)


@pytest.mark.parametrize("legacy_key", ["recipe", "recipe_path"])
def test_build_recipe_config_rejects_removed_recipe_inputs(legacy_key):
    cfg = _cfg()
    cfg.pruning.automodel[legacy_key] = {} if legacy_key == "recipe" else "/old.yaml"

    with pytest.raises(ValueError, match="automodel.parallel"):
        build_recipe_config(cfg)


def test_solution_recipe_uses_inferred_runtime_descriptor(monkeypatch):
    cfg = OmegaConf.create(
        {
            "puzzle_dir": "/puzzle",
            "_runtime": {"descriptor": "qwen3"},
            "pruning": {"block_size": 32, "micro_batch_size": 1},
            "scoring": {
                "block_size": 32,
                "micro_batch_size": 1,
                "automodel": {
                    "force_hf": False,
                    "parallel": {
                        "tp": 1,
                        "cp": 1,
                        "pp": 1,
                        "ep": 1,
                        "dp_shard": 1,
                        "dp_replicate": 1,
                    },
                },
            },
        }
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config._inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config.inject_descriptor_pipeline_config",
        lambda *args, **kwargs: None,
    )

    recipe = build_solution_recipe_config(cfg, "/checkpoint")

    assert recipe["model"]["anymodel_descriptor"] == "qwen3"


def test_build_recipe_config_uses_torch_linears_for_native_dtensor_tp(monkeypatch):
    """PyTorch Row/ColwiseParallel cannot shard Transformer Engine Linear modules."""

    cfg = _cfg()
    cfg.pruning.automodel.force_hf = False
    cfg.pruning.automodel.parallel.cp = 2

    class Descriptor:
        @staticmethod
        def automodel_model_kwargs(config, *, distributed):
            return {"backend": {"attn": "te"}}

        @staticmethod
        def automodel_tp_linear_backend(config):
            return "torch"

        @staticmethod
        def puzzletron_capabilities(config):
            return SimpleNamespace(native_automodel_supported=True)

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config._registered_descriptor",
        lambda name: Descriptor,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config._load_model_config_for_descriptor",
        lambda *args, **kwargs: object(),
    )

    recipe = build_recipe_config(cfg)

    assert recipe["model"]["backend"] == {"attn": "te", "linear": "torch"}


def test_pipeline_batch_alignment_splits_global_batch_by_dp_after_ep_overlay():
    recipe = {
        "distributed": {
            "dp_size": 32,
            "ep_size": 4,
            "pp_size": 2,
            "pipeline": {},
        },
        "step_scheduler": {},
    }

    _align_pipeline_batch_size(recipe, micro_batch_size=8)

    assert recipe["distributed"]["pipeline"]["pp_microbatch_size"] == 1
    assert recipe["distributed"]["pipeline"]["pp_batch_size"] == 2
    assert recipe["step_scheduler"]["local_batch_size"] == 2
    assert recipe["step_scheduler"]["global_batch_size"] == 64


def test_build_recipe_config_missing_descriptor_raises():
    cfg = _cfg()
    cfg.descriptor = None  # and no descriptor under recipe.model
    with pytest.raises(ValueError, match="descriptor"):
        build_recipe_config(cfg)


def test_build_recipe_config_selects_native_vlm_and_neat_packing(monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config._inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    cfg = _cfg()
    cfg.pruning.automodel.force_hf = False
    cfg.pruning.automodel.use_puzzletron_dataloader = False
    cfg.data = {
        "path": "/puzzle/data/intersyn-16",
        "modality": "multimodal",
        "layout": "packed_varlen",
        "max_sample_length": 1536,
        "packing": {
            "pack_size": 2048,
            "packing_ratio": 0.9,
            "drop_long_samples": True,
        },
    }

    recipe = build_recipe_config(cfg)

    assert recipe["model"]["_target_"] == (
        "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
    )
    assert recipe["dataset"]["_target_"] == (
        "modelopt.torch.puzzletron.dataset.load_materialized_conversation_dataset"
    )
    assert recipe["dataset"]["path_or_dataset"] == "/puzzle/data/intersyn-16"
    assert recipe["dataset"]["max_length"] == 1536
    assert recipe["packed_sequence"]["pack_size"] == 2048
    assert recipe["packed_sequence"]["packing_ratio"] == 0.9
    assert recipe["packed_sequence"]["attn_implementation"] == "flash_attention_2"
    assert recipe["packed_sequence"]["max_packs"] == 200
    assert "collate_fn" not in recipe["dataloader"]


def test_build_recipe_config_uses_native_vlm_padded_collator(monkeypatch):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.plugins.automodel.config._inject_descriptor_model_kwargs",
        lambda *args, **kwargs: None,
    )
    cfg = _cfg()
    cfg.pruning.automodel.force_hf = False
    cfg.pruning.automodel.use_puzzletron_dataloader = False
    cfg.data = {
        "path": "/puzzle/data/vlm-smoke",
        "modality": "multimodal",
        "layout": "padded_varlen",
        "max_sample_length": 512,
    }

    recipe = build_recipe_config(cfg)

    assert recipe["dataset"]["path_or_dataset"] == "/puzzle/data/vlm-smoke"
    assert "collate_fn" not in recipe["dataloader"]
