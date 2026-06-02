# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for get_distributed_modules_ownership in bypass_utils.py."""

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.bypass_distillation.bypass_utils import (
    get_bypass_config_fingerprint,
    get_distributed_modules_ownership,
    get_pipeline_ownership_context,
    set_experiment_id,
)


@pytest.mark.parametrize(
    ("module_count", "world_size", "expected_ownership"),
    [
        (4, 1, [0, 0, 0, 0]),
        (4, 2, [0, 0, 1, 1]),
        (3, 2, [0, 0, 1]),
        (7, 3, [0, 0, 0, 1, 1, 2, 2]),
        (1, 2, [0]),
    ],
)
def test_distributed_modules_ownership(module_count, world_size, expected_ownership):
    assert (
        get_distributed_modules_ownership(module_count=module_count, world_size=world_size)
        == expected_ownership
    )


def test_pipeline_ownership_context_returns_neighbors():
    ownership = [0, 0, 1, 1, 2]

    assert get_pipeline_ownership_context(ownership, rank=0) == {
        "owned_indices": [0, 1],
        "owned_index_set": {0, 1},
        "prev_rank": None,
        "next_rank": 1,
    }
    assert get_pipeline_ownership_context(ownership, rank=1) == {
        "owned_indices": [2, 3],
        "owned_index_set": {2, 3},
        "prev_rank": 0,
        "next_rank": 2,
    }
    assert get_pipeline_ownership_context(ownership, rank=2) == {
        "owned_indices": [4],
        "owned_index_set": {4},
        "prev_rank": 1,
        "next_rank": None,
    }


def test_pipeline_ownership_context_rejects_idle_rank():
    with pytest.raises(RuntimeError, match="owns no modules"):
        get_pipeline_ownership_context([0, 0, 1], rank=2)


def _experiment_cfg(keys_to_learn):
    return OmegaConf.create(
        {
            "descriptor": "test_descriptor",
            "teacher_dir": "/tmp/teacher_a",
            "dataset_path": "/tmp/dataset_a",
            "bypass": {
                "experiment_id": None,
                "dtype": "bf16",
                "seed": 42,
                "data": {
                    "block_size": 64,
                    "data_column": "text",
                    "fim_rate": 0,
                    "fim_spm_rate": 0,
                    "bos_rate": 1.0,
                    "source_datasets_to_discard": [],
                    "load_from_disk": True,
                    "keep_in_memory": False,
                    "shuffle_train_data_seed": 123,
                    "val_dataset_name": "valid",
                    "max_eval_samples": 4,
                    "eval_samples_per_process": None,
                },
                "training": {
                    "learning_rate": 1e-4,
                    "training_tokens": 1024,
                    "micro_batch_size": 1,
                    "grad_accumulation_steps": 1,
                    "weight_decay": 0.1,
                    "decay_lr": True,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "grad_clip": 1.0,
                    "grad_clip_type": "norm",
                    "warmup_ratio": 0.05,
                    "min_lr_factor": 1e-5,
                },
                "model": {
                    "student_weights_dtype": "bf16",
                    "model_config_overrides": {
                        "attention": [{"num_key_value_heads": 1, "no_op": None}]
                    },
                },
                "model_factory": {
                    "factory": "bypass_factory_fn",
                    "block_loss_func": "normalized_mse_loss",
                    "gqa_init_mode": "AverageKV",
                    "mlp_init_mode": "Truncate",
                    "mlp_init_config": {"activations_log_dir": None},
                    "linear_init_mode": "FromTeacher",
                    "submodule_for_loss_calculation": None,
                    "keys_to_learn": keys_to_learn,
                },
                "disable_validation": False,
                "save_best_ckpt": True,
                "realize_best_or_latest": "best",
            },
        }
    )


def test_experiment_id_includes_learning_target_and_fingerprint():
    attention_cfg = _experiment_cfg("subblock_attention")
    ffn_cfg = _experiment_cfg("subblock_ffn")

    set_experiment_id(attention_cfg)
    set_experiment_id(ffn_cfg)

    assert attention_cfg.bypass.experiment_id.startswith("bypass_heads_1_attention_")
    assert ffn_cfg.bypass.experiment_id.startswith("bypass_heads_1_ffn_")
    assert attention_cfg.bypass.experiment_id != ffn_cfg.bypass.experiment_id


def test_experiment_id_falls_back_when_no_architecture_parts_exist():
    cfg = _experiment_cfg("entire_block")
    cfg.bypass.model.model_config_overrides = {}

    set_experiment_id(cfg)

    assert cfg.bypass.experiment_id.startswith("bypass_custom_")
    assert cfg.bypass.experiment_id != "bypass_None"


def test_config_fingerprint_changes_with_dataset_path():
    cfg = _experiment_cfg("subblock_attention")
    original = get_bypass_config_fingerprint(cfg)
    cfg.dataset_path = "/tmp/dataset_b"
    assert get_bypass_config_fingerprint(cfg) != original


def test_config_fingerprint_changes_with_shuffle_seed():
    cfg = _experiment_cfg("subblock_attention")
    original = get_bypass_config_fingerprint(cfg)
    cfg.bypass.data.shuffle_train_data_seed = 456
    assert get_bypass_config_fingerprint(cfg) != original


def test_config_fingerprint_changes_with_teacher_dir():
    cfg = _experiment_cfg("subblock_attention")
    original = get_bypass_config_fingerprint(cfg)
    cfg.teacher_dir = "/tmp/teacher_b"
    assert get_bypass_config_fingerprint(cfg) != original


def test_config_fingerprint_changes_with_descriptor():
    cfg = _experiment_cfg("subblock_attention")
    original = get_bypass_config_fingerprint(cfg)
    cfg.descriptor = "other_descriptor"
    assert get_bypass_config_fingerprint(cfg) != original


def test_config_fingerprint_canonicalizes_single_keys_to_learn():
    cfg_a = _experiment_cfg("entire_block")
    cfg_b = _experiment_cfg(["entire_block"])

    assert get_bypass_config_fingerprint(cfg_a) == get_bypass_config_fingerprint(cfg_b)


def test_config_fingerprint_canonicalizes_keys_to_learn_order():
    cfg_a = _experiment_cfg(["subblock_ffn", "subblock_attention"])
    cfg_b = _experiment_cfg(["subblock_attention", "subblock_ffn"])

    assert get_bypass_config_fingerprint(cfg_a) == get_bypass_config_fingerprint(cfg_b)


def test_experiment_id_does_not_change_with_dataset_path():
    cfg_a = _experiment_cfg("subblock_attention")
    cfg_b = _experiment_cfg("subblock_attention")
    cfg_b.dataset_path = "/tmp/dataset_b"
    set_experiment_id(cfg_a)
    set_experiment_id(cfg_b)
    assert cfg_a.bypass.experiment_id == cfg_b.bypass.experiment_id


def test_experiment_id_changes_with_teacher_source():
    cfg_a = _experiment_cfg("subblock_attention")
    cfg_b = _experiment_cfg("subblock_attention")
    cfg_b.teacher_dir = "/tmp/teacher_b"
    set_experiment_id(cfg_a)
    set_experiment_id(cfg_b)
    assert cfg_a.bypass.experiment_id != cfg_b.bypass.experiment_id


def test_experiment_id_canonicalizes_keys_to_learn_order():
    cfg_a = _experiment_cfg(["subblock_ffn", "subblock_attention"])
    cfg_b = _experiment_cfg(["subblock_attention", "subblock_ffn"])
    set_experiment_id(cfg_a)
    set_experiment_id(cfg_b)
    assert cfg_a.bypass.experiment_id == cfg_b.bypass.experiment_id
