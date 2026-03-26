# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from datasets import Dataset
from transformers import TrainingArguments

from modelopt.torch.opt.plugins.transformers import ModelOptHFTrainer, ModelOptTrainerArguments


def _make_dummy_dataset(seq_len=16, num_samples=8, vocab_size=32):
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})


def _make_training_args(tmp_dir, **overrides):
    defaults = {
        "output_dir": str(tmp_dir),
        "max_steps": 1,
        "per_device_train_batch_size": 2,
        "no_cuda": True,
        "use_cpu": True,
        "report_to": "none",
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _make_trainer(tmp_path, model=None, lr_config=None, trainer_args=None, **args_overrides):
    if model is None:
        model = get_tiny_llama()
    return ModelOptHFTrainer(
        model=model,
        args=_make_training_args(tmp_path, **args_overrides),
        train_dataset=_make_dummy_dataset(),
        trainer_args=trainer_args,
        lr_config=lr_config,
    )


# -- load_lr_config -----------------------------------------------------------


def test_load_lr_config_valid(tmp_path):
    yaml_path = tmp_path / "lr.yaml"
    yaml_path.write_text('"*round_logits*":\n  lr: 0.01\n"*amax*":\n  lr: 0.001\n')
    cfg = ModelOptHFTrainer.load_lr_config(str(yaml_path))
    assert list(cfg.keys()) == ["*round_logits*", "*amax*"]
    assert cfg["*round_logits*"] == {"lr": 0.01}


def test_load_lr_config_invalid_not_dict(tmp_path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        ModelOptHFTrainer.load_lr_config(str(yaml_path))


def test_load_lr_config_invalid_entry(tmp_path):
    yaml_path = tmp_path / "bad2.yaml"
    yaml_path.write_text('"*foo*": 42\n')
    with pytest.raises(ValueError, match="str -> dict"):
        ModelOptHFTrainer.load_lr_config(str(yaml_path))


# -- resolve_lr_config --------------------------------------------------------


def test_resolve_prefers_dict_over_path(tmp_path):
    yaml_path = tmp_path / "lr.yaml"
    yaml_path.write_text('"*a*":\n  lr: 0.1\n')
    direct = {"*b*": {"lr": 0.2}}
    ta = ModelOptTrainerArguments(lr_config=str(yaml_path))
    assert ModelOptHFTrainer._resolve_lr_config(direct, ta) == direct


def test_resolve_falls_back_to_path(tmp_path):
    yaml_path = tmp_path / "lr.yaml"
    yaml_path.write_text('"*a*":\n  lr: 0.1\n')
    ta = ModelOptTrainerArguments(lr_config=str(yaml_path))
    assert ModelOptHFTrainer._resolve_lr_config(None, ta) == {"*a*": {"lr": 0.1}}


def test_resolve_returns_none():
    assert ModelOptHFTrainer._resolve_lr_config(None, ModelOptTrainerArguments()) is None


# -- create_optimizer ---------------------------------------------------------


def test_create_optimizer_no_lr_config(tmp_path):
    """Without lr_config, default HF groups (decay + no-decay)."""
    trainer = _make_trainer(tmp_path)
    trainer.create_optimizer()
    assert trainer.optimizer is not None
    assert len(trainer.optimizer.param_groups) == 2


def test_create_optimizer_with_lr_config(tmp_path):
    """lr_config produces separate param groups with correct LRs."""
    lr_config = {"*lm_head*": {"lr": 0.1}, "*embed*": {"lr": 0.05}}
    trainer = _make_trainer(tmp_path, lr_config=lr_config)
    trainer.create_optimizer()

    groups = trainer.optimizer.param_groups
    assert len(groups) > 2
    lrs = {g["lr"] for g in groups}
    assert 0.1 in lrs
    assert 0.05 in lrs
    assert 1e-3 in lrs  # default for unmatched


def test_create_optimizer_first_match_wins(tmp_path):
    """First matching pattern wins -- second pattern gets no params."""
    lr_config = {"*lm_head.weight": {"lr": 0.1}, "*lm_head*": {"lr": 0.2}}
    trainer = _make_trainer(tmp_path, lr_config=lr_config)
    trainer.create_optimizer()

    lrs = {g["lr"] for g in trainer.optimizer.param_groups}
    assert 0.1 in lrs
    assert 0.2 not in lrs


def test_create_optimizer_custom_weight_decay(tmp_path):
    """lr_config can override weight_decay per pattern."""
    lr_config = {"*lm_head*": {"lr": 0.01, "weight_decay": 0.99}}
    trainer = _make_trainer(tmp_path, lr_config=lr_config)
    trainer.create_optimizer()

    lm_groups = [g for g in trainer.optimizer.param_groups if g["lr"] == 0.01]
    assert len(lm_groups) == 1
    assert lm_groups[0]["weight_decay"] == 0.99


def test_create_optimizer_frozen_params_excluded(tmp_path):
    """Frozen params should not appear in any optimizer group."""
    model = get_tiny_llama()
    for p in model.parameters():
        p.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)

    trainer = _make_trainer(tmp_path, model=model, lr_config={"*lm_head*": {"lr": 0.1}})
    trainer.create_optimizer()

    total_params = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    assert total_params == 1


def test_create_optimizer_via_yaml_path(tmp_path):
    """lr_config loaded from YAML via ModelOptTrainerArguments.lr_config path."""
    yaml_path = tmp_path / "lr.yaml"
    yaml_path.write_text('"*lm_head*":\n  lr: 0.42\n')
    ta = ModelOptTrainerArguments(lr_config=str(yaml_path))

    trainer = _make_trainer(tmp_path, trainer_args=ta)
    trainer.create_optimizer()

    lrs = {g["lr"] for g in trainer.optimizer.param_groups}
    assert 0.42 in lrs
