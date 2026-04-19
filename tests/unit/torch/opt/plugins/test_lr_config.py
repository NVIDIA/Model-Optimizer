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

"""Unit tests for ModelOptHFTrainer lr_config (per-parameter optimizer kwargs)."""

import pytest
import torch
import yaml
from torch import nn
from transformers import TrainingArguments

from modelopt.torch.opt.plugins.transformers import ModelOptHFTrainer, ModelOptTrainerArguments


class TinyModel(nn.Module):
    """Minimal model with named submodules to exercise fnmatch patterns."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.self_attn = nn.Linear(16, 16)
        self.mlp = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed_tokens(input_ids)
        x = self.self_attn(x)
        x = self.mlp(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type("Out", (), {"loss": loss, "logits": logits})()


@pytest.fixture
def dummy_dataset():
    """Tiny dataset for trainer initialization."""
    return [
        {"input_ids": torch.randint(0, 32, (8,)), "labels": torch.randint(0, 32, (8,))}
        for _ in range(4)
    ]


def _write_lr_config(tmp_path, cfg: dict) -> str:
    path = tmp_path / "lr_config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def _make_trainer(tmp_path, dummy_dataset, lr_config_dict=None, lr_config_path=None):
    """Build a ModelOptHFTrainer with the given lr_config."""
    training_args = TrainingArguments(
        output_dir=str(tmp_path / "output"),
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        no_cuda=True,
        report_to="none",
        use_cpu=True,
    )
    trainer_args = ModelOptTrainerArguments()
    if lr_config_path is not None:
        trainer_args.lr_config = lr_config_path

    trainer = ModelOptHFTrainer(
        model=TinyModel(),
        args=training_args,
        trainer_args=trainer_args,
        lr_config=lr_config_dict,
        train_dataset=dummy_dataset,
    )
    return trainer


class TestLoadLrConfig:
    def test_load_basic(self, tmp_path):
        cfg = {"*lm_head*": {"lr": 1e-5}, "*mlp*": {"lr": 5e-5}}
        path = _write_lr_config(tmp_path, cfg)
        loaded = ModelOptHFTrainer.load_lr_config(path)
        assert loaded == cfg

    def test_load_with_weight_decay_and_betas(self, tmp_path):
        cfg = {
            "*self_attn*": {"lr": 5e-5, "betas": [0.9, 0.95]},
            "*mlp*": {"lr": 5e-5, "weight_decay": 0.05},
            "*embed_tokens*": {"lr": 1e-6, "eps": 1e-7},
        }
        path = _write_lr_config(tmp_path, cfg)
        loaded = ModelOptHFTrainer.load_lr_config(path)
        assert loaded["*self_attn*"]["betas"] == [0.9, 0.95]
        assert loaded["*mlp*"]["weight_decay"] == 0.05
        assert loaded["*embed_tokens*"]["eps"] == 1e-7

    def test_load_invalid_not_dict(self, tmp_path):
        path = tmp_path / "lr_config.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            ModelOptHFTrainer.load_lr_config(str(path))

    def test_load_invalid_entry(self, tmp_path):
        path = tmp_path / "lr_config.yaml"
        path.write_text('"*lm_head*": 0.001\n')
        with pytest.raises(ValueError, match="str -> dict"):
            ModelOptHFTrainer.load_lr_config(str(path))


class TestCreateOptimizerWithLrConfig:
    def test_lr_applied_per_group(self, tmp_path, dummy_dataset):
        lr_config = {
            "*lm_head*": {"lr": 1e-5},
            "*self_attn*": {"lr": 2e-5},
        }
        trainer = _make_trainer(tmp_path, dummy_dataset, lr_config_dict=lr_config)
        trainer.create_optimizer()

        groups = trainer.optimizer.param_groups
        lrs = {g["lr"] for g in groups}
        assert 1e-5 in lrs
        assert 2e-5 in lrs

    def test_weight_decay_override(self, tmp_path, dummy_dataset):
        lr_config = {
            "*lm_head*": {"lr": 1e-5, "weight_decay": 0.0},
            "*mlp*": {"lr": 5e-5, "weight_decay": 0.05},
        }
        trainer = _make_trainer(tmp_path, dummy_dataset, lr_config_dict=lr_config)
        trainer.create_optimizer()

        for group in trainer.optimizer.param_groups:
            if group["lr"] == 1e-5:
                assert group["weight_decay"] == 0.0
            elif group["lr"] == 5e-5:
                assert group["weight_decay"] == 0.05

    def test_betas_override(self, tmp_path, dummy_dataset):
        lr_config = {
            "*self_attn*": {"lr": 5e-5, "betas": [0.9, 0.95]},
        }
        trainer = _make_trainer(tmp_path, dummy_dataset, lr_config_dict=lr_config)
        trainer.create_optimizer()

        found = False
        for group in trainer.optimizer.param_groups:
            if group["lr"] == 5e-5:
                assert group["betas"] == (0.9, 0.95) or group["betas"] == [0.9, 0.95]
                found = True
        assert found, "No group found with lr=5e-5"

    def test_eps_override(self, tmp_path, dummy_dataset):
        lr_config = {
            "*embed_tokens*": {"lr": 1e-6, "eps": 1e-7},
        }
        trainer = _make_trainer(tmp_path, dummy_dataset, lr_config_dict=lr_config)
        trainer.create_optimizer()

        found = False
        for group in trainer.optimizer.param_groups:
            if group["lr"] == 1e-6:
                assert group["eps"] == 1e-7
                found = True
        assert found, "No group found with lr=1e-6"

    def test_lr_config_from_yaml_path(self, tmp_path, dummy_dataset):
        cfg = {
            "*lm_head*": {"lr": 1e-5, "weight_decay": 0.0},
            "*self_attn*": {"lr": 2e-5, "betas": [0.9, 0.95]},
        }
        path = _write_lr_config(tmp_path, cfg)
        trainer = _make_trainer(tmp_path, dummy_dataset, lr_config_path=path)
        trainer.create_optimizer()

        lrs = {g["lr"] for g in trainer.optimizer.param_groups}
        assert 1e-5 in lrs
        assert 2e-5 in lrs

    def test_unmatched_params_use_global_lr(self, tmp_path, dummy_dataset):
        lr_config = {
            "*lm_head*": {"lr": 1e-5},
        }
        trainer = _make_trainer(tmp_path, dummy_dataset, lr_config_dict=lr_config)
        trainer.create_optimizer()

        global_lr = 1e-3  # set in _make_trainer
        for group in trainer.optimizer.param_groups:
            if group["lr"] != 1e-5:
                assert group["lr"] == global_lr

    def test_no_lr_config_uses_default(self, tmp_path, dummy_dataset):
        trainer = _make_trainer(tmp_path, dummy_dataset)
        trainer.create_optimizer()

        for group in trainer.optimizer.param_groups:
            assert group["lr"] == 1e-3  # global default
