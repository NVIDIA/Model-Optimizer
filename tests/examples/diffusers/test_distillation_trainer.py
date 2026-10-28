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

"""Exercise LTX checkpoint finalization without LTX model assets or a GPU."""

import importlib.util
import itertools
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from accelerate import Accelerator
from pydantic import BaseModel
from safetensors.torch import load_file


@pytest.fixture
def trainer_module(monkeypatch):
    # LTX is an optional, separately licensed dependency. Substitute its model setup
    # and UI, but import the complete example and execute its train/save/export methods.
    progress = nullcontext()
    progress.update_training = lambda **kwargs: None
    for name, attributes in {
        "ltx_trainer": {"logger": logging.getLogger(__name__)},
        "ltx_trainer.config": {"ConfigBaseModel": BaseModel, "LtxTrainerConfig": BaseModel},
        "ltx_trainer.model_loader": {"load_transformer": None},
        "ltx_trainer.trainer": {
            "IS_MAIN_PROCESS": True,
            "LtxvTrainer": object,
            "TrainingStats": SimpleNamespace,
        },
        "ltx_trainer.gpu_utils": {"get_gpu_memory_gb": lambda device: 0},
        "ltx_trainer.hf_hub_utils": {"push_to_hub": None},
        "ltx_trainer.progress": {"TrainingProgress": lambda **kwargs: progress},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    path = (
        Path(__file__).resolve().parents[3]
        / "examples/diffusers/distillation/distillation_trainer.py"
    )
    spec = importlib.util.spec_from_file_location("ltx_checkpoint_example", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "is_global_rank0", lambda: True)
    # Deterministically cross the one-minute deadline during the first optimizer step.
    clock = itertools.count(step=60)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: next(clock)))
    return module


@pytest.mark.parametrize("grad_accum", [1, 2])
@pytest.mark.parametrize(
    ("steps", "interval", "deadline", "saved_steps"),
    [
        pytest.param(2, 1, None, [1, 2], id="interval-aligned"),
        pytest.param(3, 2, None, [2, 3], id="nonaligned-final"),
        pytest.param(3, 1, 1, [1], id="time-limit-with-interval"),
        pytest.param(3, 2, 1, [1], id="time-limit-without-interval"),
    ],
)
def test_train_final_checkpoint(
    trainer_module, tmp_path, grad_accum, steps, interval, deadline, saved_steps
):
    class TinyTrainer(trainer_module.DistillationTrainer):
        def __init__(self):
            self._accelerator = Accelerator(cpu=True, gradient_accumulation_steps=grad_accum)
            self._config = SimpleNamespace(
                seed=42,
                output_dir=str(tmp_path),
                model=SimpleNamespace(training_mode="full"),
                optimization=SimpleNamespace(
                    steps=steps,
                    gradient_accumulation_steps=grad_accum,
                    max_grad_norm=1.0,
                    batch_size=1,
                ),
                checkpoints=SimpleNamespace(interval=interval, keep_last_n=1, precision="float32"),
                validation=SimpleNamespace(interval=None, skip_initial_validation=True),
                wandb=SimpleNamespace(log_validation_videos=False),
                hub=SimpleNamespace(push_to_hub=False),
            )
            self._distillation_config = trainer_module.DistillationConfig(must_save_by=deadline)
            self._transformer = torch.nn.Linear(2, 1)
            self._lr_scheduler = None
            self._wandb_run = None
            self._checkpoint_paths = []

        def _init_optimizer(self):
            optimizer = torch.optim.AdamW(self._transformer.parameters(), lr=0.01)
            self._transformer, self._optimizer = self._accelerator.prepare(
                self._transformer, optimizer
            )
            self._trainable_params = list(self._transformer.parameters())

        def _init_dataloader(self):
            self._dataloader = [torch.ones(1, 2)] * (steps * grad_accum)

        def _training_step(self, batch):
            return self._transformer(batch).square().mean()

        def _init_timestep_sampler(self):
            pass

        def _save_config(self):
            pass

        def _log_metrics(self, *args):
            pass

        def _log_training_stats(self, *args):
            pass

    trainer = TinyTrainer()
    save_state = trainer._save_training_state
    observed_steps = []

    def record_save():
        observed_steps.append(trainer._global_step)
        return save_state()

    trainer._save_training_state = record_save
    trainer._save_checkpoint = Mock(wraps=trainer._save_checkpoint)
    try:
        exported_path, _ = trainer.train(disable_progress_bars=True)
        assert observed_steps == saved_steps
        trainer._save_checkpoint.assert_called_once()
        assert trainer._checkpoint_paths == [exported_path]
        exported = load_file(exported_path)
        assert all(
            torch.equal(exported[k], v) for k, v in trainer._transformer.state_dict().items()
        )
        checkpoint_dir = tmp_path / "checkpoints"
        assert sorted(p.name for p in checkpoint_dir.glob("step_*")) == [
            f"step_{saved_steps[-1]:06d}"
        ]
        metadata = checkpoint_dir / f"step_{saved_steps[-1]:06d}" / "distillation_metadata.json"
        assert json.loads(metadata.read_text())["global_step"] == saved_steps[-1]
    finally:
        trainer._accelerator.free_memory()
