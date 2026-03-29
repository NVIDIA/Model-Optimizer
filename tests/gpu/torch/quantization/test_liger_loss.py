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

"""Tests that liger fused loss and regular loss produce equivalent results."""

import copy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from datasets import Dataset
from transformers import TrainingArguments, set_seed
from transformers.training_args import ParallelMode

pytest.importorskip("liger_kernel")

from modelopt.torch.distill.plugins.huggingface import KDTrainer, LMLogitsLoss
from modelopt.torch.opt.plugins.transformers import ModelOptHFTrainer


@pytest.fixture(autouse=True)
def _single_gpu(monkeypatch):
    """Force single-GPU mode to prevent DataParallel wrapping."""
    monkeypatch.setattr(
        TrainingArguments, "parallel_mode", property(lambda self: ParallelMode.NOT_PARALLEL)
    )
    monkeypatch.setattr(TrainingArguments, "n_gpu", property(lambda self: 1))


def _make_dummy_dataset(seq_len=16, num_samples=8, vocab_size=32):
    """Random token dataset for minimal training."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})


def _make_training_args(tmp_dir, max_steps=2, **overrides):
    defaults = {
        "output_dir": str(tmp_dir),
        "max_steps": max_steps,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "no_cuda": False,
        "report_to": "none",
        "learning_rate": 1e-4,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _extract_losses(trainer):
    """Extract per-step training losses from trainer log history."""
    return [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]


class TestLigerCELoss:
    """Verify liger fused CE loss matches regular CE loss."""

    def test_liger_ce_loss(self, tmp_path):
        seed = 42
        set_seed(seed)
        torch.manual_seed(seed)
        base_model = get_tiny_llama().cuda()

        model_a = copy.deepcopy(base_model)
        model_b = copy.deepcopy(base_model)

        set_seed(seed)
        dataset = _make_dummy_dataset()

        set_seed(seed)
        trainer_a = ModelOptHFTrainer(
            model=model_a,
            args=_make_training_args(tmp_path / "no_liger", use_liger_kernel=False),
            train_dataset=dataset,
        )
        trainer_a.train()

        set_seed(seed)
        trainer_b = ModelOptHFTrainer(
            model=model_b,
            args=_make_training_args(tmp_path / "liger", use_liger_kernel=True),
            train_dataset=dataset,
        )
        trainer_b.train()

        losses_a = _extract_losses(trainer_a)
        losses_b = _extract_losses(trainer_b)

        assert len(losses_a) == len(losses_b) > 0
        for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
            assert torch.allclose(torch.tensor(la), torch.tensor(lb), atol=1e-5), (
                f"CE loss mismatch at step {step}: regular={la}, liger={lb}"
            )


class TestLigerJSDLoss:
    """Verify liger fused JSD loss matches regular JSD loss."""

    def test_liger_jsd_loss(self, tmp_path):
        seed = 42
        set_seed(seed)
        torch.manual_seed(seed)
        base_student = get_tiny_llama().cuda()
        base_teacher = get_tiny_llama().cuda()
        base_teacher.eval()

        student_a = copy.deepcopy(base_student)
        student_b = copy.deepcopy(base_student)
        teacher_a = copy.deepcopy(base_teacher)
        teacher_b = copy.deepcopy(base_teacher)

        set_seed(seed)
        dataset = _make_dummy_dataset()

        set_seed(seed)
        trainer_a = KDTrainer(
            model=student_a,
            args=_make_training_args(tmp_path / "no_liger", use_liger_kernel=False),
            train_dataset=dataset,
            distill_config={
                "teacher_model": teacher_a,
                "criterion": LMLogitsLoss(),
            },
        )
        trainer_a.train()

        set_seed(seed)
        trainer_b = KDTrainer(
            model=student_b,
            args=_make_training_args(tmp_path / "liger", use_liger_kernel=True),
            train_dataset=dataset,
            distill_config={
                "teacher_model": teacher_b,
                "criterion": LMLogitsLoss(),
            },
        )
        trainer_b.train()

        losses_a = _extract_losses(trainer_a)
        losses_b = _extract_losses(trainer_b)

        assert len(losses_a) == len(losses_b) > 0
        for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
            assert torch.allclose(torch.tensor(la), torch.tensor(lb), atol=1e-5), (
                f"JSD loss mismatch at step {step}: regular={la}, liger={lb}"
            )
