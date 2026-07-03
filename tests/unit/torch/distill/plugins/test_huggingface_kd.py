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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

transformers = pytest.importorskip("transformers")
TrainingArguments = transformers.TrainingArguments
default_data_collator = transformers.default_data_collator
from transformers.modeling_outputs import CausalLMOutputWithPast

from modelopt.torch.distill.losses import LogitsDistillationLoss
from modelopt.torch.distill.plugins.huggingface import IGNORE_INDEX, KDTrainer


class _TinyCausalLM(nn.Module):
    def __init__(self, name=None, events=None):
        super().__init__()
        self.name = name
        self.events = events
        self.config = SimpleNamespace(use_cache=False)
        self.embed = nn.Embedding(8, 6)
        self.lm_head = nn.Linear(6, 8, bias=False)

    def forward(self, input_ids, labels=None):
        if self.events is not None:
            self.events.append((self.name, labels is not None))
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class _ToyDataset(Dataset):
    def __init__(self):
        self.examples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([1, 2, 3, 4]),
            },
            {
                "input_ids": torch.tensor([2, 3, 4, 5]),
                "labels": torch.tensor([2, IGNORE_INDEX, 4, 5]),
            },
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _make_models(events=None):
    torch.manual_seed(0)
    student = _TinyCausalLM("student", events)
    teacher = _TinyCausalLM("teacher", events)
    with torch.no_grad():
        teacher.lm_head.weight.add_(0.25)
    return student, teacher


def _make_batch():
    return default_data_collator([_ToyDataset()[0], _ToyDataset()[1]])


def _make_trainer(tmp_path, student, teacher, use_liger_kernel=False):
    training_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_eval_batch_size=2,
        report_to=[],
        use_cpu=True,
    )
    training_args.use_liger_kernel = use_liger_kernel
    return KDTrainer(
        model=student,
        args=training_args,
        eval_dataset=_ToyDataset(),
        data_collator=default_data_collator,
        distill_args={"teacher_model": teacher},
    )


def _manual_kd_loss(student, teacher, batch):
    with torch.no_grad():
        student_outputs = student(input_ids=batch["input_ids"])
        teacher_outputs = teacher(input_ids=batch["input_ids"])
    criterion = LogitsDistillationLoss(reduction="none")
    per_token_loss = criterion(
        student_outputs.logits[..., :-1, :].contiguous().float(),
        teacher_outputs.logits[..., :-1, :].contiguous().float(),
    )
    mask = batch["labels"][..., 1:].contiguous() != IGNORE_INDEX
    return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)


def test_training_loss_is_kd_and_skips_ce(tmp_path):
    events = []
    student, teacher = _make_models(events)
    batch = _make_batch()
    expected_kd_loss = _manual_kd_loss(student, teacher, batch)
    trainer = _make_trainer(tmp_path, student, teacher)

    events.clear()
    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    assert events == [("student", False), ("teacher", False)]
    assert loss.item() == pytest.approx(expected_kd_loss.item())


def test_eval_loss_is_kd_and_ce_is_secondary_metric(tmp_path):
    student, teacher = _make_models()
    batch = _make_batch()
    expected_ce_loss = student(**batch).loss.detach()
    expected_kd_loss = _manual_kd_loss(student, teacher, batch)
    trainer = _make_trainer(tmp_path, student, teacher)

    metrics = trainer.evaluate()

    assert metrics["eval_loss"] == pytest.approx(expected_kd_loss.item())
    assert metrics["eval_ce_loss"] == pytest.approx(expected_ce_loss.item())
    assert "eval_kd_loss" not in metrics


def test_standard_kd_loss_without_labels_uses_mean(tmp_path):
    student, teacher = _make_models()
    batch = _make_batch()
    trainer = _make_trainer(tmp_path, student, teacher)

    with torch.no_grad():
        outputs = student(input_ids=batch["input_ids"])
        teacher_outputs = teacher(input_ids=batch["input_ids"])
    trainer._last_teacher_outputs = teacher_outputs

    loss = trainer._standard_kd_loss(outputs, labels=None)
    expected = LogitsDistillationLoss(reduction="none")(
        outputs.logits[..., :-1, :].contiguous().float(),
        teacher_outputs.logits[..., :-1, :].contiguous().float(),
    ).mean()

    assert loss.item() == pytest.approx(expected.item())
