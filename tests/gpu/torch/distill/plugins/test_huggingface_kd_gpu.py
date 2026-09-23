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

"""GPU tests for KDTrainer Liger kernel paths.

These tests require CUDA and the ``liger_kernel`` package.  They exercise
the fused Liger KD loss and its interaction with gradient accumulation
and label smoothing.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

transformers = pytest.importorskip("transformers")
liger_kernel = pytest.importorskip("liger_kernel")
TrainingArguments = transformers.TrainingArguments
default_data_collator = transformers.default_data_collator
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.distill.losses import LogitsDistillationLoss
from modelopt.torch.distill.plugins.huggingface import IGNORE_INDEX, KDTrainer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/unit/.../test_huggingface_kd.py fixtures)
# ---------------------------------------------------------------------------


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


def _make_trainer(tmp_path, student, teacher, use_liger_kernel=False, kd_loss_weight=1.0, **kwargs):
    use_cpu = kwargs.pop("use_cpu", True)
    training_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_eval_batch_size=2,
        report_to=[],
        use_cpu=use_cpu,
        **kwargs,
    )
    training_args.use_liger_kernel = use_liger_kernel
    return KDTrainer(
        model=student,
        args=training_args,
        eval_dataset=_ToyDataset(),
        data_collator=default_data_collator,
        distill_args={"teacher_model": teacher, "kd_loss_weight": kd_loss_weight},
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


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


def test_liger_kd_gradient_accumulation_scaling(tmp_path):
    """Liger KD and CE losses must accumulate to the equivalent unsplit batch."""
    student, teacher = _make_models()

    dataset = _ToyDataset()
    examples = [dataset[0], dataset[1], dataset[0], dataset[1]]
    full_batch = default_data_collator(examples)
    full_batch = {k: v.cuda() for k, v in full_batch.items()}

    trainer = _make_trainer(
        tmp_path, student, teacher, use_liger_kernel=True, kd_loss_weight=0.5, use_cpu=False
    )

    trainer.model.train()
    trainer.model.zero_grad()
    loss_full = trainer.compute_loss(trainer.model, full_batch.copy())
    loss_full.backward()
    grad_full = student.lm_head.weight.grad.clone()

    trainer.model.zero_grad()
    mb1 = default_data_collator(examples[:2])
    mb1 = {k: v.cuda() for k, v in mb1.items()}
    mb2 = default_data_collator(examples[2:])
    mb2 = {k: v.cuda() for k, v in mb2.items()}

    mask_full = full_batch["labels"][..., 1:] != IGNORE_INDEX
    num_items_in_batch = mask_full.sum().item()

    loss_mb1 = trainer.compute_loss(
        trainer.model, mb1.copy(), num_items_in_batch=num_items_in_batch
    )
    loss_mb1.backward()

    loss_mb2 = trainer.compute_loss(
        trainer.model, mb2.copy(), num_items_in_batch=num_items_in_batch
    )
    loss_mb2.backward()

    grad_accumulated = student.lm_head.weight.grad.clone()

    torch.testing.assert_close(grad_accumulated, grad_full, rtol=1e-3, atol=1e-3)


def test_training_loss_preserves_liger_label_smoothing(tmp_path):
    """Liger CE loss handling should preserve liger_ce_label_smoothing."""
    student, teacher = _make_models()

    batch = _make_batch()
    batch = {k: v.cuda() for k, v in batch.items()}

    trainer = _make_trainer(
        tmp_path, student, teacher, use_liger_kernel=True, kd_loss_weight=0.5, use_cpu=False
    )

    if not hasattr(trainer, "trainer_args"):
        trainer.trainer_args = SimpleNamespace()
    trainer.trainer_args.liger_ce_label_smoothing = 0.1

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    # Run manual calculation on CPU to avoid complex Liger mimicry on GPU
    student_cpu = student.cpu()
    teacher_cpu = teacher.cpu()
    batch_cpu = {k: v.cpu() for k, v in batch.items()}

    expected_kd_loss = _manual_kd_loss(student_cpu, teacher_cpu, batch_cpu)
    outputs = student_cpu(**batch_cpu)

    smoother = LabelSmoother(epsilon=0.1)
    # Liger's Causal LM fused CE kernel natively shifts labels
    expected_ce_loss = smoother(outputs, batch_cpu["labels"], shift_labels=True)

    expected = 0.5 * expected_kd_loss + 0.5 * expected_ce_loss
    assert loss.item() == pytest.approx(expected.item(), rel=1e-3)
