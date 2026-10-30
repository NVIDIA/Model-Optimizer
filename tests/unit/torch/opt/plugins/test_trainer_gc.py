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

import gc
import weakref

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from modelopt.torch.opt.plugins.transformers import ModelOptHFTrainer, ModelOptTrainerArguments


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.lm_head(self.embed(input_ids))
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


class _FreezeCounts(transformers.TrainerCallback):
    def __init__(self):
        self.after_step = {}

    def on_step_end(self, args, state, control, **kwargs):
        self.after_step[state.global_step] = gc.get_freeze_count()


def test_manual_gc_freezes_during_train_and_unfreezes_after(tmp_path):
    """``manual_gc`` freezes what lives through training and ``train()`` unfreezes it.

    Collections skip frozen objects, which is the speed-up. A frozen object in a reference cycle is
    never collected, so without the unfreeze the model would outlive the trainer.
    """
    frozen_before = gc.get_freeze_count()  # Python itself starts with a few hundred frozen objects
    freeze_counts = _FreezeCounts()
    model = _TinyModel()
    model.__dict__["_cycle"] = [model]  # only the cycle collector can free it
    model_ref = weakref.ref(model)
    dataset = [
        {"input_ids": torch.randint(0, 32, (8,)), "labels": torch.randint(0, 32, (8,))}
        for _ in range(8)
    ]
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=3,
        per_device_train_batch_size=2,
        report_to="none",
        use_cpu=True,
    )
    trainer = ModelOptHFTrainer(
        model=model,
        args=args,
        trainer_args=ModelOptTrainerArguments(manual_gc=True),
        train_dataset=dataset,
        callbacks=[freeze_counts],
    )
    try:
        trainer.train()
        # The collection before the second step freezes the model and optimizer state.
        assert all(freeze_counts.after_step[step] > frozen_before for step in (2, 3))
        # gc.unfreeze() also returns the objects frozen before training.
        assert gc.get_freeze_count() == 0
        del trainer, model
        gc.collect()
        assert model_ref() is None
    finally:
        gc.unfreeze()
