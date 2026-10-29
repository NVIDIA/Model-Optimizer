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


def test_model_is_freed_after_train_with_manual_gc(tmp_path):
    """``manual_gc`` freezes what lives through training; ``train()`` must unfreeze it.

    A frozen object in a reference cycle is never collected, so the model would outlive the trainer.
    """
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
    )
    try:
        trainer.train()
        del trainer, model
        gc.collect()
        assert model_ref() is None
    finally:
        gc.unfreeze()
