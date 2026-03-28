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

"""Tests for quantization error regularization with QATTrainer."""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from datasets import Dataset
from transformers import TrainingArguments
from transformers.training_args import ParallelMode

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.transformers_trainer import (
    QATTrainer,
    QuantErrorTrainingArguments,
    QuantizationArguments,
    QuantizationArgumentsWithConfig,
)


@pytest.fixture(autouse=True)
def _single_gpu(monkeypatch):
    """Force single-GPU mode to prevent DataParallel wrapping."""
    monkeypatch.setattr(
        TrainingArguments, "parallel_mode", property(lambda self: ParallelMode.NOT_PARALLEL)
    )
    monkeypatch.setattr(TrainingArguments, "n_gpu", property(lambda self: 1))


def _make_dummy_dataset(seq_len=16, num_samples=8, vocab_size=32):
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})


def _make_training_args(tmp_dir, max_steps=4, **overrides):
    defaults = {
        "output_dir": str(tmp_dir),
        "max_steps": max_steps,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "no_cuda": False,
        "report_to": "none",
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _pre_quantize_int8(model):
    calib_data = [torch.randint(0, 32, (2, 16), device="cuda") for _ in range(4)]

    def forward_loop(model):
        for x in calib_data:
            model(x)

    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)


class TestQuantErrorCallback:
    """Tests for quantization error regularization callback."""

    def test_qerr_int8(self, tmp_path):
        """QuantError with pre-quantized INT8 model."""
        model = get_tiny_llama().cuda()
        dataset = _make_dummy_dataset()

        _pre_quantize_int8(model)

        trainer = QATTrainer(
            model=model,
            args=_make_training_args(tmp_path, max_steps=4),
            train_dataset=dataset,
            quant_args=QuantizationArguments(quant_cfg=None),
            qerr_args=QuantErrorTrainingArguments(),
        )
        trainer.train()

        logged_keys = {k for entry in trainer.state.log_history for k in entry}
        assert "qerr/sum" in logged_keys
        assert "qerr/coeff" in logged_keys

    def test_qerr_with_lazy_quantization(self, tmp_path):
        """QuantError set up after trainer-based quantization."""
        model = get_tiny_llama().cuda()
        dataset = _make_dummy_dataset()

        trainer = QATTrainer(
            model=model,
            args=_make_training_args(tmp_path, max_steps=4),
            train_dataset=dataset,
            quant_args=QuantizationArgumentsWithConfig(quant_cfg="INT8_DEFAULT_CFG"),
            qerr_args=QuantErrorTrainingArguments(),
        )
        trainer.train()

        logged_keys = {k for entry in trainer.state.log_history for k in entry}
        assert "qerr/sum" in logged_keys
        assert "qerr/coeff" in logged_keys

    def test_no_qerr(self, tmp_path):
        """No qerr_args — no qerr metrics should be logged."""
        model = get_tiny_llama().cuda()
        dataset = _make_dummy_dataset()

        trainer = QATTrainer(
            model=model,
            args=_make_training_args(tmp_path, max_steps=2),
            train_dataset=dataset,
            quant_args=QuantizationArgumentsWithConfig(quant_cfg="INT8_DEFAULT_CFG"),
        )
        trainer.train()

        logged_keys = {k for entry in trainer.state.log_history for k in entry}
        assert "qerr/sum" not in logged_keys
        assert "qerr/coeff" not in logged_keys

