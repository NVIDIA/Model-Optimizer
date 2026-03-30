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

"""Tests for AdaRound dist_loss integration with QATTrainer / QADTrainer."""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from datasets import Dataset
from transformers import TrainingArguments
from transformers.training_args import ParallelMode

import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss
from modelopt.torch.quantization.nn.modules.tensor_quantizer import NVFP4StaticAdaRoundQuantizer
from modelopt.torch.quantization.plugins.transformers_trainer import (
    AdaRoundTrainingArguments,
    QADTrainer,
    QATTrainer,
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


NVFP4_ADAROUND_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
    },
    "algorithm": {
        "method": "adaround",
        "init_algorithm": {
            "method": "smooth_lsq",
            "scale_algorithm": {"method": "mse", "fp8_scale_sweep": True},
        },
    },
}


def _make_dummy_dataset(seq_len=16, num_samples=8, vocab_size=32):
    """Random token dataset for minimal training."""
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


def _pre_quantize(model):
    """Pre-quantize model with adaround for use in trainer tests."""
    calib_data = [torch.randint(0, 32, (2, 16), device="cuda") for _ in range(4)]

    def forward_loop(model):
        for x in calib_data:
            model(x)

    mtq.quantize(model, NVFP4_ADAROUND_CFG, forward_loop)


class TestQATTrainer:
    """QATTrainer tests."""

    def test_no_adaround(self, tmp_path):
        """Standard QATTrainer flow without adaround — no dist_loss added."""
        model = get_tiny_llama().cuda()
        dataset = _make_dummy_dataset()

        trainer = QATTrainer(
            model=model,
            args=_make_training_args(tmp_path, max_steps=2),
            train_dataset=dataset,
            quant_args=QuantizationArgumentsWithConfig(quant_cfg="INT8_DEFAULT_CFG"),
        )
        trainer.train()

        # No adaround quantizers should exist in a standard INT8 config
        assert not any(
            isinstance(m, NVFP4StaticAdaRoundQuantizer) and m._adaround_enabled
            for m in model.modules()
        )

    def test_adaround(self, tmp_path):
        """QATTrainer with adaround: dist_loss added, round_logits get gradients."""
        model = get_tiny_llama().cuda()
        dataset = _make_dummy_dataset()

        _pre_quantize(model)

        trainer = QATTrainer(
            model=model,
            args=_make_training_args(tmp_path, max_steps=4),
            train_dataset=dataset,
            quant_args=QuantizationArguments(quant_cfg=None),
            adaround_args=AdaRoundTrainingArguments(),
        )
        trainer.train()

        adaround_quantizers = [
            m
            for m in model.modules()
            if isinstance(m, NVFP4StaticAdaRoundQuantizer) and m._adaround_enabled
        ]
        assert len(adaround_quantizers) > 0
        for q in adaround_quantizers:
            assert q.round_logits.requires_grad
        # Verify adaround metrics were logged
        logged_keys = {k for entry in trainer.state.log_history for k in entry}
        assert "adaround/dist_loss" in logged_keys
        assert "adaround/beta" in logged_keys


class TestQADTrainer:
    """QADTrainer tests with adaround."""

    def test_adaround(self, tmp_path):
        """QADTrainer with adaround: KD loss + dist_loss both active."""
        student = get_tiny_llama().cuda()
        teacher = get_tiny_llama().cuda()
        teacher.eval()
        dataset = _make_dummy_dataset()

        _pre_quantize(student)

        distill_config = {
            "teacher_model": teacher,
            "criterion": LMLogitsLoss(),
        }

        trainer = QADTrainer(
            model=student,
            args=_make_training_args(tmp_path, max_steps=4),
            train_dataset=dataset,
            quant_args=QuantizationArguments(quant_cfg=None),
            adaround_args=AdaRoundTrainingArguments(),
            distill_config=distill_config,
        )
        trainer.train()

        adaround_quantizers = [
            m
            for m in student.modules()
            if isinstance(m, NVFP4StaticAdaRoundQuantizer) and m._adaround_enabled
        ]
        assert len(adaround_quantizers) > 0
        # Verify adaround metrics were logged
        logged_keys = {k for entry in trainer.state.log_history for k in entry}
        assert "adaround/dist_loss" in logged_keys
