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
from unittest.mock import PropertyMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

transformers = pytest.importorskip("transformers")
TrainingArguments = transformers.TrainingArguments
default_data_collator = transformers.default_data_collator
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.distill.losses import LogitsDistillationLoss
from modelopt.torch.distill.plugins.huggingface import IGNORE_INDEX, KDTrainer
from modelopt.torch.distill.plugins.offline_teacher import (
    OfflineTeacherCollator,
    OfflineTeacherDataset,
    fused_linear_topk_kl,
    input_checksum,
    topk_kl,
)


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


def _make_trainer(
    tmp_path,
    student,
    teacher,
    use_liger_kernel=False,
    kd_loss_alpha=1.0,
    offline_teacher_dir=None,
    **kwargs,
):
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
        distill_args={
            "teacher_model": teacher,
            "kd_loss_alpha": kd_loss_alpha,
            "offline_teacher_dir": offline_teacher_dir,
        },
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


def test_training_loss_is_pure_kd_by_default(tmp_path):
    """Default kd_loss_alpha=1.0 produces pure KD loss; student is forwarded without labels."""
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


def test_training_loss_combines_ce_and_kd_with_equal_weights(tmp_path):
    """kd_loss_alpha=0.5 blends CE and KD losses with equal weights."""
    student, teacher = _make_models()
    batch = _make_batch()
    expected_kd_loss = _manual_kd_loss(student, teacher, batch)
    expected_ce_loss = student(**batch).loss.detach()
    trainer = _make_trainer(tmp_path, student, teacher, kd_loss_alpha=0.5)

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    expected = 0.5 * expected_kd_loss + 0.5 * expected_ce_loss
    assert loss.item() == pytest.approx(expected.item())


def test_training_loss_combines_ce_and_kd_with_custom_weights(tmp_path):
    """kd_loss_alpha=0.7 blends 70% KD + 30% CE — the config from issue #2488."""
    student, teacher = _make_models()
    batch = _make_batch()
    expected_kd_loss = _manual_kd_loss(student, teacher, batch)
    expected_ce_loss = student(**batch).loss.detach()
    trainer = _make_trainer(tmp_path, student, teacher, kd_loss_alpha=0.7)

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    expected = 0.7 * expected_kd_loss + 0.3 * expected_ce_loss
    assert loss.item() == pytest.approx(expected.item())


def test_training_loss_preserves_label_smoothing(tmp_path):
    """CE loss handling should preserve HF Trainer's label_smoothing_factor."""
    student, teacher = _make_models()
    batch = _make_batch()

    trainer = _make_trainer(tmp_path, student, teacher, kd_loss_alpha=0.5)
    trainer.args.label_smoothing_factor = 0.1

    smoother = LabelSmoother(epsilon=0.1)
    trainer.label_smoother = smoother

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    expected_kd_loss = _manual_kd_loss(student, teacher, batch)
    outputs = student(**batch)
    # The dummy model is not in HF's causal LM mapping, so HF Trainer won't pass shift_labels=True
    expected_ce_loss = smoother(outputs, batch["labels"], shift_labels=False)

    expected = 0.5 * expected_kd_loss + 0.5 * expected_ce_loss
    assert loss.item() == pytest.approx(expected.item())


def test_invalid_kd_loss_alpha_raises(tmp_path):
    """kd_loss_alpha outside (0, 1] should raise ValueError at construction time."""
    student, teacher = _make_models()
    with pytest.raises(ValueError, match="kd_loss_alpha"):
        _make_trainer(tmp_path, student, teacher, kd_loss_alpha=0.0)
    with pytest.raises(ValueError, match="kd_loss_alpha"):
        _make_trainer(tmp_path, student, teacher, kd_loss_alpha=1.5)


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


def test_kd_loss_ddp_scaling(tmp_path):
    """KD loss should be scaled by world_size when average_tokens_across_devices is True."""
    student, teacher = _make_models()
    batch = _make_batch()

    trainer_single = _make_trainer(tmp_path / "single", student, teacher, kd_loss_alpha=0.5)
    trainer_single.model.train()

    mask = batch["labels"][..., 1:] != IGNORE_INDEX
    num_items = mask.sum().item()

    loss_single = trainer_single.compute_loss(
        trainer_single.model, batch.copy(), num_items_in_batch=num_items
    )

    with patch(
        "transformers.TrainingArguments.world_size", new_callable=PropertyMock
    ) as mock_world_size:
        mock_world_size.return_value = 2

        trainer_ddp = _make_trainer(tmp_path / "ddp", student, teacher, kd_loss_alpha=0.5)
        trainer_ddp.args.average_tokens_across_devices = True
        trainer_ddp.model.train()

        # DDP aggregates num_items across world_size
        loss_ddp = trainer_ddp.compute_loss(
            trainer_ddp.model, batch.copy(), num_items_in_batch=num_items * 2
        )

    # In our dummy model, CE loss natively averages over its batch, so dividing
    # the KD loss by (num_items * 2) and multiplying by world_size (2) should
    # perfectly match the single-rank calculation.
    assert loss_ddp.item() == pytest.approx(loss_single.item())


def test_offline_topk_teacher_matches_online_kd(tmp_path):
    """Dumped full-vocabulary teacher log-probs reproduce the online KD loss without a teacher."""
    student, teacher = _make_models()
    expected_kd_loss = _manual_kd_loss(student, teacher, _make_batch())
    features = [dict(_ToyDataset()[i]) for i in range(len(_ToyDataset()))]
    with torch.no_grad():
        teacher_logits = teacher(input_ids=_make_batch()["input_ids"]).logits.float()
    logprobs, ids = teacher_logits.log_softmax(-1).topk(teacher_logits.size(-1), dim=-1)
    for b, feature in enumerate(features):
        positions = (feature["labels"][1:] != IGNORE_INDEX).nonzero().flatten()
        feature["teacher_topk_positions"] = positions
        feature["teacher_topk_ids"] = ids[b, positions].int()
        feature["teacher_topk_logprobs"] = logprobs[b, positions]
    batch = OfflineTeacherCollator(default_data_collator)(features)
    trainer = _make_trainer(tmp_path, student, None, offline_teacher_dir=str(tmp_path))

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    assert loss.item() == pytest.approx(expected_kd_loss.item(), rel=1e-5)


def test_offline_teacher_dataset_and_collator(tmp_path):
    """Dumped rows come back by dataset row, cut to top_k and padded per batch; a changed row raises."""
    save_file = pytest.importorskip("safetensors.torch").save_file
    rows = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4, "labels": [IGNORE_INDEX, 2, 3, 4]},
        {
            "input_ids": [2, 3, 4, 0],
            "attention_mask": [1, 1, 1, 0],
            "labels": [IGNORE_INDEX] * 2 + [4, IGNORE_INDEX],
        },
    ]
    # Loss positions t (label t + 1 trains): 0, 1, 2 in row 0 and 1 in row 1.
    logprobs, ids = torch.randn(4, 8).log_softmax(-1).topk(4, dim=-1)
    (tmp_path / "train").mkdir()
    (tmp_path / "meta.json").write_text('{"top_k": 4}')
    save_file(
        {
            "row": torch.tensor([0, 1]),
            "num_tokens": torch.tensor([4, 3]),
            "checksum": torch.tensor(
                [
                    input_checksum(torch.tensor([1, 2, 3, 4])),
                    input_checksum(torch.tensor([2, 3, 4])),
                ]
            ),
            "offsets": torch.tensor([0, 3, 4]),
            "positions": torch.tensor([0, 1, 2, 1]),
            "ids": ids.int(),
            "logprobs": logprobs.half(),
        },
        str(tmp_path / "train" / "shard-00000.safetensors"),
    )
    ds = OfflineTeacherDataset(rows, str(tmp_path), "train", top_k=2)

    batch = OfflineTeacherCollator(default_data_collator)([ds[0], ds[1]])

    assert batch["teacher_topk_positions"].tolist() == [[0, 1, 2], [1, -1, -1]]
    assert torch.equal(batch["teacher_topk_ids"][0], ids[:3, :2].int())
    assert torch.equal(batch["teacher_topk_logprobs"][1, :1], logprobs[3:, :2].half())
    rows[1]["input_ids"] = [2, 3, 5, 0]
    with pytest.raises(ValueError, match="does not match"):
        ds[1]


@pytest.mark.parametrize("top_k", [3, 8])
def test_fused_linear_topk_kl_matches_autograd(top_k):
    """The chunked loss and its forward-computed gradients equal plain autograd on full logits."""
    torch.manual_seed(0)
    hidden = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    logprobs, ids = torch.randn(7, 8, dtype=torch.float64).log_softmax(-1).topk(top_k, dim=-1)

    expected = topk_kl(hidden @ weight.t(), ids, logprobs)
    expected_grads = torch.autograd.grad(expected, (hidden, weight))
    loss = fused_linear_topk_kl(hidden, weight, ids.int(), logprobs, chunk_size=3)
    grads = torch.autograd.grad(loss, (hidden, weight))

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad.double(), expected_grad, rtol=1e-5, atol=1e-6)
