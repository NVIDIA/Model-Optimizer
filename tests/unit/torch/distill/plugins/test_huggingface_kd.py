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


def _v3():
    return pytest.importorskip("modelopt.torch.distill.plugins.offline_kd_v3")


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size

    def get_vocab(self):
        return {f"t{i}": i for i in range(self.vocab_size)}

    def __len__(self):
        return self.vocab_size


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
        processing_class=_Tokenizer() if offline_teacher_dir else None,
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
    """Full-vocabulary teacher targets reproduce the online KD loss without a teacher."""
    v3 = _v3()
    student, teacher = _make_models()
    expected_kd_loss = _manual_kd_loss(student, teacher, _make_batch())
    features = [dict(_ToyDataset()[i]) for i in range(len(_ToyDataset()))]
    with torch.no_grad():
        teacher_logits = teacher(input_ids=_make_batch()["input_ids"]).logits
    for b, feature in enumerate(features):
        positions = (feature["labels"][1:] != IGNORE_INDEX).nonzero().flatten()
        values, indices, tail = v3.topk_targets(teacher_logits[b, positions], 8, 8)
        feature["teacher_topk_positions"] = positions
        feature["teacher_topk_ids"] = indices
        feature["teacher_topk_logprobs"] = values
        feature["teacher_topk_tail"] = tail
    batch = OfflineTeacherCollator(default_data_collator)(features)
    trainer = _make_trainer(tmp_path, student, None, offline_teacher_dir=str(tmp_path))

    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch.copy())

    assert loss.item() == pytest.approx(expected_kd_loss.item(), rel=1e-5)


def test_offline_teacher_dataset_and_collator(tmp_path):
    """V3 samples come back as the rows they were written from, cut to top_k and padded per batch."""
    v3 = _v3()
    rows = [
        ([1, 2, 3, 4, 5], [IGNORE_INDEX, 2, 3, IGNORE_INDEX, IGNORE_INDEX]),
        ([2, 3, 4], [IGNORE_INDEX] * 3),  # No loss: a fully padded row.
        ([4, 5, 6, 7], [IGNORE_INDEX, IGNORE_INDEX, 6, 7]),
    ]
    samples = []
    for ids, labels in rows:
        sample = v3.strip_row(torch.tensor(ids), torch.tensor(labels))
        values, indices, tail = v3.topk_targets(torch.randn(len(sample["positions"]), 8), 4, 8)
        samples.append({**sample, "values": values.half(), "indices": indices, "tail": tail})
    settings = v3.corpus_settings(_Tokenizer(), SimpleNamespace(), 3, recorded={}, provenance={})
    v3.publish_chunk(str(tmp_path), samples[:2], settings, start=0, vocab_size=8)
    last = v3.publish_chunk(str(tmp_path), samples[2:], settings, start=2, vocab_size=8)
    ds = OfflineTeacherDataset(str(tmp_path), _Tokenizer(), max_length=6, top_k=2)

    pad = [IGNORE_INDEX] * 6
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 0, 0, 0]
    assert ds[0]["labels"].tolist() == [IGNORE_INDEX, 2, 3, *pad[:3]]
    assert ds[0]["attention_mask"].tolist() == [1, 1, 1, 0, 0, 0]
    assert ds[1]["attention_mask"].sum() == 0 and ds[1]["labels"].tolist() == pad
    assert ds[2]["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 6, 7, *pad[:2]]
    # The dropped targets join the tail, so the kept distribution still sums to one.
    kept = ds[2]["teacher_topk_logprobs"].float().exp().sum(-1) + ds[2]["teacher_topk_tail"].exp()
    torch.testing.assert_close(kept, torch.ones(2), atol=2e-3, rtol=0)

    batch = OfflineTeacherCollator(default_data_collator)([ds[0], ds[1], ds[2]])

    assert batch["teacher_topk_positions"].tolist() == [[0, 1], [-1, -1], [1, 2]]
    assert torch.equal(batch["teacher_topk_ids"][2], samples[2]["indices"][:, :2])
    with pytest.raises(ValueError, match="another tokenizer"):
        OfflineTeacherDataset(str(tmp_path), _Tokenizer(9), max_length=6)
    (tmp_path / last.split("/")[-1] / "index.json").unlink()
    with pytest.raises(ValueError, match="finish writing"):
        OfflineTeacherDataset(str(tmp_path), _Tokenizer(), max_length=6)


@pytest.mark.parametrize("top_k", [3, 8])
def test_fused_linear_topk_kl_matches_autograd(top_k):
    """The chunked loss and its forward-computed gradients equal autograd on the vocabulary logits."""
    v3 = _v3()
    torch.manual_seed(0)
    hidden = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(10, 6, dtype=torch.float64, requires_grad=True)  # 2 rows of padding
    values, ids, tail = v3.topk_targets(torch.randn(7, 10), top_k, vocab_size=8)
    tail = tail + 0.1  # A stored tail need not make the targets sum to exactly one.

    expected = topk_kl((hidden @ weight.t())[:, :8], ids, values, tail)
    expected_grads = torch.autograd.grad(expected, (hidden, weight))
    loss = fused_linear_topk_kl(hidden, weight, ids.int(), values, tail, vocab_size=8, chunk_size=3)
    grads = torch.autograd.grad(loss, (hidden, weight))

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad.double(), expected_grad, rtol=1e-5, atol=1e-6)
