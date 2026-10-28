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

from collections import Counter

import pytest
import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from modelopt.torch.utils import BatchEchoPlan, EchoLoader, ExampleEchoPlan


def _check_plan(plan, n_batches, b, capacity=None):
    """Every example used exactly ``uses`` times, b distinct per step, never before it arrives."""
    steps = list(plan.steps(n_batches, b))
    assert len(steps) == plan.num_steps(n_batches)
    arrived = retired = 0
    uses = Counter()
    for n_pull, ords in steps:
        arrived += n_pull * b
        assert len(set(ords)) == b
        assert max(ords) < arrived
        if capacity is not None:
            assert arrived - retired <= capacity
        uses.update(ords)
        retired += sum(uses[o] == plan.uses for o in ords)
    assert arrived == n_batches * b
    assert uses == Counter(dict.fromkeys(range(n_batches * b), plan.uses))
    return steps


@pytest.mark.parametrize("uses", [1, 3])
def test_batch_echo_plan_repeats_each_batch_in_a_row(uses):
    steps = _check_plan(BatchEchoPlan(uses), n_batches=5, b=2)
    for t, (n_pull, ords) in enumerate(steps):
        assert n_pull == (t % uses == 0)
        assert ords == [2 * (t // uses), 2 * (t // uses) + 1]


@pytest.mark.parametrize(
    ("uses", "b", "min_gap", "n_batches"), [(1, 2, 4, 3), (2, 4, 1, 2), (4, 2, 4, 97), (7, 1, 9, 5)]
)
def test_example_echo_plan_invariants(uses, b, min_gap, n_batches):
    plan = ExampleEchoPlan(uses=uses, min_gap=min_gap, capacity=b * (min_gap + 3), seed=3)
    _check_plan(plan, n_batches, b, capacity=plan.capacity)


def test_example_echo_plan_keeps_the_gap_except_at_the_tail():
    b, min_gap = 2, 32
    plan = ExampleEchoPlan(uses=4, min_gap=min_gap, capacity=b * (min_gap + 2 + 1), seed=0)
    last, early_at = {}, []
    for t, (_, ords) in enumerate(plan.steps(2150, b)):
        for o in ords:
            if o in last and t - last[o] < min_gap:
                early_at.append(t)
            last[o] = t
    assert early_at
    assert min(early_at) >= plan.num_steps(2150) - plan.uses * min_gap


def test_example_echo_plan_is_seeded():
    plan = ExampleEchoPlan(uses=4, min_gap=4, capacity=16, seed=11)
    steps = list(plan.steps(60, 2, epoch=0))
    assert steps == list(plan.steps(60, 2, epoch=0))
    assert steps != list(plan.steps(60, 2, epoch=1))
    assert steps != list(ExampleEchoPlan(uses=4, min_gap=4, capacity=16, seed=12).steps(60, 2))


class _Loader:
    """A sized loader of collated batches shaped like the speculative-decoding collators' output."""

    def __init__(self, n_batches, b, length=None):
        self.batches = [
            {
                "input_ids": torch.arange(i * b, (i + 1) * b).unsqueeze(1).repeat(1, 3),
                "base_model_outputs": {
                    "aux_hidden_states": torch.arange(i * b, (i + 1) * b, dtype=torch.float32)
                    .view(b, 1, 1)
                    .repeat(1, 3, 2),
                    "base_hidden_prenorm": True,
                },
            }
            for i in range(n_batches)
        ]
        self.length = n_batches if length is None else length
        self.dataset = list(range(n_batches * b))
        self.epochs = []

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.batches)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


@pytest.mark.parametrize("store", [None, "cpu"])
@pytest.mark.parametrize(
    "plan", [BatchEchoPlan(3), ExampleEchoPlan(3, min_gap=2, capacity=8, seed=5)]
)
def test_echo_loader_yields_every_example_uses_times(plan, store):
    base = _Loader(n_batches=9, b=2)
    loader = EchoLoader(base, plan, batch_size=2, store=store, key_field="key", key_seed=9)
    loader.set_epoch(1)
    assert base.epochs == [1]
    assert loader.dataset is base.dataset
    out = list(loader)
    assert len(out) == len(loader) == plan.uses * 9
    seen, keys = Counter(), {}
    for batch in out:
        ids = batch["input_ids"][:, 0]
        # Every field of a row still belongs to the same example, and non-tensor fields survive.
        assert torch.equal(batch["base_model_outputs"]["aux_hidden_states"][:, 0, 0].long(), ids)
        assert batch["base_model_outputs"]["base_hidden_prenorm"] is True
        seen.update(ids.tolist())
        for i, (seed, use) in zip(ids.tolist(), batch["key"].tolist()):
            keys.setdefault(i, []).append((seed, use))
    assert seen == Counter(dict.fromkeys(range(18), plan.uses))
    # One seed per example, walked through use by use; seeds differ between examples.
    assert all(
        [u for _, u in k] == list(range(plan.uses)) and len({s for s, _ in k}) == 1
        for k in keys.values()
    )
    assert len({k[0][0] for k in keys.values()}) == 18


def test_echo_loader_batches_and_held_examples_are_copies():
    base = _Loader(n_batches=1, b=2)
    it = iter(EchoLoader(base, BatchEchoPlan(3), batch_size=2))
    next(it)["input_ids"].fill_(-1)  # a model editing its inputs in place
    base.batches[0]["input_ids"].add_(100)  # someone else editing the fetched batch
    assert next(it)["input_ids"][:, 0].tolist() == [0, 1]


def test_echo_loader_fails_loud_on_bad_loaders():
    with pytest.raises(RuntimeError):  # len() promised more batches than it yields
        list(EchoLoader(_Loader(n_batches=2, b=2, length=3), BatchEchoPlan(2), batch_size=2))
    with pytest.raises(ValueError):  # batches are not batch_size long
        list(EchoLoader(_Loader(n_batches=2, b=2), BatchEchoPlan(1), batch_size=3))
    mixed = _Loader(n_batches=2, b=2)
    mixed.batches[1]["base_model_outputs"]["base_hidden_prenorm"] = False
    with pytest.raises(ValueError):  # restacking would mislabel a per-batch field
        list(EchoLoader(mixed, _FixedPlan([(2, [0, 2])]), batch_size=2))


class _FixedPlan:
    """A hand-written plan, to put examples of different fresh batches in one step."""

    uses = 1

    def __init__(self, steps):
        self._steps = steps

    def num_steps(self, n_batches):
        return len(self._steps)

    def steps(self, n_batches, batch_size, epoch=0):
        return iter(self._steps)


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, i):
        return {"input_ids": torch.tensor([i]), "x": torch.full((2,), float(i))}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 1)
        self.seen = Counter()

    def forward(self, input_ids, x):
        self.seen.update(input_ids[:, 0].tolist())
        return {"loss": self.w(x).pow(2).mean()}


class _EchoTrainer(Trainer):
    def get_train_dataloader(self):
        plan = ExampleEchoPlan(4, min_gap=4, capacity=14, seed=1)
        return EchoLoader(
            super().get_train_dataloader(), plan, self.args.per_device_train_batch_size
        )


def test_hf_trainer_trains_every_echoed_step(tmp_path):
    """HF's loop sizes epochs and the LR schedule from len(loader) and drives set_epoch/iter."""
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_train_epochs=2,
        lr_scheduler_type="linear",
        dataloader_drop_last=True,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        disable_tqdm=True,
    )
    model = _Model()
    trainer = _EchoTrainer(model=model, args=args, train_dataset=_Dataset())
    trainer.train()
    assert trainer.state.global_step == 2 * 4 * 12
    assert model.seen == Counter(dict.fromkeys(range(24), 2 * 4))
    assert trainer.lr_scheduler.get_last_lr()[0] == 0.0
