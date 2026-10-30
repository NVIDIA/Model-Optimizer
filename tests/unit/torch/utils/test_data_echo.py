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

from modelopt.torch.utils import EchoLoader, LaneEchoPlan, replay_echo_plan


@pytest.mark.parametrize(
    ("uses", "b", "min_gap", "n_batches", "rank"),
    [(1, 2, 4, 3, 0), (4, 2, 1, 5, 0), (4, 2, 32, 97, 3), (3, 3, 7, 25, 1), (4, 1, 9, 6, 2)],
)
def test_lane_plan_keeps_the_contract(uses, b, min_gap, n_batches, rank):
    stats = replay_echo_plan(LaneEchoPlan(uses, min_gap, rank=rank), n_batches, b)
    assert stats["steps"] == uses * n_batches


def test_lane_plan_reuses_at_fixed_gaps_and_fetches_steadily():
    """Away from the epoch ends, each lane reuses at its own gap and the rank fetches every 4 steps."""
    uses, b = 4, 2
    plan = LaneEchoPlan(uses, min_gap=32, rank=3)
    gaps = plan.lane_gaps(b)
    assert len(set(gaps)) == b and all(32 <= g < 40 and g % 2 for g in gaps)
    steps = list(plan.steps(400, b))
    edge = uses * max(gaps)
    last = {}
    for t, (_, ords) in enumerate(steps):
        for lane, o in enumerate(ords):
            if o in last and edge <= t < len(steps) - edge:
                assert t - last[o] == gaps[lane]
            last[o] = t
    pulls = [n for n, _ in steps[edge:-edge]]
    assert all(sum(pulls[t : t + uses]) == 1 for t in range(len(pulls) - uses))


def test_lane_plan_meets_a_new_partner_at_each_use():
    """The lanes of a rank drift apart, and ranks deal the gaps differently."""
    plan = LaneEchoPlan(4, min_gap=32)
    assert plan.lane_gaps(2) != LaneEchoPlan(4, min_gap=32, rank=1).lane_gaps(2)
    steps = list(plan.steps(400, 2))
    partner = {}
    for t, (_, ords) in enumerate(steps[160:-160]):
        for o, other in ((ords[0], ords[1]), (ords[1], ords[0])):
            assert partner.get(o) != other
            partner[o] = other


def test_lane_plan_with_min_gap_1_repeats_each_batch_in_a_row():
    steps = list(LaneEchoPlan(3, min_gap=1).steps(4, 2))
    assert steps == [(t % 3 == 0, [2 * (t // 3), 2 * (t // 3) + 1]) for t in range(12)]


def test_replay_rejects_plans_that_break_the_contract():
    with pytest.raises(ValueError):  # trains on an example before its batch is pulled
        replay_echo_plan(_FixedPlan([(0, [0, 1])]), 1, 2)
    with pytest.raises(ValueError):  # the same example twice in one step
        replay_echo_plan(_FixedPlan([(1, [0, 0])]), 1, 2)
    with pytest.raises(ValueError):  # more uses than the plan declares
        replay_echo_plan(_FixedPlan([(1, [0, 1]), (0, [0, 1])]), 1, 2)


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
    "plan", [LaneEchoPlan(3, min_gap=1), LaneEchoPlan(3, min_gap=2, gap_spread=4, rank=1)]
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
    it = iter(EchoLoader(base, LaneEchoPlan(3, min_gap=1), batch_size=2))
    next(it)["input_ids"].fill_(-1)  # a model editing its inputs in place
    base.batches[0]["input_ids"].add_(100)  # someone else editing the fetched batch
    assert next(it)["input_ids"][:, 0].tolist() == [0, 1]


def test_echo_loader_fails_loud_on_bad_loaders():
    with pytest.raises(RuntimeError):  # len() promised more batches than it yields
        list(EchoLoader(_Loader(n_batches=2, b=2, length=3), LaneEchoPlan(2, 1), batch_size=2))
    with pytest.raises(ValueError):  # batches are not batch_size long
        list(EchoLoader(_Loader(n_batches=2, b=2), LaneEchoPlan(1), batch_size=3))
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
        plan = LaneEchoPlan(4, min_gap=4)
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
