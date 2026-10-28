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

"""Data echoing: train on each fetched example for several steps.

When fetching an example is the bottleneck (hidden states streamed from an inference server,
large dumps on a slow filesystem), each example can serve ``uses`` steps instead of one
(`Choi et al., 2019 <https://arxiv.org/abs/1907.05550>`_). Per-step randomness in the model,
for example a fresh draw of training positions, makes every use a different example.

The decision and its execution are kept apart, so the decision is testable without data:

* a plan (:class:`ExampleEchoPlan`, :class:`BatchEchoPlan`) decides from a seed alone which
  examples each step trains on, naming examples by their arrival ordinal on this rank;
* :class:`EchoLoader` executes a plan over an iterable of collated batches: it pulls a fresh
  batch when the plan says so, holds each example until its last use, and stacks the examples a
  step needs.

Uses of one example stay on the rank that fetched it, so nothing is shared across processes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["BatchEchoPlan", "EchoLoader", "ExampleEchoPlan"]


@dataclass(frozen=True)
class BatchEchoPlan:
    """Train on each fresh batch ``uses`` times in a row ("batch echoing").

    ``uses=1`` is the plain loader: one fresh batch per step.
    """

    uses: int = 1

    def __post_init__(self):
        if self.uses < 1:
            raise ValueError(
                f"uses={self.uses}: every fetched example must be trained on at least once."
            )

    def num_steps(self, n_batches: int) -> int:
        """Steps in an epoch of ``n_batches`` fresh batches."""
        return self.uses * n_batches

    def steps(
        self, n_batches: int, batch_size: int, epoch: int = 0
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(fresh batches to pull before this step, ordinals this step trains on)``."""
        for t in range(self.num_steps(n_batches)):
            first = (t // self.uses) * batch_size
            yield (1 if t % self.uses == 0 else 0), list(range(first, first + batch_size))


@dataclass(frozen=True)
class ExampleEchoPlan:
    """Train on each example ``uses`` times, spread over steps and mixed with other examples.

    Each step trains on ``batch_size`` distinct examples. An example's uses are at least
    ``min_gap`` steps apart; a step picks at random among the ``pool`` + needed examples that
    have waited longest, and a fresh batch is pulled whenever fewer than that many are ready.
    At most ``capacity`` examples are held at once.

    Every example gets exactly ``uses`` steps and the epoch has exactly ``uses * n_batches``
    steps. Near the epoch end (or when ``capacity`` binds) an example may be reused sooner than
    ``min_gap``; :meth:`stats` counts those reuses.
    """

    uses: int
    min_gap: int = 32
    capacity: int = 70
    pool: int = 2
    seed: int = 0

    def __post_init__(self):
        if self.uses < 1:
            raise ValueError(
                f"uses={self.uses}: every fetched example must be trained on at least once."
            )
        if self.min_gap < 1 or self.pool < 0:
            raise ValueError(f"min_gap={self.min_gap} must be >= 1 and pool={self.pool} >= 0.")

    def num_steps(self, n_batches: int) -> int:
        """Steps in an epoch of ``n_batches`` fresh batches."""
        return self.uses * n_batches

    def steps(
        self, n_batches: int, batch_size: int, epoch: int = 0
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(fresh batches to pull before this step, ordinals this step trains on)``."""
        b = batch_size
        if self.capacity < 2 * b:
            # With fewer than 2 batches of room a step could need a fresh batch while the held
            # examples cannot fill it, and the plan would stall mid-epoch.
            raise ValueError(
                f"capacity={self.capacity} must hold at least two batches ({2 * b} examples)."
            )
        rng = random.Random(self.seed * 1_000_003 + epoch)
        left: dict[int, int] = {}  # ordinal -> uses left
        ready_at: dict[int, int] = {}  # ordinal -> first step it may be used again
        pulled = 0
        total = self.num_steps(n_batches)
        for t in range(total):
            steps_left = total - t
            n_pull = 0
            while (
                pulled < n_batches
                and len(left) + b <= self.capacity
                and sum(ready_at[o] <= t for o in left) < b + self.pool
            ):
                for o in range(pulled * b, (pulled + 1) * b):
                    left[o], ready_at[o] = self.uses, t
                pulled += 1
                n_pull += 1
            # An example with as many uses left as steps left must be used now, or the epoch
            # cannot finish with distinct examples per step. Uses left always total b * steps_left,
            # so at most b examples are ever in that state.
            chosen = [o for o in left if left[o] == steps_left]
            ready = sorted(
                (o for o in left if ready_at[o] <= t and o not in chosen),
                key=lambda o: (ready_at[o], o),
            )
            need = b - len(chosen)
            if len(ready) >= need:
                chosen += rng.sample(ready[: need + self.pool], need)
            else:
                waiting = sorted(
                    (o for o in left if o not in chosen and o not in ready),
                    key=lambda o: (ready_at[o], o),
                )
                chosen += ready + waiting[: need - len(ready)]
            if len(set(chosen)) != b:
                raise RuntimeError(
                    f"step {t}: plan picked {len(set(chosen))} distinct examples, need {b}; the held "
                    "uses cannot cover the remaining steps, so the epoch length would be wrong."
                )
            for o in chosen:
                left[o] -= 1
                ready_at[o] = t + self.min_gap
                if left[o] == 0:
                    del left[o], ready_at[o]
            yield n_pull, chosen

    def stats(self, n_batches: int, batch_size: int, epoch: int = 0) -> dict[str, float]:
        """Replay the plan without data: examples held at once, gaps between uses, early reuses."""
        used: dict[int, int] = {}
        last: dict[int, int] = {}
        gaps: list[int] = []
        arrived = retired = max_held = 0
        for t, (n_pull, ords) in enumerate(self.steps(n_batches, batch_size, epoch)):
            arrived += n_pull * batch_size
            max_held = max(max_held, arrived - retired)
            for o in ords:
                if o in last:
                    gaps.append(t - last[o])
                last[o] = t
                used[o] = used.get(o, 0) + 1
                retired += used[o] == self.uses
        return {
            "steps": self.num_steps(n_batches),
            "max_held": max_held,
            "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
            "min_gap": min(gaps) if gaps else 0,
            "early_reuses": sum(g < self.min_gap for g in gaps),
        }


def _leaves(x: Any) -> Iterator[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)


def _unbatch(batch: Any, i: int, device) -> Any:
    """Example ``i`` of a collated batch, copied (to ``device`` if given) so it does not pin the whole batch."""
    if isinstance(batch, torch.Tensor):
        return batch[i].clone() if device is None else batch[i].to(device, copy=True)
    if isinstance(batch, dict):
        return {k: _unbatch(v, i, device) for k, v in batch.items()}
    return batch


def _stack(examples: list[Any]) -> Any:
    """Collate per-example structures back into a batch (a copy, so the held examples cannot be edited)."""
    first = examples[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(examples)
    if isinstance(first, dict):
        return {k: _stack([e[k] for e in examples]) for k in first}
    if any(e != first for e in examples[1:]):
        raise ValueError(
            f"non-tensor batch field differs across examples ({first!r} vs others); the collator "
            "declared it per batch, so restacking examples from different batches would mislabel some."
        )
    return first


def _to(x: Any, device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x


class EchoLoader:
    """Iterable over collated batches that trains on each fetched example according to a plan.

    Wraps ``loader`` (any sized iterable of collated batches, e.g. an accelerate-prepared
    DataLoader). One epoch yields ``plan.num_steps(len(loader))`` batches of ``batch_size``.
    """

    def __init__(
        self,
        loader,
        plan: ExampleEchoPlan | BatchEchoPlan,
        batch_size: int,
        *,
        store: str | None = None,
        key_field: str | None = None,
        key_seed: int = 0,
    ):
        """Wrap ``loader``; ``batch_size`` is the per-rank batch it yields.

        ``store``: device that holds examples between uses (e.g. ``"cpu"`` when they would not
        fit next to training); None keeps them where the loader put them. ``key_field``: if set,
        each batch gains this field, a ``[batch, 2]`` int64 of (per-example seed, uses so far),
        from which the consumer can derive per-use randomness that differs across the uses of one
        example, e.g. consecutive slices of one random order. ``key_seed`` should differ per rank.
        """
        self.loader = loader
        self.plan = plan
        self.batch_size = batch_size
        self.store = store
        self.key_field = key_field
        self.key_seed = key_seed
        self.epoch = 0

    @property
    def dataset(self):
        """The wrapped loader's dataset (the unique examples, not the echoed ones)."""
        return self.loader.dataset

    def __len__(self) -> int:
        return self.plan.num_steps(len(self.loader))

    def set_epoch(self, epoch: int) -> None:
        """Reseed the plan and forward the epoch to the wrapped loader."""
        self.epoch = epoch
        if hasattr(self.loader, "set_epoch"):
            self.loader.set_epoch(epoch)

    def __iter__(self) -> Iterator[Any]:
        b, n_batches = self.batch_size, len(self.loader)
        fresh = iter(self.loader)
        held: dict[int, list] = {}  # ordinal -> [example, uses left, tensor versions when fetched]
        arrived = 0
        for n_pull, ords in self.plan.steps(n_batches, b, self.epoch):
            for _ in range(n_pull):
                try:
                    batch = next(fresh)
                except StopIteration:
                    raise RuntimeError(
                        f"the wrapped loader ran out after {arrived // b} of the {n_batches} batches its "
                        "len() promised; the plan would train on examples that never arrived."
                    ) from None
                rows = {t.shape[0] for t in _leaves(batch)}
                if rows != {b}:
                    raise ValueError(
                        f"batch has leading dims {sorted(rows)}, expected {b}: the plan assigns ordinals "
                        "per batch_size examples, so a short batch would shift every later ordinal."
                    )
                device = next(_leaves(batch)).device
                for i in range(b):
                    example = _unbatch(batch, i, self.store)
                    held[arrived] = [
                        example,
                        self.plan.uses,
                        [t._version for t in _leaves(example)],
                    ]
                    arrived += 1
            for o in ords:
                if [t._version for t in _leaves(held[o][0])] != held[o][2]:
                    raise RuntimeError(
                        f"held example {o} was modified in place since it was fetched; its later uses "
                        "would train on the modified values."
                    )
            examples = [held[o][0] for o in ords]
            if self.store is not None:
                # Stack on the device: a host-side stack copies the whole batch into fresh pageable memory.
                examples = [_to(e, device) for e in examples]
            out = _stack(examples)
            if self.key_field is not None:
                out[self.key_field] = torch.tensor(
                    [
                        [(self.key_seed * 1_000_003 + o) & 0x7FFFFFFF, self.plan.uses - held[o][1]]
                        for o in ords
                    ],
                    dtype=torch.int64,
                    device=device,
                )
            yield out
            for o in ords:
                held[o][1] -= 1
                if held[o][1] == 0:
                    del held[o]
        if held:
            raise RuntimeError(
                f"{len(held)} examples still held at the end of the epoch; the plan dropped uses."
            )
