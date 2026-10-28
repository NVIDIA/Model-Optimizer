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

The echo rule and its execution are kept apart, so a rule can change without touching the data
path, and is testable without data:

* an :class:`EchoPlan` decides which examples each step trains on and when to fetch, naming
  examples by their arrival ordinal on this rank; :class:`LaneEchoPlan` is the rule in use, and
  :func:`replay_echo_plan` checks any plan against the contract;
* :class:`EchoLoader` executes a plan over an iterable of collated batches: it pulls a fresh
  batch when the plan says so, holds each example until its last use, and stacks the examples a
  step needs.

Uses of one example stay on the rank that fetched it, so nothing is shared across processes.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["EchoLoader", "EchoPlan", "LaneEchoPlan", "replay_echo_plan"]


class EchoPlan(Protocol):
    """An echo rule: which held examples each step trains on, and when to fetch.

    Fresh batch ``j`` brings ordinals ``j * batch_size`` to ``(j + 1) * batch_size - 1``. A plan
    trains on ``batch_size`` distinct ordinals per step, each only after the batch that brings it
    was pulled, and on every ordinal exactly ``uses`` times over ``num_steps(n_batches)`` steps.
    """

    @property
    def uses(self) -> int:
        """Steps each example is trained on."""
        ...

    def num_steps(self, n_batches: int) -> int:
        """Steps in an epoch of ``n_batches`` fresh batches."""
        ...

    def steps(
        self, n_batches: int, batch_size: int, epoch: int = 0
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(fresh batches to pull before this step, ordinals this step trains on)``."""
        ...


@dataclass(frozen=True)
class LaneEchoPlan:
    """Train on each example ``uses`` times in a fixed order, at a fixed gap per batch slot.

    Slot ``i`` of every step (a lane) trains on example ``i`` of each fresh batch. In a lane with
    gap ``g``, coprime with ``uses``, use ``k`` of the lane's ``j``-th example falls on lane time
    ``uses * j + k * g``, so every lane time names exactly one (j, k); lane times that name no
    example, the first and last few of an epoch, are skipped. A lane thus uses each example
    exactly ``uses`` times, ``g`` steps apart away from the epoch ends, and needs one fresh
    example every ``uses`` steps, so fetching is steady rather than bursty.

    Lanes take different gaps from ``[min_gap, min_gap + gap_spread)``, dealt by rank and slot, so
    they drift apart and an example meets different examples at each of its uses. ``min_gap=1``
    repeats each fresh batch ``uses`` times in a row (batch echoing). A rank holds about
    ``batch_size * (uses - 1) / uses * gap`` examples; :func:`replay_echo_plan` gives the peak.
    """

    uses: int
    min_gap: int = 32
    gap_spread: int | None = None  # None: min_gap // 4
    rank: int = 0

    def __post_init__(self):
        if self.uses < 1:
            raise ValueError(
                f"uses={self.uses}: every fetched example must be trained on at least once."
            )
        if (
            self.min_gap < 1
            or (self.gap_spread is not None and self.gap_spread < 0)
            or self.rank < 0
        ):
            raise ValueError(
                f"min_gap={self.min_gap} must be >= 1, gap_spread={self.gap_spread} and "
                f"rank={self.rank} >= 0."
            )

    def lane_gaps(self, batch_size: int) -> list[int]:
        """The gap of each lane on this rank."""
        spread = self.min_gap // 4 if self.gap_spread is None else self.gap_spread
        window = range(self.min_gap, self.min_gap + spread)
        gaps = [g for g in window if math.gcd(g, self.uses) == 1] or [
            next(g for g in itertools.count(self.min_gap) if math.gcd(g, self.uses) == 1)
        ]
        # A rank's lanes sit evenly across the window and each rank shifts them by one, so lanes
        # differ within a rank and across ranks, which also staggers the steps on which ranks fetch.
        return [
            gaps[(self.rank + i * len(gaps) // batch_size) % len(gaps)] for i in range(batch_size)
        ]

    def num_steps(self, n_batches: int) -> int:
        """Steps in an epoch of ``n_batches`` fresh batches."""
        return self.uses * n_batches

    def steps(
        self, n_batches: int, batch_size: int, epoch: int = 0
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(fresh batches to pull before this step, ordinals this step trains on)``.

        The order is the same every epoch; the wrapped loader's shuffling varies the examples.
        """
        pulled = 0
        lanes = (self._lane(gap, n_batches) for gap in self.lane_gaps(batch_size))
        for batches in zip(*lanes):
            n_pull = max(0, max(batches) + 1 - pulled)
            pulled += n_pull
            yield n_pull, [j * batch_size + i for i, j in enumerate(batches)]

    def _lane(self, gap: int, n_batches: int) -> Iterator[int]:
        """The fresh batch whose example one lane trains on, step by step."""
        inverse = pow(gap, -1, self.uses)
        for t in range(self.uses * (n_batches - 1) + (self.uses - 1) * gap + 1):
            use = t * inverse % self.uses
            j = (t - use * gap) // self.uses
            if 0 <= j < n_batches:
                yield j


def replay_echo_plan(plan: EchoPlan, n_batches: int, batch_size: int) -> dict[str, float]:
    """Replay ``plan`` without data, check it keeps the :class:`EchoPlan` contract, and summarize it.

    Returns the epoch's steps, the most examples held at once, and the smallest and mean number
    of steps between two uses of an example.
    """
    used: dict[int, int] = {}
    last: dict[int, int] = {}
    gaps: list[int] = []
    arrived = retired = max_held = n_steps = 0
    for t, (n_pull, ords) in enumerate(plan.steps(n_batches, batch_size)):
        arrived += n_pull * batch_size
        if len(set(ords)) != batch_size or max(ords) >= arrived:
            raise ValueError(
                f"step {t} trains on {ords}: not {batch_size} distinct arrived examples."
            )
        max_held = max(max_held, arrived - retired)
        for o in ords:
            if o in last:
                gaps.append(t - last[o])
            last[o] = t
            used[o] = used.get(o, 0) + 1
            retired += used[o] == plan.uses
        n_steps = t + 1
    if n_steps != plan.num_steps(n_batches) or arrived != n_batches * batch_size:
        raise ValueError(f"the plan ran {n_steps} steps over {arrived // batch_size} batches.")
    if any(u != plan.uses for u in used.values()) or len(used) != arrived:
        raise ValueError(f"the plan does not train on every example exactly {plan.uses} times.")
    return {
        "steps": n_steps,
        "max_held": max_held,
        "min_gap": min(gaps, default=0),
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
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
        plan: EchoPlan,
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
        """Forward the epoch to the plan and the wrapped loader."""
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
