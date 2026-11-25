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

"""Exact size reduction for the joint mask-reuse topology program.

Only two families of rows couple the layers of the joint program: the worst
prompt maximum and one work row per context bucket. Put a nonnegative weight on
every calibration prompt (the weights sum to one) and a nonnegative price on
every bucket's work row. Each consumer head then chooses its donor on its own,
a layer's cost depends only on its nearest anchor, and the cheapest anchor
layout is a shortest path over the layers. That priced value is a lower bound on
the worst-prompt risk of every policy that meets the work targets.

The same prices bound each variable: the cheapest priced policy that uses one
edge or one donor option cannot cost more than any real policy that uses it. A
variable whose bound exceeds the risk of a verified policy therefore appears in
no policy that is at least as good, and the integer program may omit it without
changing its optimum. Nothing here chooses a policy; the integer program and
the exact replay still do.

The cost pass has the same shape: tile cost is the objective, every prompt's
risk row gets a nonnegative price against the shared risk ceiling, and the
priced value bounds the cost of every policy within that ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__: list[str] = []

# Reported bounds give up this relative margin to floating-point summation.
_BOUND_RELATIVE_MARGIN = 1e-12


@dataclass(frozen=True)
class _BucketTables:
    """Dropped mass and skipped work of one context bucket at one target sparsity."""

    # (prompts, layers, span, heads, donors): [p, s, k, h, d] is the mass dropped
    # when consumer head h of layer s + k + 1 reuses donor d of anchor layer s.
    risk: np.ndarray
    # (prompts, layers): mass dropped by layer s itself when it is an anchor.
    anchor: np.ndarray
    # (layers, donors): tiles a consumer head skips by reusing donor d of layer s.
    skipped: np.ndarray
    required: int
    # Eligible tiles of one head, summed over the calibration prompts.
    eligible: int


@dataclass(frozen=True)
class _Prices:
    weights: tuple[np.ndarray, ...]
    work: np.ndarray
    bound: float
    edges: tuple[tuple[int, int], ...]
    rounds: int
    master_gap: float
    stopped_above: bool


@dataclass(frozen=True)
class _Reduction:
    edges: np.ndarray
    pairs: np.ndarray
    donors: tuple[np.ndarray, ...]
    fallbacks: tuple[np.ndarray, ...]


def _lower(value: float) -> float:
    return value - _BOUND_RELATIVE_MARGIN * max(1.0, abs(value))


def _clean(point: np.ndarray, groups: Sequence[int], limits: np.ndarray) -> np.ndarray:
    """Project a master solution onto valid prices; the bound holds only there."""
    cleaned = np.array(point, dtype=np.float64)
    offset = 0
    for size in groups:
        weights = np.clip(cleaned[offset : offset + size], 0.0, None)
        total = weights.sum()
        cleaned[offset : offset + size] = weights / total if total > 0 else 1.0 / size
        offset += size
    cleaned[offset:] = np.clip(cleaned[offset:], 0.0, limits)
    return cleaned


def _maximize_concave(
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray, object]],
    groups: Sequence[int],
    limits: np.ndarray,
    *,
    start: np.ndarray,
    tolerance: float,
    max_rounds: int,
    unit: float = 1.0,
    stop_above: float | None = None,
    expired: Callable[[], bool] | None = None,
):
    """Cutting-plane ascent of a concave piecewise-linear function of the prices.

    The first ``sum(groups)`` prices are weights that sum to one within each
    group; the rest lie between zero and ``limits``. The master program sees
    the boxed prices as fractions of their limits and the value in ``unit``.
    """
    from scipy.optimize import linprog

    size = sum(groups) + len(limits)
    scale = np.concatenate((np.ones(sum(groups)), limits))
    equalities = np.zeros((len(groups), size + 1))
    offset = 1
    for row, count in enumerate(groups):
        equalities[row, offset : offset + count] = 1.0
        offset += count
    bounds = [(None, None)] + [(0.0, 1.0)] * size
    objective = np.zeros(size + 1)
    objective[0] = -1.0
    point = _clean(start, groups, limits)
    cut_rows: list[np.ndarray] = []
    cut_bounds: list[float] = []
    best: tuple[float, np.ndarray, object] | None = None
    upper = math.inf
    rounds = 0
    stopped = False
    for _ in range(max_rounds):
        value, gradient, detail = evaluate(point)
        rounds += 1
        if best is None or value > best[0]:
            best = (value, point.copy(), detail)
        # Compare what will be reported, after the floating-point margin.
        if stop_above is not None and _lower(best[0]) > stop_above:
            stopped = True
            break
        if upper - best[0] <= tolerance or (expired is not None and expired()):
            break
        cut_rows.append(np.concatenate(([1.0], -gradient * scale / unit)))
        cut_bounds.append((value - float(gradient @ point)) / unit)
        result = linprog(
            objective,
            A_ub=np.asarray(cut_rows),
            b_ub=np.asarray(cut_bounds),
            A_eq=equalities if groups else None,
            b_eq=np.ones(len(groups)) if groups else None,
            bounds=bounds,
            method="highs",
        )
        if result.status != 0 or result.x is None:
            break
        upper = min(upper, -float(result.fun) * unit)
        if upper - best[0] <= tolerance:
            break
        point = _clean(result.x[1:] * scale, groups, limits)
    assert best is not None
    return best, upper, rounds, stopped


class _PricedRelaxation:
    """Priced relaxation of the joint program for one target per context bucket.

    Without ``ceiling`` the value bounds the worst-prompt risk. With it, the
    value bounds the tile cost of every policy whose risk is at most ``ceiling``.
    ``keep`` restricts the relaxation to the variables of an earlier reduction.
    """

    def __init__(
        self,
        tables: Sequence[_BucketTables],
        edge_allowed: np.ndarray,
        donor_allowed: np.ndarray | None = None,
        ceiling: float | None = None,
        keep: _Reduction | None = None,
    ) -> None:
        self.tables = tuple(tables)
        self.keep = keep
        # [s, j] is the edge from anchor layer s to the next anchor s + j + 1.
        self.edge_allowed = edge_allowed if keep is None else edge_allowed & keep.edges
        self.layers, width = edge_allowed.shape
        self.span = width - 1
        self.donor_allowed = donor_allowed
        self.ceiling = ceiling
        self.priced = tuple(index for index, table in enumerate(self.tables) if table.required > 0)
        self.prompt_counts = tuple(table.risk.shape[0] for table in self.tables)
        self.heads = self.tables[0].risk.shape[3]
        prompts = sum(self.prompt_counts)
        # Dropped mass per head is at most one, so no useful price is larger.
        self.work_limit = 16.0 * self.layers * self.heads
        if ceiling is None:
            self.tile, self.unit, self.groups = 0.0, 1.0, (prompts,)
            self.limits = np.full(len(self.priced), self.work_limit)
        else:
            # Tile cost of exact attention in every layer scales the cost prices.
            dense = 2.0 * self.layers * self.heads * sum(table.eligible for table in self.tables)
            self.tile, self.unit, self.groups = 2.0, max(dense, 1.0), ()
            self.limits = (
                64.0
                * self.unit
                * np.concatenate(
                    (
                        np.full(prompts, 1.0 / max(ceiling, 1e-9 * self.layers * self.heads)),
                        np.ones(len(self.priced)),
                    )
                )
            )

    def start(self) -> np.ndarray:
        prompts = sum(self.prompt_counts)
        if self.ceiling is not None:
            return np.concatenate(
                (self.limits[:prompts] / (64.0 * prompts), self.limits[prompts:] * 0)
            )
        return np.concatenate(
            (
                np.full(prompts, 1.0 / prompts),
                np.full(len(self.priced), 0.05 * self.layers * self.heads),
            )
        )

    def point(self, prices: _Prices) -> np.ndarray:
        return np.concatenate((*prices.weights, prices.work[list(self.priced)]))

    def most_work_edges(self) -> tuple[tuple[int, int], ...]:
        """Anchor layout favoured when skipped work outweighs any dropped mass."""
        point = self.start()
        point[sum(self.prompt_counts) :] = self.work_limit
        return self._evaluate(point)[2]

    def connected(self) -> bool:
        """Whether the allowed edges still contain a complete anchor path."""
        zero = np.zeros((self.layers, self.span + 1))
        return math.isfinite(self._forward(zero)[0][self.layers])

    def _offset(self, weights, work) -> float:
        value = float(sum(work[bucket] for bucket in self.priced))
        if self.ceiling is not None:
            value -= self.ceiling * float(sum(weight.sum() for weight in weights))
        return value

    def _split(self, point: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        weights = []
        offset = 0
        for count in self.prompt_counts:
            weights.append(point[offset : offset + count])
            offset += count
        work = np.zeros(len(self.tables))
        for position, bucket in enumerate(self.priced):
            work[bucket] = point[offset + position]
        return weights, work

    def _costs(self, weights, work, keep_values: bool = False):
        pair = np.zeros((self.layers, self.span))
        anchor = np.zeros(self.layers)
        picks = []
        for index, (table, weight, price) in enumerate(zip(self.tables, weights, work)):
            values = np.tensordot(weight, table.risk, axes=1)
            # Each skipped tile saves two tiles of cost and earns the work price.
            rate = self.tile + (price / table.required if price else 0.0)
            if rate:
                values -= rate * table.skipped[:, None, None, :]
            if self.donor_allowed is not None:
                values[:, :, ~self.donor_allowed] = np.inf
            if self.keep is not None:
                values[~self.keep.donors[index]] = np.inf
            donor = values.argmin(axis=3)
            best = np.take_along_axis(values, donor[..., None], axis=3)[..., 0]
            # Values are relative to exact fallback: no mass and no skipped work.
            reused = best < 0
            if self.keep is not None:
                reused |= ~self.keep.fallbacks[index]
            best = np.where(reused, best, 0.0)
            pair += best.sum(axis=2) + self.tile * table.eligible * self.heads
            anchor += weight @ table.anchor
            # An anchor layer costs eligible plus retained tiles for every head.
            anchor += (
                0.5 * self.tile * (2 * table.eligible * self.heads - table.skipped.sum(axis=1))
            )
            picks.append((donor, reused, best, values if keep_values else None))
        edge = anchor[:, None] + np.concatenate(
            (np.zeros((self.layers, 1)), np.cumsum(pair, axis=1)), axis=1
        )
        return edge, picks

    def _forward(self, edge: np.ndarray) -> tuple[list[float], list[int]]:
        costs, allowed = edge.tolist(), self.edge_allowed.tolist()
        best = [math.inf] * (self.layers + 1)
        previous = [-1] * (self.layers + 1)
        best[0] = 0.0
        for stop in range(1, self.layers + 1):
            for start in range(max(0, stop - self.span - 1), stop):
                if not allowed[start][stop - start - 1]:
                    continue
                value = best[start] + costs[start][stop - start - 1]
                # Strict comparison keeps the earliest start among equal paths.
                if value < best[stop]:
                    best[stop], previous[stop] = value, start
        return best, previous

    def _backward(self, edge: np.ndarray) -> list[float]:
        costs, allowed = edge.tolist(), self.edge_allowed.tolist()
        best = [math.inf] * (self.layers + self.span + 2)
        best[self.layers] = 0.0
        for start in range(self.layers - 1, -1, -1):
            for extra in range(min(self.span, self.layers - start - 1) + 1):
                if allowed[start][extra]:
                    best[start] = min(best[start], costs[start][extra] + best[start + extra + 1])
        return best

    def _evaluate(self, point: np.ndarray, fixed=None):
        weights, work = self._split(point)
        edge, picks = self._costs(weights, work)
        if fixed is None:
            best, previous = self._forward(edge)
            value = best[self.layers]
            edges = []
            stop = self.layers
            while stop > 0 and previous[stop] >= 0:
                edges.append((previous[stop], stop))
                stop = previous[stop]
            edges.reverse()
            if not math.isfinite(value):
                raise ValueError("no anchor path is available to the priced relaxation")
        else:
            edges = list(fixed)
            value = float(sum(edge[start, stop - start - 1] for start, stop in edges))
        starts = [start for start, _ in edges]
        anchors = np.asarray(starts, dtype=np.intp)
        layer = np.asarray(
            [start for start, stop in edges for _ in range(stop - start - 1)], dtype=np.intp
        )
        offset = np.asarray(
            [extra for start, stop in edges for extra in range(stop - start - 1)], dtype=np.intp
        )
        heads = np.arange(self.heads)
        risks, works = [], []
        for table, (donor, reused, _, _) in zip(self.tables, picks):
            risk = table.anchor[:, anchors].sum(axis=1)
            skipped = 0.0
            if len(layer):
                chosen, used = donor[layer, offset], reused[layer, offset]
                picked = table.risk[:, layer[:, None], offset[:, None], heads[None, :], chosen]
                risk = risk + (picked * used).sum(axis=(1, 2))
                skipped = float((table.skipped[layer[:, None], chosen] * used).sum())
            risks.append(risk if self.ceiling is None else risk - self.ceiling)
            works.append(1.0 - skipped / table.required if table.required else 0.0)
        value += self._offset(weights, work)
        gradient = np.concatenate((*risks, [works[bucket] for bucket in self.priced]))
        return value, gradient, tuple(edges)

    def maximize(
        self,
        *,
        start: np.ndarray | None = None,
        tolerance: float,
        max_rounds: int,
        stop_above: float | None = None,
        expired: Callable[[], bool] | None = None,
    ) -> _Prices:
        """Raise the lower bound; any returned prices give a valid bound."""
        best, upper, rounds, stopped = _maximize_concave(
            self._evaluate,
            self.groups,
            self.limits,
            start=self.start() if start is None or len(start) != len(self.start()) else start,
            tolerance=tolerance,
            max_rounds=max_rounds,
            unit=self.unit,
            stop_above=stop_above,
            expired=expired,
        )
        value, point, edges = best
        weights, work = self._split(point)
        return _Prices(
            weights=tuple(np.array(weight) for weight in weights),
            work=work,
            bound=_lower(value),
            edges=edges,
            rounds=rounds,
            master_gap=max(0.0, upper - value),
            stopped_above=stopped,
        )

    def reduce(self, prices: _Prices, threshold: float) -> _Reduction:
        """Keep every variable some policy with value at most ``threshold`` can use."""
        edge, picks = self._costs(prices.weights, prices.work, keep_values=True)
        forward, _ = self._forward(edge)
        backward = np.asarray(self._backward(edge))
        offset = self._offset(prices.weights, prices.work)
        stops = np.arange(self.layers)[:, None] + np.arange(self.span + 1)[None, :] + 1
        with np.errstate(invalid="ignore"):
            through = np.asarray(forward[: self.layers])[:, None] + edge + backward[stops]
            through = np.where(self.edge_allowed, through + offset, np.inf)
        edges = through <= threshold
        covered = np.where(edges, through, np.inf)
        # Consumer s + k + 1 is served by anchor s on every edge that ends later.
        pair = np.minimum.accumulate(covered[:, ::-1], axis=1)[:, ::-1][:, 1:]
        room = threshold - pair
        donors, fallbacks = [], []
        for index, (_, _, best, values) in enumerate(picks):
            with np.errstate(invalid="ignore"):
                donors.append(values - best[..., None] <= room[..., None, None])
            fallback = -best <= room[..., None]
            if self.keep is not None:
                fallback &= self.keep.fallbacks[index]
            fallbacks.append(fallback)
        return _Reduction(edges, np.isfinite(pair), tuple(donors), tuple(fallbacks))


def _fixed_layout_choice(
    risk: np.ndarray,
    skipped: np.ndarray,
    valid: np.ndarray,
    base: np.ndarray,
    required: int,
    *,
    work_limit: float,
    tolerance: float,
    max_rounds: int,
    sweeps: int,
    expired: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """Choose one option per head of a fixed layout for one context bucket.

    ``risk`` is (prompts, heads, options), ``skipped`` and ``valid`` are
    (heads, options) and ``base`` is the anchors' mass per prompt. The result
    meets the work target and has a small worst-prompt mass, or is None when
    the layout cannot meet the target. It is a starting policy, not a proof.
    """
    prompts, heads, _ = risk.shape
    rows = np.arange(heads)
    priced_work = 1 if required > 0 else 0

    def priced(weights, price):
        values = np.tensordot(weights, risk, axes=1)
        if price:
            values = values - (price / required) * skipped
        return np.where(valid, values, np.inf)

    def totals(choice):
        return base + risk[:, rows, choice].sum(axis=1), int(skipped[rows, choice].sum())

    def evaluate(point):
        weights = point[:prompts]
        price = float(point[prompts]) if priced_work else 0.0
        values = priced(weights, price)
        choice = values.argmin(axis=1)
        risks, done = totals(choice)
        value = float(weights @ base + values[rows, choice].sum() + price)
        gradient = np.concatenate((risks, [1.0 - done / required])) if priced_work else risks
        return value, gradient, choice

    start = np.concatenate((np.full(prompts, 1.0 / prompts), np.full(priced_work, 1.0)))
    (_, point, choice), _, _, _ = _maximize_concave(
        evaluate,
        [prompts],
        np.full(priced_work, work_limit),
        start=start,
        tolerance=tolerance,
        max_rounds=max_rounds,
        expired=expired,
    )
    weights = point[:prompts]
    risks, done = totals(choice)
    if done < required:
        most = np.where(valid, skipped, -1).argmax(axis=1)
        if int(skipped[rows, most].sum()) < required:
            return None
        # A higher work price only moves heads toward options that skip more.
        low = float(point[prompts])
        high = max(2.0 * low, 1e-9)
        candidate = priced(weights, high).argmin(axis=1)
        for _ in range(200):
            if totals(candidate)[1] >= required:
                break
            low, high = high, 2.0 * high
            candidate = priced(weights, high).argmin(axis=1)
        else:
            candidate = most
        for _ in range(60):
            middle = 0.5 * (low + high)
            trial = priced(weights, middle).argmin(axis=1)
            if totals(trial)[1] >= required:
                high, candidate = middle, trial
            else:
                low = middle
        choice = candidate
        risks, done = totals(choice)
        if done < required:
            return None
    choice = np.array(choice)
    for _ in range(sweeps):
        changed = False
        for head in range(heads):
            current = choice[head]
            moved = risks[:, None] - risk[:, head, current][:, None] + risk[:, head, :]
            peak = moved.max(axis=0)
            allowed = valid[head] & (done - skipped[head, current] + skipped[head] >= required)
            peak = np.where(allowed, peak, np.inf)
            option = int(peak.argmin())
            if peak[option] < risks.max():
                done += int(skipped[head, option] - skipped[head, current])
                risks = moved[:, option]
                choice[head] = option
                changed = True
        if not changed or (expired is not None and expired()):
            break
    return choice
