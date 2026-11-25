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

"""Priced bounds and reductions of the joint mask-reuse program against enumeration."""

import itertools
import math

import numpy as np
import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_reduction import (
    _BucketTables,
    _fixed_layout_choice,
    _PricedRelaxation,
)

LAYERS, HEADS, SPAN, PROMPTS = 4, 2, 2, 2
EDGES = np.array(
    [[start + extra + 1 <= LAYERS for extra in range(SPAN + 1)] for start in range(LAYERS)]
)


def _tables(seed, work):
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(2):
        eligible = int(rng.integers(6, 12))
        tables.append(
            _BucketTables(
                risk=rng.integers(1, 40, size=(PROMPTS, LAYERS, SPAN, HEADS, HEADS)) / 1000,
                anchor=rng.integers(1, 30, size=(PROMPTS, LAYERS)) / 1000,
                skipped=rng.integers(0, eligible + 1, size=(LAYERS, HEADS)),
                required=math.ceil(work * eligible * LAYERS * HEADS),
                eligible=eligible,
            )
        )
    return tables


def _layouts():
    for flags in itertools.product((False, True), repeat=LAYERS - 1):
        anchors = [0, *(layer + 1 for layer, flag in enumerate(flags) if flag)]
        edges = list(zip(anchors, [*anchors[1:], LAYERS]))
        if all(stop - start - 1 <= SPAN for start, stop in edges):
            yield edges


def _assignments(table, edges):
    """Every donor-or-fallback assignment of one bucket: variables, worst risk, work, tile cost."""
    heads = [
        (start, consumer - start - 1, head)
        for start, stop in edges
        for consumer in range(start + 1, stop)
        for head in range(HEADS)
    ]
    anchors = [start for start, _ in edges]
    anchor_cost = sum(2 * table.eligible * HEADS - int(table.skipped[a].sum()) for a in anchors)
    for donors in itertools.product(range(-1, HEADS), repeat=len(heads)):
        risk = table.anchor[:, anchors].sum(axis=1)
        skipped = 0
        for (start, offset, head), donor in zip(heads, donors):
            if donor >= 0:
                risk = risk + table.risk[:, start, offset, head, donor]
                skipped += int(table.skipped[start, donor])
        cost = anchor_cost + 2 * (table.eligible * len(heads) - skipped)
        if skipped >= table.required:
            yield list(zip(heads, donors)), float(risk.max()), cost


def _kept(keep, bucket, edges, variables):
    return all(keep.edges[start, stop - start - 1] for start, stop in edges) and all(
        keep.fallbacks[bucket][start, offset, head]
        if donor < 0
        else keep.donors[bucket][start, offset, head, donor]
        for (start, offset, head), donor in variables
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("work", [0.0, 0.3])
def test_priced_bounds_keep_every_policy_at_the_optimum(seed, work):
    tables = _tables(seed, work)
    policies = {
        tuple(edges): [list(_assignments(table, edges)) for table in tables] for edges in _layouts()
    }
    # Buckets share only the layout: the worst prompt is the worse bucket's best assignment.
    best_risk = min(
        max(min(risk for _, risk, _ in bucket) for bucket in buckets)
        for buckets in policies.values()
        if all(buckets)
    )
    risk_pass = _PricedRelaxation(tables, EDGES)
    prices = risk_pass.maximize(tolerance=1e-9, max_rounds=300)
    assert 0 < prices.bound <= best_risk + 1e-12
    keep = risk_pass.reduce(prices, best_risk + 1e-9)

    ceiling = best_risk + 0.002
    best_cost = min(
        sum(min(cost for _, risk, cost in bucket if risk <= ceiling) for bucket in buckets)
        for buckets in policies.values()
        if all(any(risk <= ceiling for _, risk, _ in bucket) for bucket in buckets)
    )
    within = risk_pass.reduce(prices, ceiling + 1e-9)
    for cost_pass in (
        _PricedRelaxation(tables, EDGES, ceiling=ceiling),
        _PricedRelaxation(tables, EDGES, ceiling=ceiling, keep=within),
    ):
        assert cost_pass.connected()
        cost_prices = cost_pass.maximize(tolerance=1e-6, max_rounds=300)
        assert 0 < cost_prices.bound <= best_cost + 1e-6
        cheapest = cost_pass.reduce(cost_prices, best_cost + 1e-6)
        found = False
        for edges, buckets in policies.items():
            parts = [
                [(v, cost) for v, risk, cost in bucket if risk <= ceiling] for bucket in buckets
            ]
            if not all(parts) or sum(min(c for _, c in part) for part in parts) != best_cost:
                continue
            found = True
            for bucket, part in enumerate(parts):
                least = min(cost for _, cost in part)
                assert all(_kept(cheapest, bucket, edges, v) for v, cost in part if cost == least)
        assert found

    for edges, buckets in policies.items():
        if not all(buckets) or max(min(r for _, r, _ in b) for b in buckets) > best_risk:
            continue
        for bucket, assignments in enumerate(buckets):
            assert all(
                _kept(keep, bucket, edges, v) for v, risk, _ in assignments if risk <= best_risk
            )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("work", [0.2, 0.5, 0.9])
def test_fixed_layout_choice_meets_the_work_target_when_one_can(seed, work):
    table = _tables(seed, work)[0]
    edges = [(0, 3), (3, 4)]
    feasible = list(_assignments(table, edges))
    risk = np.concatenate(
        (table.risk[:, 0, :, :, :].reshape(PROMPTS, -1, HEADS), np.zeros((PROMPTS, 4, 1))), axis=2
    )
    skipped = np.concatenate((np.tile(table.skipped[0], (4, 1)), np.zeros((4, 1), int)), axis=1)
    choice = _fixed_layout_choice(
        risk,
        skipped,
        np.ones_like(skipped, dtype=bool),
        table.anchor[:, [0, 3]].sum(axis=1),
        table.required,
        work_limit=16.0 * LAYERS * HEADS,
        tolerance=1e-9,
        max_rounds=100,
        sweeps=4,
    )
    assert (choice is not None) == bool(feasible)
    if choice is not None:
        assert skipped[np.arange(4), choice].sum() >= table.required
        worst = (
            table.anchor[:, [0, 3]].sum(axis=1) + risk[:, np.arange(4), choice].sum(axis=1)
        ).max()
        assert worst >= min(r for _, r, _ in feasible) - 1e-12
