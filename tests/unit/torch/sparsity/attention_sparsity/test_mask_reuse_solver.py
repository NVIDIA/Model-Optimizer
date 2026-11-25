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

"""Proof status, bounded exact ordering, and budget contracts for mask-reuse solves."""

import json
import math
import random
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pulp
import pytest
from _test_utils.torch.mask_reuse import brute, captures, checkpoint, solve

from modelopt.torch.sparsity.attention_sparsity.calibration import (
    mask_reuse_solver as solver_module,
)
from modelopt.torch.sparsity.attention_sparsity.calibration import mask_reuse_topology
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_solver import (
    MaskReuseSolverConfig,
    MaskReuseSolverError,
    _SolverSession,
)


def test_highs_stale_mip_start_cannot_fix_a_nonoptimal_risk(monkeypatch):
    pytest.importorskip("highspy")
    fixture = Path(__file__).parents[4] / "_test_utils/torch/data/mask_reuse_highs_stale"
    start = json.loads(fixture.with_suffix(".json").read_text())
    variables, problem = pulp.LpProblem.fromMPS(str(fixture.with_suffix(".mps")))
    for index, name in enumerate(start["names"]):
        variables[f"c{index}"].name = name
    problem._mask_reuse_initial_values = {
        variables[f"c{i}"]: value for i, value in enumerate(start["values"])
    }
    risk = next(iter(problem.objective))
    session = _SolverSession()
    original_solver = session._solver

    def use_affected_native_configuration():
        solver = original_solver()
        if not session._highs_cold:
            solver.optionsDict["presolve"] = "choose"
        return solver

    monkeypatch.setattr(session, "_solver", use_affected_native_configuration)
    # Exact binary fractions; brute force and a cold native solve agree.
    assert session.minimize(problem, risk, "risk")
    assert risk.value() == 4268 / 65536


@pytest.mark.parametrize("seed", [3, 16, 25, 30, 34, 39])
def test_native_roundoff_records_risk_and_cost_bounds(tmp_path, seed):
    pytest.importorskip("highspy")
    cp = checkpoint(tmp_path)
    ranges = ((129, 512, 256), (513, 1024, 768), (1025, 4096, 2048), (4097, 8192, 8192))
    result = solve(tmp_path, cp, captures(cp, seed=seed, buckets=ranges), 0.1, "highs")
    assert result["bmm1_skip_objective"]["target_met"]
    selected = result["provenance"]["selection_result"]
    assert selected["status"] == "within_tolerance"
    assert selected["risk_lower_bound"] <= selected["risk_upper_bound"] + 1e-12


@pytest.mark.parametrize("seed", [14, 37])
def test_native_cbc_proof_reports_infeasible_per_range_target(tmp_path, seed):
    cp = checkpoint(tmp_path)
    ranges = ((129, 512, 256), (513, 1024, 768), (1025, 4096, 2048), (4097, 8192, 8192))
    result = solve(tmp_path, cp, captures(cp, seed=seed, buckets=ranges), 0.1, "cbc")
    assert result["bmm1_skip_objective"]["target_met"] is False
    assert result["calibration_report"]["joint_frontier"][0]["risk"]["native_infeasibility_proven"]


@pytest.mark.parametrize("seed", [1, 4, 9])
def test_reduced_program_preserves_the_full_optimum(tmp_path, monkeypatch, seed):
    pytest.importorskip("highspy")
    cp = checkpoint(tmp_path, layers=5, heads=3)
    ranges = ((129, 512, 256), (513, 1024, 768))
    rows = captures(cp, seed=seed, layers=5, heads=3, span=2, buckets=ranges)
    reduced = solve(tmp_path, cp, rows, 0.3, "highs")
    monkeypatch.setattr(mask_reuse_topology, "_PRICE_REDUCTION", False)
    full = solve(tmp_path, cp, rows, 0.3, "highs")
    key = "worst_development_prompt_model_wide_total_dropped_mass"
    risks = []
    for result in (reduced, full):
        assert result["provenance"]["selection_result"]["status"] == "within_tolerance"
        risks.append(result["calibration_report"]["risk_selection_objective"][key])
    # Both answers are the cheapest policy within the same risk tolerance.
    assert risks[0] == pytest.approx(risks[1], abs=1e-7)
    assert (
        reduced["calibration_report"]["combined_tile_cost"]
        == full["calibration_report"]["combined_tile_cost"]
    )


def test_objective_target_stops_one_solve_and_not_the_next():
    pytest.importorskip("highspy")
    rng = random.Random(1)
    items = [pulp.LpVariable(f"x{index}", cat="Binary") for index in range(60)]
    problem = pulp.LpProblem("cover", pulp.LpMinimize)
    for row in range(180):
        problem += pulp.lpSum(items[index] for index in rng.sample(range(60), 4)) >= 1 + row % 2
    weights = [rng.randrange(10, 60) for _ in items]
    cost = pulp.lpSum(weight * item for weight, item in zip(weights, items))
    session = _SolverSession()
    # Any cover meets this target, so the search stops at its first one, unproved.
    problem._mask_reuse_objective_target = float(sum(weights))
    with pytest.raises(MaskReuseSolverError, match="unresolved"):
        session.minimize(problem, cost, "risk")
    assert session.records[-1]["dual_bound"] < session.records[-1]["primal_bound"]
    # The persistent native model must not keep the target for the next solve.
    problem._mask_reuse_objective_target = None
    assert session.minimize(problem, cost, "risk")


def test_lazy_prompt_rows_preserve_the_eager_optimum(tmp_path, monkeypatch):
    pytest.importorskip("highspy")
    cp = checkpoint(tmp_path)
    rows = captures(cp, seed=1, buckets=((129, 512, 256), (129, 512, 384), (129, 512, 512)))
    eager = solve(tmp_path, cp, rows, 0.1, "highs")
    monkeypatch.setattr(mask_reuse_topology, "_GENERATED_RISK_TERM_THRESHOLD", -1)
    lazy = solve(tmp_path, cp, rows, 0.1, "highs")
    for key in ("anchors", "nearest", "discovery_context_policies", "risk_selection_objective"):
        assert lazy[key] == eager[key]


@pytest.mark.parametrize(
    ("seed", "tied"), [(0, False), (5, False), (7, False), (11, False), (0, True), (5, True)]
)
def test_cpu_selector_matches_all_brute_force_objectives(tmp_path, seed, tied):
    pytest.importorskip("highspy")
    cp = checkpoint(tmp_path)
    rows = captures(cp, seed=seed, buckets=((129, 512, 256), (129, 512, 384)))
    if tied:
        # Binary-fraction ties exercise the cost stage. Random
        # continuous risks usually have a unique optimum and cannot test them.
        for row in rows:
            for stats in row["anchor_stats_by_layer"].values():
                stats["dropped_mass"] = [round(x * 1000) % 4 / 8 for x in stats["dropped_mass"]]
            for candidates in row["consumer_candidates_by_layer"].values():
                for stats in candidates.values():
                    stats["dropped_mass"] = [
                        [round(x * 1000) % 4 / 8 for x in values]
                        for values in stats["dropped_mass"]
                    ]
    expected = brute([row for row in rows if row["invocation"]["split"] == "calibration"], 0.1)
    result = solve(tmp_path, cp, rows, 0.1, "highs")
    assert expected is not None
    key, _, _ = expected
    assert result["calibration_report"]["combined_tile_cost"] == key[1]
    assert result["anchors"] == list(key[4])
    assert result["risk_selection_objective"][
        "worst_development_prompt_model_wide_total_dropped_mass"
    ] == pytest.approx(key[0], abs=1e-13)
    rows_for_replay = [r for r in rows if r["invocation"]["split"] == "calibration"]
    replayed = brute(
        rows_for_replay, 0.1, policy=(result["anchors"], result["discovery_context_policies"][0])
    )
    assert replayed is not None and replayed[0][:2] == key[:2]


@pytest.mark.parametrize(
    ("status", "solution"),
    [
        (pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound),
        (pulp.LpStatusUndefined, pulp.LpSolutionNoSolutionFound),
        (pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible),
        (pulp.LpStatusNotSolved, pulp.LpSolutionIntegerFeasible),
    ],
)
def test_unresolved_or_partial_incumbent_is_not_infeasibility(monkeypatch, status, solution):
    def stopped(problem, solver):
        problem.assignStatus(status, solution)
        return status

    monkeypatch.setattr(pulp.LpProblem, "solve", stopped)
    problem = pulp.LpProblem("interrupted")
    x = pulp.LpVariable("x", 0, 1, cat="Binary")
    with pytest.raises(MaskReuseSolverError, match="unresolved") as caught:
        _SolverSession(MaskReuseSolverConfig(backend="cbc")).minimize(problem, x, "risk")
    assert caught.value.diagnostics[-1]["status"] == pulp.LpStatus[status]


@pytest.mark.parametrize(
    ("skipped_tiles", "minimum_tiles", "accepted"),
    [
        (0, None, True),
        (0, 0, True),
        (11, 10, True),
        (10, 11, False),
        (2**60, 2**60, True),
        (2**60, 2**60 + 1, False),
    ],
)
def test_discrete_work_target_uses_exact_integers(skipped_tiles, minimum_tiles, accepted):
    session = _SolverSession()
    session.records.append({"phase": "risk", "status": "Optimal"})
    if accepted:
        assert session.require_work_target(skipped_tiles, minimum_tiles, phase="extraction") is None
    else:
        with pytest.raises(
            MaskReuseSolverError, match="invalid discrete work target in extraction"
        ) as caught:
            session.require_work_target(skipped_tiles, minimum_tiles, phase="extraction")
        assert f"skipped_tiles={skipped_tiles}" in str(caught.value)
        assert f"minimum_tiles={minimum_tiles}" in str(caught.value)
        assert caught.value.diagnostics == session.records
        assert caught.value.diagnostics is not session.records


@pytest.mark.parametrize("backend", ["highs"])
def test_proven_infeasible_and_expired_budget_are_distinct(backend):
    if backend == "highs" and not pulp.HiGHS().available():
        pytest.skip("optional highspy solver is not installed")
    problem = pulp.LpProblem("impossible")
    x = pulp.LpVariable("x", 0, 1, cat="Binary")
    problem += x >= 2
    assert (
        _SolverSession(MaskReuseSolverConfig(backend=backend)).minimize(problem, x, "risk") is False
    )
    session = _SolverSession(MaskReuseSolverConfig(time_limit_seconds=1.0))
    session.started -= 2.0
    with pytest.raises(MaskReuseSolverError, match="budget exhausted"):
        session.minimize(problem, x, "risk")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "invalid"},
        {"time_limit_seconds": 0},
        {"time_limit_seconds": float("nan")},
        {"risk_absolute_tolerance": -1},
    ],
)
def test_solver_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MaskReuseSolverConfig(**kwargs)


@pytest.mark.parametrize("backend", ["highs"])
def test_generated_constraint_is_checked_for_the_cost_objective(backend):
    if backend == "highs" and not pulp.HiGHS().available():
        pytest.skip("optional highspy solver is not installed")
    problem = pulp.LpProblem("later_phase_separation")
    risk = pulp.LpVariable("risk", lowBound=0)
    head = pulp.LpVariable("head", 0, 1, cat="Binary")
    problem += head <= 1
    session = _SolverSession(MaskReuseSolverConfig(backend=backend))
    assert session.minimize(problem, risk, "risk")
    problem += risk <= 1e-7
    calls = []

    def separate():
        calls.append((risk.value(), head.value()))
        if head.value() - risk.value() > 1e-5:
            problem.addConstraint(head <= risk, "omitted_prompt_risk")
            return True
        return False

    problem._mask_reuse_separate_constraints = separate
    assert session.minimize(problem, -head, "cost")
    assert [head_value for _, head_value in calls] == [1, 0]
    assert head.value() == 0
    assert [row["phase"] for row in session.records] == [
        "risk",
        "cost",
        "cost",
    ]


def test_timeout_after_constraint_generation_does_not_fix_objective(monkeypatch):
    problem = pulp.LpProblem("interrupted_separation")
    head = pulp.LpVariable("head", 0, 1, cat="Binary")
    problem += head <= 1
    solves = []

    def solve(problem, solver):
        solves.append(solver)
        head.varValue = 1
        if len(solves) == 1:
            problem.assignStatus(pulp.LpStatusOptimal, pulp.LpSolutionOptimal)
        else:
            problem.assignStatus(pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound)
        return problem.status

    def separate():
        problem.addConstraint(head <= 0, "omitted_prompt_risk")
        return True

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)
    problem._mask_reuse_separate_constraints = separate
    with pytest.raises(MaskReuseSolverError, match="unresolved") as caught:
        _SolverSession(MaskReuseSolverConfig(backend="cbc")).minimize(problem, -head, "cost")
    assert len(solves) == 2
    assert "cost" not in problem.constraints
    assert caught.value.diagnostics[-1]["solution_status"] == "No Solution Found"


@pytest.mark.parametrize("setup_seconds", [4.0, 11.0])
def test_highs_budget_is_refreshed_after_model_transfer(monkeypatch, setup_seconds):
    pytest.importorskip("highspy")
    clock = [0.0]
    native_limits = []
    original_build = solver_module._BatchHiGHS.buildSolverModel
    original_call = pulp.HiGHS.callSolver

    def build(self, problem):
        original_build(self, problem)
        clock[0] += setup_seconds

    def call(self, problem):
        native_limits.append(problem.solverModel.getOptionValue("time_limit")[1])
        return original_call(self, problem)

    monkeypatch.setattr(solver_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(solver_module._BatchHiGHS, "buildSolverModel", build)
    monkeypatch.setattr(pulp.HiGHS, "callSolver", call)
    problem = pulp.LpProblem("budget_after_transfer")
    x = pulp.LpVariable("x", 0, 1, cat="Binary")
    problem += x >= 1
    session = _SolverSession(MaskReuseSolverConfig(backend="highs", time_limit_seconds=10))
    if setup_seconds < 10:
        assert session.minimize(problem, x, "risk")
        assert native_limits == [10 - setup_seconds]
    else:
        with pytest.raises(MaskReuseSolverError, match="budget exhausted"):
            session.minimize(problem, x, "risk")
        assert native_limits == []
        assert "risk" not in problem.constraints


@pytest.mark.parametrize("delay_stage", ["readback", "separation"])
def test_late_optimum_is_not_fixed_after_shared_budget_expiry(monkeypatch, delay_stage):
    pytest.importorskip("highspy")
    clock = [0.0]
    monkeypatch.setattr(solver_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    problem = pulp.LpProblem("late_optimum")
    x = pulp.LpVariable("x", 0, 1, cat="Binary")
    problem += x >= 1
    session = _SolverSession(MaskReuseSolverConfig(backend="highs", time_limit_seconds=10))
    if delay_stage == "readback":
        original_reader = solver_module._BatchHiGHS.findSolutionValues

        def reader(self, problem):
            result = original_reader(self, problem)
            clock[0] += 11
            return result

        monkeypatch.setattr(solver_module._BatchHiGHS, "findSolutionValues", reader)
    else:

        def separate():
            clock[0] += 11
            return False

        problem._mask_reuse_separate_constraints = separate
    with pytest.raises(MaskReuseSolverError, match="budget exhausted"):
        session.minimize(problem, x, "risk")
    assert "risk" not in problem.constraints


@pytest.mark.parametrize(("seed", "cost"), [(800616, 993), (801018, 1563), (800021, 253)])
def test_single_risk_ceiling_across_target_tuples(tmp_path, seed, cost):
    # Calibration records from the three independently enumerated decimal-tie cases.
    fixture = Path(__file__).parents[4] / "_test_utils/torch/data/mask_reuse_band_cases.jsonl"
    cases = [
        r for line in fixture.read_text().splitlines() if (r := json.loads(line))["seed"] == seed
    ]
    layers, heads = len(cases[0]["anchor_mass"]), len(cases[0]["anchor_mass"][0])
    cp = checkpoint(tmp_path, layers=layers, heads=heads)
    rows = []
    fit = {
        "formula": "a * exp(b * target_sparsity)",
        "prefill": {"a": 1, "b": 1, "min_observed_sparsity": 0.05, "max_observed_sparsity": 0.95},
    }
    for prompt_index, r in enumerate(cases):
        for i, target in enumerate(r["targets"]):
            records = captures(
                cp, layers=layers, heads=heads, span=r["span"], buckets=[(*r["range"], r["length"])]
            )
            shift = i * r["shift"]
            for row in records:
                inv = row["invocation"]
                ident = f"{inv['split']}-{prompt_index}"
                digest = sha256(ident.encode()).hexdigest()
                log2 = target * math.log2(math.e) - math.log2(r["length"])
                inv.update(
                    prompt_id=ident,
                    source_group_sha256=digest,
                    source_capture_sha256=digest,
                    target_sparsity_hex=target.hex(),
                    threshold_log2_hex=log2.hex(),
                    threshold_lambda_hex=(2**log2).hex(),
                )
                for a, stats in row["anchor_stats_by_layer"].items():
                    stats.update(
                        retained_tiles=r["retained"][i][int(a)],
                        dropped_mass=[v + shift for v in r["anchor_mass"][int(a)]],
                    )
                for c, candidates in row["consumer_candidates_by_layer"].items():
                    for a, stats in candidates.items():
                        stats["dropped_mass"] = [
                            [v + shift for v in values] for values in r["consumer_mass"][f"{a},{c}"]
                        ]
                rows.append(row)
    result = solve(tmp_path, cp, rows, cases[0]["rho"], "highs", fit=fit)
    assert result["bmm1_skip_objective"]["target_met"]
    assert result["calibration_report"]["combined_tile_cost"] == cost
    if seed == 800021:
        assert [p["target_sparsity"] for p in result["discovery_context_policies"]] == [0.7, 0.7]
