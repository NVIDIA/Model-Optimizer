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

from types import SimpleNamespace

import pulp
import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import (
    mask_reuse_solver as solver_module,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_solver import (
    MaskReuseSolverConfig,
    MaskReuseSolverError,
    _SolverSession,
)


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


@pytest.mark.parametrize("backend", ["highs"])
def test_coupled_donor_signature_collision_is_resolved_lexicographically(backend):
    if backend == "highs" and not pulp.HiGHS().available():
        pytest.skip("optional highspy solver is not installed")
    problem = pulp.LpProblem("coupled_donors")
    donors = [pulp.LpVariable(f"donor_{i}", 0, 1, cat="Binary") for i in range(2)]
    problem += pulp.lpSum(donors) == 1
    _SolverSession(MaskReuseSolverConfig(backend=backend)).lexicographic(
        problem, donors, base=2, phase="map"
    )
    assert [v.value() for v in donors] == [0.0, 1.0]


@pytest.mark.parametrize("backend", ["highs"])
def test_94_layer_anchor_order_uses_bounded_coefficients(backend):
    if backend == "highs" and not pulp.HiGHS().available():
        pytest.skip("optional highspy solver is not installed")
    problem = pulp.LpProblem("94_layers")
    anchors = [pulp.LpVariable(f"anchor_{i}", 0, 1, cat="Binary") for i in range(94)]
    problem += pulp.lpSum(anchors) == 47
    _SolverSession(MaskReuseSolverConfig(backend=backend)).lexicographic(
        problem, (1 - x for x in anchors), base=2, phase="map"
    )
    assert [x.value() for x in anchors] == [1.0] * 47 + [0.0] * 47
    assert max(abs(c) for row in problem.constraints.values() for c in row.values()) <= 2**20


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
@pytest.mark.parametrize("later_phase", ["cost", "lexicographic"])
def test_generated_constraint_is_checked_before_fixing_later_objective(backend, later_phase):
    if backend == "highs" and not pulp.HiGHS().available():
        pytest.skip("optional highspy solver is not installed")
    problem = pulp.LpProblem("later_phase_separation")
    risk = pulp.LpVariable("risk", lowBound=0)
    head = pulp.LpVariable("head", 0, 1, cat="Binary")
    problem += head <= 1
    session = _SolverSession(MaskReuseSolverConfig(backend=backend))
    assert session.minimize_and_fix(problem, risk, "risk", integral=False) == 0
    calls = []

    def separate():
        calls.append((risk.value(), head.value()))
        if head.value() - risk.value() > 1e-5:
            problem.addConstraint(head <= risk, "omitted_prompt_risk")
            return True
        return False

    problem._mask_reuse_separate_constraints = separate
    if later_phase == "cost":
        assert session.minimize_and_fix(problem, -head, "cost", integral=True) == 0
    else:
        session.lexicographic(problem, [1 - head], base=2, phase="donor")
    assert [head_value for _, head_value in calls] == [1, 0]
    assert head.value() == 0
    assert [row["phase"] for row in session.records] == [
        "risk",
        "cost" if later_phase == "cost" else "donor_0",
        "cost" if later_phase == "cost" else "donor_0",
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
        _SolverSession(MaskReuseSolverConfig(backend="cbc")).minimize_and_fix(
            problem, -head, "cost", integral=True
        )
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
            session.minimize_and_fix(problem, x, "risk", integral=True)
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
        session.minimize_and_fix(problem, x, "risk", integral=True)
    assert "risk" not in problem.constraints
