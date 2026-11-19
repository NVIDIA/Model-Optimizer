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

"""Proof-aware, bounded optimization for offline mask-reuse calibration."""

from __future__ import annotations

import math
import struct
import time
import weakref
from array import array
from contextlib import suppress
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import pulp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any

__all__ = [
    "MaskReuseAllocationPause",
    "MaskReuseSolverConfig",
    "MaskReuseSolverError",
    "MaskReuseSolverFailure",
]


@dataclass(frozen=True)
class MaskReuseSolverConfig:
    """Select an exact solver and a cooperative budget shared by all selection phases."""

    backend: Literal["cbc", "highs"] = "highs"
    time_limit_seconds: float = 600.0
    risk_absolute_tolerance: float = 0.0
    log: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("cbc", "highs"):
            raise ValueError("solver backend must be cbc or highs")
        for name in ("time_limit_seconds",):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.risk_absolute_tolerance) not in (int, float)
            or self.risk_absolute_tolerance != 0
        ):
            raise ValueError("exact risk ordering requires risk_absolute_tolerance=0")


class MaskReuseSolverError(RuntimeError):
    """A solve was unresolved or invalid; this is not a proof of infeasibility."""

    def __init__(self, message: str, diagnostics: list[dict[str, object]]) -> None:
        """Retain completed phase diagnostics without exporting an unproved policy."""
        super().__init__(message)
        self.diagnostics = list(diagnostics)


class MaskReuseAllocationPause(MaskReuseSolverError):  # noqa: N818 - operational pause, not a proof error
    """The allocation must finalize; no unfinished native result is a proof."""


class MaskReuseSolverFailure(MaskReuseSolverError):  # noqa: N818 - backwards-compatible error subtype
    """A solver/runtime contract failed, rather than merely running out of time."""


class _BatchHiGHS(pulp.HiGHS):
    """Transfer the same PuLP model without millions of scalar native API calls."""

    def callSolver(self, problem):  # noqa: N802 - PuLP override
        remaining = getattr(self, "_remaining_budget_seconds", None)
        if remaining is not None:
            # Model packing consumes the same session budget as optimization.
            problem.solverModel.setOptionValue("time_limit", remaining())
        return super().callSolver(problem)

    def buildSolverModel(self, problem):  # noqa: N802 - PuLP override
        # HiGHS is optional; importing it is deferred until this backend is selected.
        import highspy

        variables = problem.variables()
        constraints = list(problem.constraints.values())
        model = highspy.HighsLp()
        model.num_col_ = len(variables)
        model.num_row_ = len(constraints)
        sign = -1 if problem.sense == pulp.LpMaximize else 1
        model.col_cost_ = array("d", (sign * problem.objective.get(v, 0.0) for v in variables))
        model.col_lower_ = array(
            "d", (-highspy.kHighsInf if v.lowBound is None else v.lowBound for v in variables)
        )
        model.col_upper_ = array(
            "d", (highspy.kHighsInf if v.upBound is None else v.upBound for v in variables)
        )
        if self.mip and any(v.cat == pulp.LpInteger for v in variables):
            model.integrality_ = [
                highspy.HighsVarType.kInteger
                if v.cat == pulp.LpInteger
                else highspy.HighsVarType.kContinuous
                for v in variables
            ]
        for index, variable in enumerate(variables):
            variable.index = index
        starts, indices, values = array("i", [0]), array("i"), array("d")
        lower, upper = array("d"), array("d")
        for index, constraint in enumerate(constraints):
            constraint.index = index
            for variable, coefficient in constraint.items():
                if coefficient != 0:
                    indices.append(variable.index)
                    values.append(coefficient)
            starts.append(len(indices))
            lb, ub = constraint.getLb(), constraint.getUb()
            lower.append(-highspy.kHighsInf if lb is None else lb)
            upper.append(highspy.kHighsInf if ub is None else ub)
        model.row_lower_, model.row_upper_ = lower, upper
        matrix = model.a_matrix_
        matrix.num_col_, matrix.num_row_ = len(variables), len(constraints)
        matrix.format_ = highspy.MatrixFormat.kRowwise
        matrix.start_, matrix.index_, matrix.value_ = starts, indices, values
        if problem.solverModel.passModel(model) == highspy.HighsStatus.kError:
            raise pulp.PulpSolverError("HiGHS rejected the batched calibration model")
        initial = problem.__dict__.pop("_mask_reuse_initial_values", None)
        if initial is not None and not getattr(self, "_disable_starts", False):
            # The selector validated this full assignment; omitted entries are
            # explicit zeros, not unspecified values in a partial MIP start.
            solution = highspy.HighsSolution()
            solution.col_value = array("d", (initial.get(v, 0.0) for v in variables))
            if problem.solverModel.setSolution(solution) == highspy.HighsStatus.kError:
                raise pulp.PulpSolverError("HiGHS rejected the calibration initial assignment")

    def findSolutionValues(self, problem):  # noqa: N802 - PuLP override
        # Native row properties copy their entire array on each access. Cache
        # them once while retaining PuLP's status mapping and value assignment.
        native = problem.solverModel
        solution = native.getSolution()
        cached = SimpleNamespace(
            **{
                name: getattr(solution, name)
                for name in ("col_value", "col_dual", "row_value", "row_dual")
            }
        )
        problem.solverModel = SimpleNamespace(
            getSolution=lambda: cached,
            getModelStatus=native.getModelStatus,
            getObjectiveValue=native.getObjectiveValue,
        )
        try:
            return super().findSolutionValues(problem)
        finally:
            problem.solverModel = native


class _PersistentHiGHS(_BatchHiGHS):
    """Reuse only a content-verified model with an unchanged ordered row prefix."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._problem_ref: weakref.ReferenceType | None = None
        self._native: Any = None
        self._signature: Any = None
        self._next_signature = None
        self._costs = {}
        self._incumbent = None

    def _discard(self):
        previous = self._problem_ref() if self._problem_ref is not None else None
        if previous is not None and previous.solverModel is self._native:
            previous.solverModel = None
        self._problem_ref = self._native = self._signature = None
        self._next_signature = None
        self._costs = {}
        self._incumbent = None

    def _owns_problem(self, problem) -> bool:
        return self._problem_ref is not None and self._problem_ref() is problem

    def _restore_incumbent(self, problem):
        # This is a MIP hint, not a replacement for native proof or risk separation.
        import highspy

        values, self._incumbent = self._incumbent, None
        if values is None or getattr(self, "_disable_starts", False):
            return
        variables = problem.variables()
        if len(values) != len(variables) or not all(math.isfinite(value) for value in values):
            return
        status, tolerance = self._native.getOptionValue("mip_feasibility_tolerance")
        self._check(status, "incumbent feasibility option")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise pulp.PulpSolverError("invalid HiGHS incumbent feasibility tolerance")
        for variable, value in zip(variables, values):
            if (
                (variable.lowBound is not None and value < variable.lowBound - tolerance)
                or (variable.upBound is not None and value > variable.upBound + tolerance)
                or (variable.cat == pulp.LpInteger and abs(value - round(value)) > tolerance)
            ):
                return
        for constraint in problem.constraints.values():
            residual = constraint.constant
            for variable, coefficient in constraint.items():
                residual += coefficient * values[variable.index]
            if not math.isfinite(residual) or (
                abs(residual) > tolerance
                if constraint.sense == pulp.LpConstraintEQ
                else residual * constraint.sense < -tolerance
            ):
                return
        solution = highspy.HighsSolution()
        solution.col_value = values
        self._check(self._native.setSolution(solution), "incumbent restoration")

    def _fingerprint(self, problem):
        columns = sha256()
        for index, variable in enumerate(problem.variables()):
            variable.index = index
            name = (variable.name + "\0" + variable.cat).encode()
            columns.update(
                struct.pack(
                    "<QQdd",
                    id(variable),
                    len(name),
                    -math.inf if variable.lowBound is None else variable.lowBound,
                    math.inf if variable.upBound is None else variable.upBound,
                )
            )
            columns.update(name)
        rows = []
        for index, (name, constraint) in enumerate(problem.constraints.items()):
            constraint.index = index
            row = sha256(name.encode())
            lower, upper = constraint.getLb(), constraint.getUb()
            row.update(
                struct.pack(
                    "<idd",
                    constraint.sense,
                    -math.inf if lower is None else lower,
                    math.inf if upper is None else upper,
                )
            )
            for variable, coefficient in constraint.items():
                if coefficient != 0:
                    row.update(struct.pack("<Qd", variable.index, coefficient))
            rows.append(row.digest())
        settings = (
            self.mip,
            self.msg,
            self.gapAbs,
            self.gapRel,
            self.threads,
            self.callbackTuple,
            tuple(sorted((k, v) for k, v in self.optionsDict.items() if k != "time_limit")),
        )
        return (columns.digest(), problem.sense, settings), tuple(rows)

    @staticmethod
    def _check(status, operation):
        if status.name == "kError":
            raise pulp.PulpSolverError(f"HiGHS rejected persistent {operation}")

    def createAndConfigureSolver(self, problem):  # noqa: N802 - PuLP override
        signature = self._fingerprint(problem)
        self._reusing = (
            problem is not None
            and self._signature is not None
            and self._owns_problem(problem)
            and problem.solverModel is self._native
            and "_mask_reuse_initial_values" not in problem.__dict__
            and signature[0] == self._signature[0]
            and signature[1][: len(self._signature[1])] == self._signature[1]
        )
        if self._reusing:
            # clearSolver invalidates the primal even though its numeric buffer survives.
            if self.mip and any(v.cat == pulp.LpInteger for v in problem.variables()):
                import highspy

                solution, info = self._native.getSolution(), self._native.getInfo()
                if (
                    solution.value_valid
                    and info.primal_solution_status == highspy.kSolutionStatusFeasible
                ):
                    self._incumbent = array("d", solution.col_value)
            self._check(self._native.clearSolver(), "optimization-state reset")
            options = {
                "output_flag": self.msg,
                "mip_abs_gap": self.gapAbs,
                "mip_rel_gap": self.gapRel,
                "threads": self.threads,
                **self.optionsDict,
            }
            for name, value in options.items():
                if value is not None and name != "time_limit":
                    self._check(self._native.setOptionValue(name, value), "option update")
        else:
            self._discard()
            super().createAndConfigureSolver(problem)
        self._next_signature = signature

    def buildSolverModel(self, problem):  # noqa: N802 - PuLP override
        sign = -1 if problem.sense == pulp.LpMaximize else 1
        costs = {
            variable.index: sign * float(coefficient)
            for variable, coefficient in problem.objective.items()
            if coefficient != 0
        }
        if not self._reusing:
            super().buildSolverModel(problem)
            self._native = problem.solverModel
            owner = weakref.ref(self)

            def release(_):
                if (solver := owner()) is not None:
                    solver._discard()

            self._problem_ref = weakref.ref(problem, release)
        else:
            updates = dict.fromkeys(self._costs, 0.0)
            updates.update(costs)
            if updates:
                self._check(
                    self._native.changeColsCost(
                        len(updates), array("i", updates), array("d", updates.values())
                    ),
                    "objective update",
                )
            appended = list(problem.constraints.values())[len(self._signature[1]) :]
            if appended:
                starts, indices, values = array("i", [0]), array("i"), array("d")
                lower, upper = array("d"), array("d")
                for constraint in appended:
                    lb, ub = constraint.getLb(), constraint.getUb()
                    lower.append(-math.inf if lb is None else lb)
                    upper.append(math.inf if ub is None else ub)
                    for variable, coefficient in constraint.items():
                        if coefficient != 0:
                            indices.append(variable.index)
                            values.append(coefficient)
                    starts.append(len(indices))
                self._check(
                    self._native.addRows(
                        len(appended), lower, upper, len(indices), starts, indices, values
                    ),
                    "row append",
                )
            self._restore_incumbent(problem)
        self._signature, self._costs = self._next_signature, costs

    def callSolver(self, problem):  # noqa: N802 - PuLP override
        remaining = getattr(self, "_remaining_budget_seconds", None)
        seconds = (
            remaining()
            if remaining is not None
            else self.optionsDict.get("time_limit", self.timeLimit)
        )
        if seconds is not None:
            mip = self.mip and problem.isMIP()
            if mip:
                status, relaxation = problem.solverModel.getOptionValue("solve_relaxation")
                self._check(status, "relaxation option")
                mip = not relaxation
            # MIP has a per-run timer; LP uses the cumulative native clock.
            limit = seconds if mip else problem.solverModel.getRunTime() + seconds
            self._check(
                problem.solverModel.setOptionValue("time_limit", limit),
                "time limit",
            )
        pulp.HiGHS.callSolver(self, problem)
        if problem.solverModel.getModelStatus().name.endswith("Error"):
            raise pulp.PulpSolverError("HiGHS persistent solve failed")

    def actualSolve(self, problem):  # noqa: N802 - PuLP override
        try:
            return super().actualSolve(problem)
        except BaseException:
            self._discard()
            problem.solverModel = None
            problem.assignStatus(pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound)
            for variable in problem.variables():
                variable.varValue = variable.dj = None
            for constraint in problem.constraints.values():
                constraint.slack = constraint.pi = None
            raise


class _SolverSession:
    def __init__(
        self,
        config: MaskReuseSolverConfig | None = None,
        *,
        spent_seconds: float = 0.0,
        records: Iterable[dict[str, object]] = (),
        allocation_remaining: Callable[[], float] | None = None,
    ) -> None:
        if isinstance(spent_seconds, bool) or not math.isfinite(spent_seconds) or spent_seconds < 0:
            raise ValueError("restored solver time must be finite and nonnegative")
        self.config = config or MaskReuseSolverConfig()
        if allocation_remaining is not None and (
            not callable(allocation_remaining) or self.config.backend != "highs"
        ):
            raise ValueError("allocation limits require a callback and the HiGHS backend")
        self._allocation_remaining = allocation_remaining
        self._native_limit_source = None
        self._native_limit_seconds: float | None = None
        self.started = time.monotonic()
        self._spent_seconds = spent_seconds
        self.records: list[dict[str, object]] = [dict(row) for row in records]
        self._highs_solver: _PersistentHiGHS | None = None
        self._highs_cold = False

    @property
    def elapsed_seconds(self) -> float:
        return self._spent_seconds + time.monotonic() - self.started

    def _remaining_seconds(self) -> float:
        remaining = self.config.time_limit_seconds - self.elapsed_seconds
        if not math.isfinite(remaining):
            raise MaskReuseSolverFailure("invalid solver remaining time", self.records)
        if remaining <= 0:
            raise MaskReuseSolverError("calibration solver budget exhausted", self.records)
        return remaining

    def _check_allocation(self) -> float | None:
        if self._allocation_remaining is None:
            return None
        remaining = self._allocation_remaining()
        if isinstance(remaining, bool) or not math.isfinite(remaining):
            raise MaskReuseSolverFailure("invalid allocation remaining time", self.records)
        if remaining <= 0:
            raise MaskReuseAllocationPause("allocation work cutoff reached", self.records)
        return remaining

    def _native_remaining_seconds(self) -> float:
        # Original stage exhaustion takes precedence if both clocks have expired.
        stage = self._remaining_seconds()
        allocation = self._check_allocation()
        limited = allocation is not None and allocation < stage
        self._native_limit_source = "allocation" if limited else "stage"
        self._native_limit_seconds = stage if allocation is None else min(stage, allocation)
        return self._native_limit_seconds

    def _solver(self):
        remaining = self._native_remaining_seconds()
        kwargs = {
            "msg": self.config.log,
            "threads": 1,
            "timeLimit": remaining,
            "gapRel": 0.0,
            "gapAbs": 0.0,
        }
        if self.config.backend == "highs":
            if self._highs_solver is None:
                self._highs_solver = _PersistentHiGHS(
                    **kwargs,
                    random_seed=1,
                    parallel="off",
                    small_matrix_value=1e-12,
                    mip_feasibility_tolerance=1e-9,
                    mip_rel_gap=0.0,
                    mip_abs_gap=0.0,
                    # HiGHS 1.16 presolve can misreport warm-start assignments and
                    # overrun its time budget on large discovery models.
                    presolve="choose" if self._highs_cold else "off",
                )
                self._highs_solver._disable_starts = self._highs_cold
            solver = self._highs_solver
            solver.timeLimit = remaining
            solver._remaining_budget_seconds = self._native_remaining_seconds
        else:
            solver = pulp.PULP_CBC_CMD(
                **kwargs,
                options=[
                    "randomSeed 1",
                    "randomCbcSeed 1",
                    "increment 1e-12",
                    "preprocess off",
                    "primalTolerance 1e-9",
                    "integerTolerance 1e-9",
                ],
            )
        if not solver.available():
            raise MaskReuseSolverFailure(
                f"{self.config.backend} solver is unavailable", self.records
            )
        return solver

    def _retry_without_warm_start(self, problem) -> bool:
        if self.config.backend != "highs" or self._highs_cold:
            return False
        assert self._highs_solver is not None
        self.records[-1]["cold_retry"] = True
        self._highs_solver._discard()
        self._highs_solver = None
        self._highs_cold = True
        problem.__dict__.pop("_mask_reuse_initial_values", None)
        return True

    def minimize(self, problem: pulp.LpProblem, expression, phase: str) -> bool:
        objective = pulp.lpSum([expression])
        if not any(objective.values()):
            # PuLP's temporary dummy column survives in CBC's later MPS bounds.
            zero = getattr(problem, "_mask_reuse_zero", None)
            if zero is None:
                zero = pulp.LpVariable("mask_reuse_zero", 0, 0)
                problem._mask_reuse_zero = zero
                problem += zero == 0, "mask_reuse_zero"
            objective += zero
        problem.setObjective(objective)
        while True:
            started = time.monotonic()
            try:
                status = problem.solve(self._solver())
                # PuLP maps kUnboundedOrInfeasible to its infeasible pair. Only
                # the native kInfeasible status proves infeasibility for HiGHS.
                native_status = (
                    problem.solverModel.getModelStatus().name
                    if self.config.backend == "highs"
                    else None
                )
            except MaskReuseSolverError:
                raise
            except Exception as error:
                raise MaskReuseSolverFailure(
                    f"solver error in {phase}: {error}", self.records
                ) from error
            observer = getattr(problem, "_mask_reuse_observe_assignment", None)
            if observer is not None:
                observer()
            solution_status = problem.sol_status
            self.records.append(
                {
                    "phase": phase,
                    "status": pulp.LpStatus.get(status, str(status)),
                    "solution_status": pulp.LpSolution.get(solution_status, str(solution_status)),
                    "elapsed_seconds": time.monotonic() - started,
                    **(
                        {
                            "native_status": native_status,
                            "native_limit_source": self._native_limit_source,
                            "native_limit_seconds": self._native_limit_seconds,
                        }
                        if self._allocation_remaining is not None
                        else {}
                    ),
                }
            )
            if native_status is not None and native_status not in {
                "kOptimal",
                "kInfeasible",
                "kTimeLimit",
                "kIterationLimit",
                "kObjectiveBound",
                "kObjectiveTarget",
                "kSolutionLimit",
                "kInterrupt",
                "kHighsInterrupt",
                "kMemoryLimit",
            }:
                raise MaskReuseSolverFailure(
                    f"native solver failure in {phase}: {native_status}", self.records
                )
            infeasible = (
                status == pulp.LpStatusInfeasible and solution_status == pulp.LpSolutionInfeasible
            )
            optimal = status == pulp.LpStatusOptimal and solution_status == pulp.LpSolutionOptimal
            if optimal and native_status == "kOptimal" and problem.isMIP():
                native = problem.solverModel
                info = native.getInfo()
                self.records[-1].update(
                    native_gap=info.mip_gap,
                    primal_bound=info.objective_function_value,
                    dual_bound=info.mip_dual_bound,
                )
                # A status and gap certify the reported objective, not necessarily
                # the returned point (HiGHS can retain a stale MIP-start buffer).
                lp, solution = native.getLp(), native.getSolution()
                actual = (
                    math.fsum(c * v for c, v in zip(lp.col_cost_, solution.col_value)) + lp.offset_
                )
                reported = info.objective_function_value
                consistent = (
                    solution.value_valid
                    and math.isfinite(actual)
                    and math.isfinite(reported)
                    and abs(actual - reported) <= 8 * max(math.ulp(actual), math.ulp(reported))
                )
                self.records[-1]["assignment_objective"] = actual
                if (not consistent or info.mip_gap != 0.0) and self._retry_without_warm_start(
                    problem
                ):
                    continue
                if not consistent:
                    raise MaskReuseSolverFailure(
                        f"native objective differs from returned assignment in {phase}",
                        self.records,
                    )
                if info.mip_gap != 0.0:
                    raise MaskReuseSolverError("native MIP gap is nonzero", self.records)
            if (
                native_status is not None
                and (
                    infeasible != (native_status == "kInfeasible")
                    or optimal != (native_status == "kOptimal")
                )
            ) or status == pulp.LpStatusUnbounded:
                raise MaskReuseSolverFailure(
                    f"invalid solver status in {phase}: {native_status}", self.records
                )
            if optimal and not problem.valid(1e-5):
                raise MaskReuseSolverFailure(f"invalid solver assignment in {phase}", self.records)
            if optimal:
                # Recheck discrete flow, counts, saving targets and fixed integer
                # objectives in integer arithmetic, not native feasibility tolerances.
                for row in problem.constraints.values():
                    if row.constant != int(row.constant) or any(
                        variable.cat != pulp.LpInteger or coefficient != int(coefficient)
                        for variable, coefficient in row.items()
                    ):
                        continue
                    residual = int(row.constant) + sum(
                        int(coefficient) * round(variable.varValue)
                        for variable, coefficient in row.items()
                    )
                    if (
                        (row.sense == pulp.LpConstraintEQ and residual != 0)
                        or (row.sense == pulp.LpConstraintLE and residual > 0)
                        or (row.sense == pulp.LpConstraintGE and residual < 0)
                    ):
                        raise MaskReuseSolverFailure(
                            f"invalid discrete constraint in {phase}: {row.name}", self.records
                        )
            # A definitive failure must not be hidden by a coincident deadline.
            self._remaining_seconds()
            if infeasible:
                self._check_allocation()
                return False
            if not optimal:
                if native_status == "kTimeLimit" and self._native_limit_source == "allocation":
                    raise MaskReuseAllocationPause(
                        f"allocation-limited native time limit in {phase}", self.records
                    )
                raise MaskReuseSolverError(f"unresolved calibration solve in {phase}", self.records)
            self._check_allocation()
            # Generated risk rows must be checked before fixing every objective,
            # including later tie-breaks that can move to a different assignment.
            separate = getattr(problem, "_mask_reuse_separate_constraints", None)
            added = separate is not None and separate()
            self._remaining_seconds()
            self._check_allocation()
            if not added:
                fixed_risk = getattr(problem, "_mask_reuse_fixed_risk", None)
                if fixed_risk is not None:
                    variable, bound = fixed_risk
                    actual = self._discrete_risk(problem, variable)
                    if actual > bound + Fraction(8 * math.ulp(float(bound))):
                        if self._retry_without_warm_start(problem):
                            continue
                        raise MaskReuseSolverFailure(
                            f"later tie-break worsened the fixed risk in {phase}", self.records
                        )
                return True

    @staticmethod
    def _discrete_risk(problem, variable):
        """Evaluate original prompt-risk rows using the selected integer decisions."""
        required = Fraction(0)
        for row in problem.constraints.values():
            coefficient = row.get(variable, 0)
            if coefficient >= 0 or row.sense != pulp.LpConstraintLE:
                continue
            others = [(v, c) for v, c in row.items() if v is not variable]
            if any(v.cat != pulp.LpInteger for v, _ in others):
                continue
            numerator = Fraction(row.constant) + sum(
                (Fraction(c) * round(v.varValue) for v, c in others if round(v.varValue)),
                Fraction(0),
            )
            required = max(required, -numerator / Fraction(coefficient))
        return required

    def require_work_target(
        self, skipped_tiles: int, minimum_tiles: int | None, *, phase: str
    ) -> None:
        """Check exact work after extracting a discrete policy from solver values."""
        if minimum_tiles is not None and skipped_tiles < minimum_tiles:
            raise MaskReuseSolverFailure(
                f"invalid discrete work target in {phase}: "
                f"skipped_tiles={skipped_tiles}, minimum_tiles={minimum_tiles}",
                self.records,
            )

    def minimize_and_fix(self, problem: pulp.LpProblem, expression, phase: str, *, integral: bool):
        if not self.minimize(problem, expression, phase):
            return None
        value = float(pulp.value(expression) or 0.0)
        if not math.isfinite(value):
            raise MaskReuseSolverFailure(f"nonfinite optimum in {phase}", self.records)
        if integral:
            if abs(value - round(value)) > 1e-5:
                # This preview is not the feasibility/optimality certificate below.
                terms = pulp.lpSum([expression])
                rounded_objective = (
                    int(terms.constant) if terms.constant == int(terms.constant) else None
                )
                maximum_residual = 0.0
                for variable, coefficient in terms.items():
                    if coefficient == 0:
                        continue
                    if variable.cat == pulp.LpInteger:
                        column_value = float(variable.value())
                        rounded_column = round(column_value)
                        maximum_residual = max(maximum_residual, abs(column_value - rounded_column))
                        if rounded_objective is not None and coefficient == int(coefficient):
                            rounded_objective += int(coefficient) * rounded_column
                        else:
                            rounded_objective = None
                    else:
                        rounded_objective = None
                self.records[-1]["integer_objective_diagnostics"] = {
                    "diagnostic_only": True,
                    "raw_objective": value,
                    "nearest_integer": round(value),
                    "objective_distance": abs(value - round(value)),
                    "objective_tolerance": 1e-5,
                    "integer_variable_max_residual": maximum_residual,
                    "rounded_integer_objective": rounded_objective,
                }
                certified = None
                if phase == "fix_cost" and self.config.backend == "highs":
                    # Unsupported native metadata cannot repair the original failure.
                    with suppress(
                        AttributeError,
                        IndexError,
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        certified = self._certify_integer_cost(problem, terms)
                self._remaining_seconds()
                self._check_allocation()
                if certified is None:
                    raise MaskReuseSolverFailure(f"nonintegral optimum in {phase}", self.records)
                value = certified
            value = round(value)
            problem += pulp.lpSum([expression]) == value, phase
        else:
            if isinstance(expression, pulp.LpVariable):
                # CBC's solution file rounds the continuous epigraph value.
                # Fix the objective of the actual selected discrete policy.
                exact_value = self._discrete_risk(problem, expression)
                problem._mask_reuse_fixed_risk = expression, exact_value
                value = float(exact_value)
            problem += (
                pulp.lpSum([expression]) <= value,
                phase,
            )
        return value

    def _certify_integer_cost(self, problem, expression):
        """Certify a private integer point using exact feasibility and a native MIP bound."""
        # Optional backend; this path is entered only after a HiGHS fix_cost solve.
        import highspy

        if problem is None:
            return None
        solver = self._highs_solver
        native = problem.solverModel
        if (
            solver is None
            or not solver.mip
            or native is not solver._native
            or not solver._owns_problem(problem)
            or problem.sense != pulp.LpMinimize
            or expression.constant != 0
            or problem.status != pulp.LpStatusOptimal
            or problem.sol_status != pulp.LpSolutionOptimal
            or native.getModelStatus() != highspy.HighsModelStatus.kOptimal
        ):
            return None
        for name, expected in (
            ("solve_relaxation", False),
            ("mip_abs_gap", 0.0),
            ("mip_rel_gap", 0.0),
            ("mip_feasibility_tolerance", 1e-9),
            ("small_matrix_value", 1e-12),
            ("threads", 1),
            ("parallel", "off"),
            ("random_seed", 1),
        ):
            status, actual = native.getOptionValue(name)
            if status != highspy.HighsStatus.kOk or actual != expected:
                return None
        solution, info, lp = native.getSolution(), native.getInfo(), native.getLp()
        variables, rows = problem.variables(), list(problem.constraints.values())
        raw = list(solution.col_value)
        costs, lower, upper = (
            [float(value) for value in values]
            for values in (lp.col_cost_, lp.col_lower_, lp.col_upper_)
        )
        integrality = list(lp.integrality_)
        row_lower, row_upper = (
            [float(value) for value in values] for values in (lp.row_lower_, lp.row_upper_)
        )
        if (
            not solution.value_valid
            or not info.valid
            or info.primal_solution_status != highspy.kSolutionStatusFeasible
            or info.mip_gap != 0.0
            or lp.sense_ != highspy.ObjSense.kMinimize
            or lp.offset_ != 0
            or lp.num_col_ != len(variables)
            or lp.num_row_ != len(rows)
            or any(
                len(values) != len(variables) for values in (raw, costs, lower, upper, integrality)
            )
            or len(row_lower) != len(rows)
            or len(row_upper) != len(rows)
        ):
            return None
        candidate, integer_cost = {}, 0
        for index, variable in enumerate(variables):
            self._remaining_seconds()
            self._check_allocation()
            value = float(raw[index])
            coefficient = expression.get(variable, 0)
            integer = variable.cat == pulp.LpInteger
            if (
                variable.index != index
                or not math.isfinite(value)
                or variable.value() != value
                or costs[index] != coefficient
                or problem.objective.get(variable, 0) != coefficient
                or integrality[index]
                != (highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous)
                or lower[index] != (-math.inf if variable.lowBound is None else variable.lowBound)
                or upper[index] != (math.inf if variable.upBound is None else variable.upBound)
            ):
                return None
            if coefficient and (
                not integer
                or not math.isfinite(coefficient)
                or coefficient != int(coefficient)
                or abs(coefficient) >= 2**53
                or float(coefficient) != coefficient
            ):
                return None
            if integer:
                rounded = round(value)
                if abs(value - rounded) > 1e-8 or abs(rounded) >= 2**53:
                    return None
                candidate[variable] = Fraction(rounded)
            else:
                candidate[variable] = Fraction(value)
            if (
                variable.lowBound is not None and candidate[variable] < Fraction(variable.lowBound)
            ) or (
                variable.upBound is not None and candidate[variable] > Fraction(variable.upBound)
            ):
                return None
            integer_cost += int(coefficient) * candidate[variable] if coefficient else 0
        if (
            problem.objective.constant != 0
            or abs(integer_cost) >= 2**53
            or any(variable not in candidate for variable in expression)
        ):
            return None

        def satisfied(row):
            residual = Fraction(row.constant) + sum(
                (
                    Fraction(coefficient) * candidate[variable]
                    for variable, coefficient in row.items()
                ),
                Fraction(0),
            )
            return residual == 0 if row.sense == pulp.LpConstraintEQ else residual * row.sense >= 0

        for index, row in enumerate(rows):
            self._remaining_seconds()
            self._check_allocation()
            lb, ub = row.getLb(), row.getUb()
            status, indices, coefficients = native.getRowEntries(index)
            expected_entries = {
                variable.index: coefficient for variable, coefficient in row.items() if coefficient
            }
            if (
                status != highspy.HighsStatus.kOk
                or row_lower[index] != (-math.inf if lb is None else lb)
                or row_upper[index] != (math.inf if ub is None else ub)
                or len(indices) != len(expected_entries)
                or len(coefficients) != len(expected_entries)
                or {int(i): float(c) for i, c in zip(indices, coefficients)} != expected_entries
                or not satisfied(row)
            ):
                return None
        omitted = getattr(problem, "_mask_reuse_omitted_risk_rows", None)
        if (
            getattr(problem, "_mask_reuse_separate_constraints", None) is not None
            and omitted is None
        ):
            return None
        if omitted is not None:
            for row in omitted():
                self._remaining_seconds()
                self._check_allocation()
                if not satisfied(row):
                    return None
        objective, bound = float(info.objective_function_value), float(info.mip_dual_bound)
        if (
            not math.isfinite(objective)
            or not math.isfinite(bound)
            or objective != native.getObjectiveValue()
            or objective != pulp.value(expression)
            or bound > objective
            or not integer_cost - 1 < Fraction(bound) <= integer_cost
        ):
            return None
        self.records[-1]["integer_cost_certificate"] = {
            "kind": "exact_point_native_bound_not_rational_global_proof",
            "integer_cost": int(integer_cost),
            "native_objective": objective,
            "native_bound": bound,
        }
        return int(integer_cost)

    def lexicographic(
        self, problem: pulp.LpProblem, expressions: Iterable, *, base: int, phase: str
    ) -> None:
        """Fix a unique digit tuple using bounded mixed-radix blocks, never huge weights."""
        # A constant digit cannot break a tie. Skipping it preserves the exact
        # tuple order and avoids thousands of no-op solves for dense fallback.
        digits = [term for term in expressions if len(pulp.lpSum([term]))]
        width = max(1, int(math.log(2**20, base)))
        for offset in range(0, len(digits), width):
            group = digits[offset : offset + width]
            expression = pulp.lpSum(
                digit * base ** (len(group) - i - 1) for i, digit in enumerate(group)
            )
            if (
                self.minimize_and_fix(problem, expression, f"{phase}_{offset}", integral=True)
                is None
            ):
                raise MaskReuseSolverFailure(f"lost feasibility during {phase}", self.records)

    def provenance(self) -> dict[str, object]:
        return {
            **asdict(self.config),
            "threads": 1,
            "random_seed": 1,
            "relative_gap": 0.0,
            "absolute_gap": 0.0,
            "pulp_version": pulp.__version__,
            "phases": [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "elapsed_seconds",
                        "native_status",
                        "native_limit_source",
                        "native_limit_seconds",
                    }
                }
                for row in self.records
            ],
        }
