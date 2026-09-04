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

"""Versioned expected-result verification for reproducible campaigns."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["ExpectationResult", "verify_expected_results"]

_CONTRACT_SCHEMA = "modelopt.puzzletron-expected-results/v1"
_OBSERVATION_SCHEMA = "modelopt.puzzletron-reference-observation/v1"
_RESULT_SCHEMA = "modelopt.puzzletron-expectation-comparison/v1"
_MAX_JSON_BYTES = 16 << 20
_CONTRACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class _InvalidExpectation(ValueError):
    pass


@dataclass(frozen=True)
class ExpectationResult:
    """One expectation-verification outcome and its process exit contract."""

    status: str
    exit_code: int
    comparison_path: str | None
    fields: tuple[dict[str, Any], ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "expectation_status": self.status,
            "expectation_exit_code": self.exit_code,
            "expectation_comparison_path": self.comparison_path,
            "expectation_reason": self.reason,
            "expectation_fields": list(self.fields),
        }


def _load_json_value(path: Path, *, label: str) -> Any:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
    except OSError as error:
        raise _InvalidExpectation(f"{label} is unreadable: {path}") from error
    if len(raw) > _MAX_JSON_BYTES:
        raise _InvalidExpectation(f"{label} exceeds {_MAX_JSON_BYTES} bytes: {path}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _InvalidExpectation(f"{label} is not valid JSON: {path}") from error
    return payload


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json_value(path, label=label)
    if not isinstance(payload, dict):
        raise _InvalidExpectation(f"{label} must contain an object: {path}")
    return payload


def _confined_path(
    root: Path,
    value: object,
    *,
    label: str,
    require_relative: bool,
) -> Path:
    if not isinstance(value, str) or not value:
        kind = "relative path" if require_relative else "path"
        raise _InvalidExpectation(f"{label} must be a non-empty {kind}")
    candidate = Path(value)
    if require_relative and candidate.is_absolute():
        raise _InvalidExpectation(f"{label} must be relative")
    resolved_root = root.resolve()
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (resolved_root / candidate).resolve()
    )
    if not resolved.is_relative_to(resolved_root):
        raise _InvalidExpectation(f"{label} escapes its allowed root")
    return resolved


def _relative_path(root: Path, value: object, *, label: str) -> Path:
    return _confined_path(root, value, label=label, require_relative=True)


def _pointer(payload: object, value: object, *, field_name: str) -> object:
    if not isinstance(value, str) or (value and not value.startswith("/")):
        raise _InvalidExpectation(f"field {field_name!r} has an invalid JSON pointer")
    current = payload
    if value == "":
        return current
    for raw_token in value[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise _InvalidExpectation(f"required field {field_name!r} is missing at {value!r}")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
                if index < 0 or str(index) != token:
                    raise ValueError
                current = current[index]
            except (ValueError, IndexError) as error:
                raise _InvalidExpectation(
                    f"required field {field_name!r} is missing at {value!r}"
                ) from error
        else:
            raise _InvalidExpectation(f"required field {field_name!r} is missing at {value!r}")
    return current


def _require_finite(value: object, *, field_name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _InvalidExpectation(f"required field {field_name!r} is nonfinite")
    if isinstance(value, Mapping):
        for nested in value.values():
            _require_finite(nested, field_name=field_name)
    elif isinstance(value, list):
        for nested in value:
            _require_finite(nested, field_name=field_name)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidExpectation(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise _InvalidExpectation(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise _InvalidExpectation(f"{label} must be a finite number")
    return parsed


def _nonnegative_number(value: object, *, label: str) -> float:
    parsed = _finite_number(value, label=label)
    if parsed < 0:
        raise _InvalidExpectation(f"{label} must be a nonnegative finite number")
    return parsed


def _exact_equal(actual: object, reference: object) -> bool:
    return json.dumps(
        actual,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        reference,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _bounded_comparison(
    *, field_name: str, actual: object, reference: object, rule: Mapping[str, Any]
) -> tuple[bool, float, float | None]:
    if (
        isinstance(actual, bool)
        or isinstance(reference, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(reference, (int, float))
    ):
        raise _InvalidExpectation(f"bounded field {field_name!r} must compare numbers")
    actual_value = _finite_number(actual, label=f"bounded field {field_name!r} actual")
    reference_value = _finite_number(
        reference,
        label=f"bounded field {field_name!r} reference",
    )
    tolerance = _nonnegative_number(
        rule.get("max_regression"), label=f"field {field_name!r} max_regression"
    )
    denominator_value = rule.get("denominator")
    denominator = None
    if denominator_value is not None:
        denominator = _nonnegative_number(
            denominator_value,
            label=f"field {field_name!r} denominator",
        )
        if denominator == 0:
            raise _InvalidExpectation(f"field {field_name!r} denominator must be positive")
        if not 0 <= actual_value <= denominator or not 0 <= reference_value <= denominator:
            raise _InvalidExpectation(
                f"bounded field {field_name!r} must be within its denominator"
            )
    direction = rule.get("direction")
    delta = actual_value - reference_value
    if direction == "higher-is-better":
        return delta >= -tolerance, delta, denominator
    if direction == "lower-is-better":
        return delta <= tolerance, delta, denominator
    if direction == "two-sided":
        return abs(delta) <= tolerance, delta, denominator
    raise _InvalidExpectation(f"bounded field {field_name!r} has an invalid direction")


def _validate_contract(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], list[object]]:
    if payload.get("schema") != _CONTRACT_SCHEMA:
        raise _InvalidExpectation(f"contract schema must be {_CONTRACT_SCHEMA}")
    contract_id = payload.get("id")
    if not isinstance(contract_id, str) or _CONTRACT_ID.fullmatch(contract_id) is None:
        raise _InvalidExpectation("contract id must be a filesystem-safe non-empty identifier")
    artifacts = payload.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or not artifacts
        or any(not isinstance(key, str) for key in artifacts)
    ):
        raise _InvalidExpectation(
            "contract artifacts must map names to JSON artifact specifications"
        )
    fields = payload.get("fields")
    if not isinstance(fields, list) or not fields:
        raise _InvalidExpectation("contract fields must be a non-empty list")
    return contract_id, artifacts, fields


def _load_artifact(name: str, specification: object, *, puzzle_dir: Path) -> Any:
    if isinstance(specification, str):
        path_value = specification
        follow: object = []
    elif isinstance(specification, Mapping):
        if set(specification) - {"path", "follow"} or "path" not in specification:
            raise _InvalidExpectation(
                f"artifact {name!r} must contain only path and optional follow pointers"
            )
        path_value = specification["path"]
        follow = specification.get("follow", [])
    else:
        raise _InvalidExpectation(f"artifact {name!r} has an invalid specification")
    if not isinstance(follow, list) or any(not isinstance(value, str) for value in follow):
        raise _InvalidExpectation(f"artifact {name!r} follow must be a list of JSON pointers")

    path = _relative_path(puzzle_dir, path_value, label=f"artifact {name!r}")
    payload = _load_json_value(path, label=f"artifact {name!r}")
    for index, pointer in enumerate(follow):
        next_path = _pointer(payload, pointer, field_name=f"artifact {name!r} follow[{index}]")
        path = _confined_path(
            puzzle_dir,
            next_path,
            label=f"artifact {name!r} follow[{index}]",
            require_relative=False,
        )
        payload = _load_json_value(path, label=f"artifact {name!r} follow[{index}]")
    return payload


def _compare(
    *, contract: Mapping[str, Any], observation: Mapping[str, Any], puzzle_dir: Path
) -> tuple[str, list[dict[str, Any]]]:
    contract_id, artifact_paths, rules = _validate_contract(contract)
    if observation.get("schema") != _OBSERVATION_SCHEMA:
        raise _InvalidExpectation(f"observation schema must be {_OBSERVATION_SCHEMA}")
    if observation.get("contract_id") != contract_id:
        raise _InvalidExpectation("observation contract_id differs from the selected contract")
    qualification = observation.get("qualification")
    if isinstance(qualification, Mapping) and str(qualification.get("status", "")).startswith(
        "pending-"
    ):
        raise _InvalidExpectation(
            f"reference observation is not qualified: {qualification.get('status')}"
        )
    reference_values = observation.get("values")
    if not isinstance(reference_values, Mapping):
        raise _InvalidExpectation("observation values must contain an object")

    artifacts = {
        name: _load_artifact(name, specification, puzzle_dir=puzzle_dir)
        for name, specification in artifact_paths.items()
    }
    compared: list[dict[str, Any]] = []
    seen: set[str] = set()
    regression = False
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            raise _InvalidExpectation("contract field entries must be objects")
        name = raw_rule.get("name")
        artifact = raw_rule.get("artifact")
        classification = raw_rule.get("classification")
        if not isinstance(name, str) or not name or name in seen:
            raise _InvalidExpectation("contract field names must be unique non-empty strings")
        seen.add(name)
        if artifact not in artifacts:
            raise _InvalidExpectation(f"field {name!r} names an unknown artifact")
        if classification not in {"exact", "bounded", "informational"}:
            raise _InvalidExpectation(f"field {name!r} has an invalid classification")
        actual = _pointer(artifacts[str(artifact)], raw_rule.get("pointer"), field_name=name)
        _require_finite(actual, field_name=name)
        reference_present = name in reference_values
        if classification != "informational" and not reference_present:
            raise _InvalidExpectation(f"observation is missing required field {name!r}")
        reference = reference_values.get(name)
        if reference_present:
            _require_finite(reference, field_name=name)

        passed = True
        delta: float | None = None
        denominator: float | None = None
        if classification == "exact":
            passed = _exact_equal(actual, reference)
        elif classification == "bounded":
            passed, delta, denominator = _bounded_comparison(
                field_name=name,
                actual=actual,
                reference=reference,
                rule=raw_rule,
            )
        elif (
            reference_present
            and isinstance(actual, (int, float))
            and isinstance(reference, (int, float))
        ):
            delta = _finite_number(
                actual, label=f"informational field {name!r} actual"
            ) - _finite_number(reference, label=f"informational field {name!r} reference")
        regression = regression or not passed
        compared.append(
            {
                "name": name,
                "classification": classification,
                "actual": actual,
                "reference": reference if reference_present else None,
                "passed": passed,
                **({"delta": delta} if delta is not None else {}),
                **({"denominator": denominator} if denominator is not None else {}),
            }
        )
    return ("regression" if regression else "passed"), compared


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_expected_results(
    contract_path: str | Path,
    *,
    puzzle_dir: str | Path,
    output_path: str | Path | None = None,
) -> ExpectationResult:
    """Project campaign artifacts and compare them with a versioned observation."""

    contract_path = Path(contract_path).expanduser().resolve()
    puzzle_dir = Path(puzzle_dir).expanduser().resolve()
    comparison_path: Path | None = None
    try:
        contract = _load_json(contract_path, label="expectation contract")
        contract_id, _, _ = _validate_contract(contract)
        comparison_path = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else puzzle_dir / "artifacts" / "expectations" / f"{contract_id}.json"
        )
        observation_path = _relative_path(
            contract_path.parent,
            contract.get("observation"),
            label="contract observation",
        )
        observation = _load_json(observation_path, label="reference observation")
        status, fields = _compare(
            contract=contract,
            observation=observation,
            puzzle_dir=puzzle_dir,
        )
        result = ExpectationResult(
            status=status,
            exit_code=1 if status == "regression" else 0,
            comparison_path=str(comparison_path),
            fields=tuple(fields),
        )
    except _InvalidExpectation as error:
        result = ExpectationResult(
            status="invalid",
            exit_code=2,
            comparison_path=str(comparison_path) if comparison_path is not None else None,
            reason=str(error),
        )
    if comparison_path is not None:
        try:
            _write_result(
                comparison_path,
                {
                    "schema": _RESULT_SCHEMA,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "reason": result.reason,
                    "fields": list(result.fields),
                },
            )
        except OSError as error:
            return ExpectationResult(
                status="invalid",
                exit_code=2,
                comparison_path=str(comparison_path),
                reason=f"comparison output is unwritable: {comparison_path}: {error}",
            )
    return result
