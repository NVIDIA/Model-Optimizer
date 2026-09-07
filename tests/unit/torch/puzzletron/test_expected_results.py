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

"""Behavioral tests for versioned campaign expectation verification."""

import json
from pathlib import Path
from types import SimpleNamespace

from examples.puzzletron import orchestrate
from examples.puzzletron.expectations import verify_expected_results
from examples.puzzletron.verify_expected_results import main as verification_main


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _contract(tmp_path: Path) -> Path:
    path = tmp_path / "expected" / "contract.json"
    _write(
        path,
        {
            "schema": "modelopt.puzzletron-expected-results/v1",
            "id": "smoke-v1",
            "observation": "observation.json",
            "artifacts": {"result": "artifacts/result.json"},
            "fields": [
                {
                    "name": "identity",
                    "artifact": "result",
                    "pointer": "/identity",
                    "classification": "exact",
                },
                {
                    "name": "correct_rows",
                    "artifact": "result",
                    "pointer": "/correct_rows",
                    "classification": "bounded",
                    "direction": "higher-is-better",
                    "max_regression": 1,
                    "denominator": 8,
                },
                {
                    "name": "timing_ms",
                    "artifact": "result",
                    "pointer": "/timing_ms",
                    "classification": "informational",
                },
            ],
        },
    )
    return path


def test_expectation_verifier_projects_fields_and_allows_one_directional_row_flip(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {"profile": "smoke-native-v1"}, "correct_rows": 7},
        },
    )
    run = tmp_path / "run"
    _write(
        run / "artifacts/result.json",
        {
            "identity": {"profile": "smoke-native-v1"},
            "correct_rows": 6,
            "timing_ms": 12.5,
        },
    )

    result = verify_expected_results(contract, puzzle_dir=run)

    assert result.status == "passed"
    assert result.exit_code == 0
    assert [field["classification"] for field in result.fields] == [
        "exact",
        "bounded",
        "informational",
    ]
    assert result.fields[1]["denominator"] == 8
    assert json.loads(Path(result.comparison_path).read_text())["status"] == "passed"


def test_expectation_verifier_distinguishes_regression_from_invalid_evidence(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {"profile": "smoke-native-v1"}, "correct_rows": 7},
        },
    )
    run = tmp_path / "run"
    result_path = run / "artifacts/result.json"
    _write(
        result_path,
        {
            "identity": {"profile": "smoke-native-v1"},
            "correct_rows": 5,
            "timing_ms": 12.5,
        },
    )
    assert verify_expected_results(contract, puzzle_dir=run).exit_code == 1

    _write(result_path, {"identity": {"profile": "smoke-native-v1"}, "correct_rows": 7})
    invalid = verify_expected_results(contract, puzzle_dir=run)
    assert invalid.status == "invalid"
    assert invalid.exit_code == 2
    assert "timing_ms" in invalid.reason


def test_expectation_verifier_rejects_invalid_denominator_evidence(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {"profile": "smoke-native-v1"}, "correct_rows": 7},
        },
    )
    run = tmp_path / "run"
    _write(
        run / "artifacts/result.json",
        {
            "identity": {"profile": "smoke-native-v1"},
            "correct_rows": 9,
            "timing_ms": 12.5,
        },
    )

    invalid = verify_expected_results(contract, puzzle_dir=run)

    assert invalid.status == "invalid"
    assert invalid.exit_code == 2
    assert "denominator" in invalid.reason
    assert Path(invalid.comparison_path).is_file()


def test_expectation_verifier_rejects_unsafe_contract_identifier(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    payload = json.loads(contract.read_text())
    payload["id"] = "../outside"
    _write(contract, payload)

    invalid = verify_expected_results(contract, puzzle_dir=tmp_path / "run")

    assert invalid.status == "invalid"
    assert invalid.exit_code == 2
    assert invalid.comparison_path is None


def test_expectation_verifier_follows_root_confined_artifact_pointers(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    contract_payload = json.loads(contract.read_text())
    contract_payload["artifacts"]["curve"] = {
        "path": "artifacts/summary.json",
        "follow": ["/observations_path", "/0/artifacts/result_manifest_path"],
    }
    contract_payload["fields"].append(
        {
            "name": "profile",
            "artifact": "curve",
            "pointer": "/evaluation_identity/profile",
            "classification": "exact",
        }
    )
    _write(contract, contract_payload)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {
                "identity": {"profile": "smoke-native-v1"},
                "correct_rows": 7,
                "profile": "smoke-native-v1",
            },
        },
    )
    run = tmp_path / "run"
    manifest = run / "artifacts/curve.json"
    observations = run / "artifacts/observations.json"
    _write(manifest, {"evaluation_identity": {"profile": "smoke-native-v1"}})
    _write(observations, [{"artifacts": {"result_manifest_path": str(manifest)}}])
    _write(run / "artifacts/summary.json", {"observations_path": str(observations)})
    _write(
        run / "artifacts/result.json",
        {
            "identity": {"profile": "smoke-native-v1"},
            "correct_rows": 7,
            "timing_ms": 12.5,
        },
    )

    result = verify_expected_results(contract, puzzle_dir=run)

    assert result.status == "passed"
    assert result.fields[-1]["actual"] == "smoke-native-v1"


def test_expectation_verifier_rejects_artifact_pointer_escape(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    payload = json.loads(contract.read_text())
    payload["artifacts"]["escaped"] = {
        "path": "artifacts/summary.json",
        "follow": ["/observations_path"],
    }
    _write(contract, payload)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {}, "correct_rows": 0},
        },
    )
    run = tmp_path / "run"
    _write(run / "artifacts/result.json", {"identity": {}, "correct_rows": 0, "timing_ms": 1})
    _write(run / "artifacts/summary.json", {"observations_path": str(tmp_path / "outside.json")})

    invalid = verify_expected_results(contract, puzzle_dir=run)

    assert invalid.status == "invalid"
    assert invalid.exit_code == 2
    assert "escapes" in invalid.reason


def test_verification_cli_returns_result_exit_code_and_json(
    tmp_path: Path,
    capsys,
) -> None:
    contract = _contract(tmp_path)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {}, "correct_rows": 7},
        },
    )
    run = tmp_path / "run"
    result_path = run / "artifacts/result.json"
    _write(result_path, {"identity": {}, "correct_rows": 5, "timing_ms": 1})
    argv = ["--contract", str(contract), "--puzzle-dir", str(run)]

    assert verification_main(argv) == 1
    assert json.loads(capsys.readouterr().out)["expectation_status"] == "regression"


def test_orchestrator_returns_expectation_regression_exit_code(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    contract = _contract(tmp_path)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {}, "correct_rows": 7},
        },
    )
    run = tmp_path / "run"
    _write(run / "artifacts/result.json", {"identity": {}, "correct_rows": 5, "timing_ms": 1})
    plan = SimpleNamespace(puzzle_dir=run, stages=())

    monkeypatch.setattr(orchestrate, "load_runner_config", lambda _path: object())
    monkeypatch.setattr(orchestrate, "validate_runner_ready", lambda _runner: None)
    monkeypatch.setattr(orchestrate, "load_execution_config", lambda _path: {})
    monkeypatch.setattr(orchestrate, "compile_campaign_plan", lambda **_kwargs: plan)

    class _Controller:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **_kwargs):
            return {
                "failed_stages": [],
                "failed_log_paths": {},
                "halted": False,
                "report_status": "completed",
            }

    monkeypatch.setattr(orchestrate, "CampaignController", _Controller)

    exit_code = orchestrate.main(
        [
            "--experiment",
            "experiment.yaml",
            "--runner",
            "runner.yaml",
            "--execution",
            "execution.yaml",
            "--expect",
            str(contract),
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["expectation_status"] == "regression"


def test_exact_fields_preserve_json_types(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    _write(
        contract.with_name("observation.json"),
        {
            "schema": "modelopt.puzzletron-reference-observation/v1",
            "contract_id": "smoke-v1",
            "values": {"identity": {"profile": 1}, "correct_rows": 7},
        },
    )
    run = tmp_path / "run"
    _write(
        run / "artifacts/result.json",
        {"identity": {"profile": True}, "correct_rows": 7, "timing_ms": 1},
    )

    result = verify_expected_results(contract, puzzle_dir=run)

    assert result.status == "regression"
    assert result.exit_code == 1
