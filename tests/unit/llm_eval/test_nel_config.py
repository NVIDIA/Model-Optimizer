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

"""Behavioral tests for the NeMo Evaluator task-config compiler."""

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from examples.llm_eval import nel_config

REPO_ROOT = Path(__file__).resolve().parents[3]
NEL_CONFIG = REPO_ROOT / "examples" / "llm_eval" / "nel_config.py"
TASK_CATALOG = REPO_ROOT / "examples" / "llm_eval" / "task_contracts.yaml"

AIME_BINDINGS = {
    "aime_judge_model_id": "aime-judge",
    "aime_judge_url": "https://judge.example/v1",
}
HLE_BINDINGS = {
    "hle_judge_model_id": "hle-judge",
    "ns_judge_url": "https://judge.example/v1",
}
LCR_BINDINGS = {
    "lcr_judge_model_id": "lcr-judge",
    "lcr_parallelism": 16,
    "ns_judge_url": "https://judge.example/v1",
}
TAU2_BINDINGS = {
    "tau2_endpoint_url": "https://judge.example/v1/chat/completions",
    "tau2_judge_model_id": "tau2-judge",
    "tau2_parallelism": 32,
    "tau2_user_model_id": "tau2-user",
}


@pytest.fixture
def task_catalog() -> dict[str, Any]:
    catalog = yaml.safe_load(TASK_CATALOG.read_text(encoding="utf-8"))
    assert isinstance(catalog, dict)
    return catalog


def test_cli_adds_default_tasks_without_changing_the_base(tmp_path, task_catalog):
    base = {
        "defaults": [{"execution": "slurm/default"}, "_self_"],
        "evaluation": {
            "nemo_evaluator_config": {"config": {"params": {"parallelism": 8}}},
            "tasks": [{"name": "custom_smoke", "container": "example.invalid/smoke:1"}],
        },
    }
    base_path = tmp_path / "base.yaml"
    output_path = tmp_path / "compiled.yaml"
    _write_yaml(base_path, base)

    result = _run_compiler(base_path, "--output", output_path)

    assert result.returncode == 0, result.stderr
    assert _read_yaml(base_path) == base
    compiled = _read_yaml(output_path)
    assert compiled["defaults"] == base["defaults"]
    assert (
        compiled["evaluation"]["nemo_evaluator_config"]
        == base["evaluation"]["nemo_evaluator_config"]
    )
    assert compiled["evaluation"]["tasks"] == [
        *base["evaluation"]["tasks"],
        *[task_catalog["tasks"][name]["contract"] for name in task_catalog["default_tasks"]],
    ]
    livecodebench = next(
        task for task in compiled["evaluation"]["tasks"] if task["name"] == "ns_livecodebench"
    )
    livecodebench_extra = livecodebench["nemo_evaluator_config"]["config"]["params"]["extra"]
    assert livecodebench_extra["use_sandbox"] is True
    assert livecodebench_extra["args"] == "++eval_config.timeout_buffer=300"


def test_cli_preserves_requested_task_order(tmp_path):
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, {})

    result = _run_compiler(
        base_path,
        "--task",
        "ns_ifbench",
        "--task",
        "mmlu_pro_aa_v3",
    )

    assert result.returncode == 0, result.stderr
    tasks = yaml.safe_load(result.stdout)["evaluation"]["tasks"]
    assert [task["name"] for task in tasks] == ["ns_ifbench", "mmlu_pro_aa_v3"]


@pytest.mark.parametrize(
    ("task_name", "bindings"),
    [
        pytest.param("AIME_2025_aa_v2", AIME_BINDINGS, id="aime"),
        pytest.param("ns_hle_aa", HLE_BINDINGS, id="hle"),
        pytest.param("ns_aa_lcr", LCR_BINDINGS, id="lcr"),
        pytest.param("tau2_bench_telecom", TAU2_BINDINGS, id="tau2"),
    ],
)
def test_compile_applies_external_task_bindings(task_catalog, task_name, bindings):
    expected = _expected_contract(task_catalog, task_name, bindings)

    compiled = nel_config.compile_nel_config({}, [task_name], bindings)

    assert compiled["evaluation"]["tasks"] == [expected]


def test_compile_preserves_tau2_user_and_judger_contract():
    compiled = nel_config.compile_nel_config({}, ["tau2_bench_telecom"], TAU2_BINDINGS)

    extra = compiled["evaluation"]["tasks"][0]["nemo_evaluator_config"]["config"]["params"]["extra"]
    assert extra["user"] == {
        "model_id": "tau2-user",
        "url": "https://judge.example/v1/chat/completions",
        "api_key": "INFERENCE_API_KEY",
    }
    assert extra["judger"] == {
        "model_id": "tau2-judge",
        "url": "https://judge.example/v1/chat/completions",
        "api_key": "INFERENCE_API_KEY",
    }
    assert "judge" not in extra


@pytest.mark.parametrize(
    ("task_name", "bindings", "missing_value", "expected_option"),
    [
        pytest.param(
            "AIME_2025_aa_v2",
            AIME_BINDINGS,
            "aime_judge_model_id",
            "--aime-judge-model-id",
            id="string",
        ),
        pytest.param(
            "ns_aa_lcr",
            LCR_BINDINGS,
            "lcr_parallelism",
            "--lcr-parallelism",
            id="integer",
        ),
    ],
)
def test_compile_requires_external_task_bindings(
    task_name, bindings, missing_value, expected_option
):
    incomplete = bindings.copy()
    incomplete.pop(missing_value)

    with pytest.raises(ValueError, match=rf"requires {expected_option}"):
        nel_config.compile_nel_config({}, [task_name], incomplete)


@pytest.mark.parametrize(
    "endpoint",
    [
        pytest.param("https://judge.example/not-v1", id="wrong-path"),
        pytest.param("https://" + "user:secret" + "@judge.example/v1", id="credentials"),
        pytest.param("https://judge.example:bad/v1", id="malformed-port"),
    ],
)
def test_compile_rejects_invalid_external_endpoints(endpoint):
    bindings = HLE_BINDINGS | {"ns_judge_url": endpoint}

    with pytest.raises(ValueError, match=r"HTTP\(S\) endpoint ending in /v1") as error:
        nel_config.compile_nel_config({}, ["ns_hle_aa"], bindings)

    assert "user:secret" not in str(error.value)


@pytest.mark.parametrize(
    ("parallelism", "message"),
    [
        pytest.param(0, "must be greater than zero", id="nonpositive"),
        pytest.param(513, "must not exceed 512", id="above-task-limit"),
    ],
)
def test_compile_validates_external_parallelism(parallelism, message):
    bindings = TAU2_BINDINGS | {"tau2_parallelism": parallelism}

    with pytest.raises(ValueError, match=message):
        nel_config.compile_nel_config({}, ["tau2_bench_telecom"], bindings)


def test_compile_rejects_boolean_parallelism():
    bindings = LCR_BINDINGS | {"lcr_parallelism": True}

    with pytest.raises(ValueError, match="Invalid parallelism binding"):
        nel_config.compile_nel_config({}, ["ns_aa_lcr"], bindings)


def test_compile_rejects_options_for_unselected_tasks():
    with pytest.raises(ValueError, match=r"not used.*--hle-judge-model-id"):
        nel_config.compile_nel_config(
            {},
            ["ns_ifbench"],
            {"hle_judge_model_id": HLE_BINDINGS["hle_judge_model_id"]},
        )


@pytest.mark.parametrize(
    ("base", "message"),
    [
        pytest.param({"evaluation": []}, "evaluation must be a mapping", id="evaluation"),
        pytest.param(
            {"evaluation": {"tasks": {}}},
            "evaluation.tasks must be a list",
            id="tasks",
        ),
        pytest.param(
            {"evaluation": {"tasks": [{"name": "same"}, {"name": "same"}]}},
            "evaluation.tasks contains duplicate task",
            id="duplicate-task",
        ),
    ],
)
def test_compile_rejects_malformed_base_config(base, message):
    with pytest.raises(ValueError, match=message):
        nel_config.compile_nel_config(base)


def test_compile_accepts_matching_task_and_rejects_conflict():
    compiled = nel_config.compile_nel_config({}, ["ns_hle_aa"], HLE_BINDINGS)

    assert nel_config.compile_nel_config(compiled, ["ns_hle_aa"], HLE_BINDINGS) == compiled

    conflicting = copy.deepcopy(compiled)
    conflicting["evaluation"]["tasks"][0]["container"] = "example.invalid/different:1"
    with pytest.raises(ValueError, match="conflicts with the maintained canonical contract"):
        nel_config.compile_nel_config(conflicting, ["ns_hle_aa"], HLE_BINDINGS)


def test_cli_does_not_interpolate_host_secrets(tmp_path, monkeypatch):
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, {})
    monkeypatch.setenv("JUDGE_API_KEY", "secret-sentinel")

    result = _run_compiler(
        base_path,
        "--task",
        "AIME_2025_aa_v2",
        "--aime-judge-model-id",
        AIME_BINDINGS["aime_judge_model_id"],
        "--aime-judge-url",
        AIME_BINDINGS["aime_judge_url"],
    )

    assert result.returncode == 0, result.stderr
    assert "secret-sentinel" not in result.stdout
    assert "host:JUDGE_API_KEY" in result.stdout


def test_cli_refuses_to_overwrite_output(tmp_path):
    base_path = tmp_path / "base.yaml"
    output_path = tmp_path / "compiled.yaml"
    _write_yaml(base_path, {})
    output_path.write_text("preserve me\n", encoding="utf-8")

    result = _run_compiler(base_path, "--output", output_path)

    assert result.returncode == 2
    assert "File exists" in result.stderr
    assert output_path.read_text(encoding="utf-8") == "preserve me\n"


def test_catalog_excludes_reduced_run_limits(task_catalog):
    assert "limit_samples" not in yaml.safe_dump(task_catalog)


def _expected_contract(
    task_catalog: dict[str, Any], task_name: str, bindings: dict[str, str | int]
) -> dict[str, Any]:
    entry = task_catalog["tasks"][task_name]
    expected = copy.deepcopy(entry["contract"])
    for binding in entry.get("bindings", []):
        cursor = expected
        for key in binding["path"][:-1]:
            cursor = cursor[key]
        cursor[binding["path"][-1]] = bindings[binding["value"]]
    return expected


def _run_compiler(base_path: Path, *args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(NEL_CONFIG), "--base-config", str(base_path), *map(str, args)],
        capture_output=True,
        check=False,
        text=True,
    )


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
