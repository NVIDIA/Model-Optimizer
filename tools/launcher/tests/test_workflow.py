# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Behavioral coverage for v2 compilation and real CPU DAG execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from modelopt_launcher.workflow import load_workflow

_LAUNCHER = Path(__file__).resolve().parents[1]
_EXAMPLE = _LAUNCHER / "examples/smoke/workflow_v2.yaml"
_PROFILE = _LAUNCHER / "examples/smoke/local_profile.yaml"


def _cli(arguments, tmp_path):
    env = {key: os.environ[key] for key in ("PATH", "LANG") if key in os.environ}
    env.update(HOME=str(tmp_path), NEMORUN_HOME=str(tmp_path))
    return subprocess.run(
        [sys.executable, "-m", "modelopt_launcher.launch", *arguments],
        cwd=_LAUNCHER.parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_named_dag_real_cli_and_export(tmp_path):
    """The portable spec survives export, branches concurrently and joins real outputs."""
    exported = tmp_path / "resolved.yaml"
    result = _cli(
        [
            "--yaml",
            str(_EXAMPLE),
            "--profile",
            str(_PROFILE),
            "--param",
            "value=11",
            "--to-yaml",
            str(exported),
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    plan = yaml.safe_load(exported.read_text())
    assert plan["parameters"]["value"] == 11
    assert plan["tasks"]["report"]["depends_on"] == ["accuracy", "speed"]
    # The exported workflow remains executable with a separately supplied profile.
    result = _cli(["--yaml", str(exported), "--profile", str(_PROFILE), "--yes"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    manifests = list((tmp_path / "experiments").rglob("metadata.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "succeeded"
    assert set(manifest["tasks"]) == {"produce", "accuracy", "speed", "report"}
    assert manifest["tasks"]["accuracy"]["depends_on"] == ["produce"]
    summary = json.loads(Path(manifest["outputs"]["summary"]).read_text())
    assert summary == {"synthetic": True, "accuracy": 11, "speed": 11}


@pytest.mark.parametrize("failure", ["command", "missing_output", "multiline"])
def test_failed_producer_blocks_consumer(tmp_path, failure):
    """A missing output or a failed producer prevents downstream side effects."""
    workflow = {
        "api_version": "modelopt/v2",
        "name": "failure_test",
        "tasks": {
            "produce": {
                "inline": {
                    "command": "false",
                    "missing_output": "true",
                    "multiline": 'false\nprintf test > "$RESULT"',
                }[failure],
                "outputs": {"checkpoint": {"path": "missing.json", "env": "RESULT"}},
            },
            "consume": {
                "depends_on": ["produce"],
                "inline": 'touch "$MODELOPT_WORKSPACE/consumer-ran"',
            },
        },
    }
    spec = tmp_path / "failure.yaml"
    spec.write_text(yaml.safe_dump(workflow))
    result = _cli(["--yaml", str(spec), "--profile", str(_PROFILE), "--yes"], tmp_path)
    assert result.returncode != 0
    assert not list(tmp_path.rglob("consumer-ran"))
    manifests = list((tmp_path / "experiments").rglob("metadata.json"))
    assert len(manifests) == 1  # Failure retains the run identity and graph.
    assert json.loads(manifests[0].read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "defect",
    [
        "cycle",
        "unknown_dependency",
        "unknown_output",
        "bad_mount",
        "overlapping_inputs",
        "escaping_output",
        "bad_env",
        "unknown_field",
        "bad_parameter",
        "bad_version",
        "bad_dependency_type",
        "unknown_workflow_output",
    ],
)
def test_invalid_workflow_fails_before_side_effects(tmp_path, defect):
    """Bad specs are refused by compilation, before provider or executor calls."""
    data = yaml.safe_load(_EXAMPLE.read_text())
    if defect == "cycle":
        data["tasks"]["produce"]["depends_on"] = ["report"]
    elif defect == "unknown_dependency":
        data["tasks"]["produce"]["depends_on"] = ["missing"]
    elif defect == "unknown_output":
        data["tasks"]["accuracy"]["inputs"]["/inputs/model"]["from"] = "produce.missing"
    elif defect == "bad_mount":
        data["tasks"]["accuracy"]["inputs"] = {"/inputs/../model": {"from": "produce.checkpoint"}}
    elif defect == "overlapping_inputs":
        data["tasks"]["accuracy"]["inputs"]["/inputs/model/nested"] = {"from": "produce.checkpoint"}
    elif defect == "escaping_output":
        data["tasks"]["produce"]["outputs"]["checkpoint"]["path"] = "../outside"
    elif defect == "bad_env":
        data["tasks"]["produce"]["environment"]["RESULT"] = "override"
    elif defect == "unknown_field":
        data["tasks"]["produce"]["retriez"] = 2
    elif defect == "bad_parameter":
        data["tasks"]["produce"]["inline"] = "echo ${parameters.missing}"
    elif defect == "bad_version":
        data["api_version"] = "modelopt/v3"
    elif defect == "bad_dependency_type":
        data["tasks"]["produce"]["depends_on"] = "report"
    else:
        data["outputs"]["summary"]["from"] = "report.missing"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_workflow(path, _PROFILE)
    assert not (tmp_path / "experiments").exists()


def test_arbitrary_task_count_and_forward_references(tmp_path):
    """Named tasks are topologically ordered rather than limited to five slots."""
    data = {
        "api_version": "modelopt/v2",
        "name": "many_tasks",
        "tasks": {
            f"stage_{i}": {"inline": "true", "depends_on": [f"stage_{i - 1}"] if i else []}
            for i in reversed(range(8))
        },
    }
    spec = tmp_path / "many.yaml"
    spec.write_text(yaml.safe_dump(data, sort_keys=False))
    pipeline = load_workflow(spec, _PROFILE)["pipeline"]
    assert [task.name for task in pipeline.tasks] == [f"stage_{i}" for i in range(8)]
    spec.write_text(
        "api_version: modelopt/v2\nname: duplicate\ntasks:\n  same: {inline: true}\n  same: {inline: false}\n"
    )
    with pytest.raises(ValueError):
        load_workflow(spec, _PROFILE)
