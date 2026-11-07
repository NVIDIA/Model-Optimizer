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
from types import SimpleNamespace

import pytest
import yaml
from modelopt_launcher.workflow import load_workflow

_LAUNCHER = Path(__file__).resolve().parents[1]
_EXAMPLE = _LAUNCHER / "examples/smoke/workflow_v2.yaml"
_PROFILE = _LAUNCHER / "examples/smoke/local_profile.yaml"


@pytest.mark.parametrize(
    ("missing_dataset", "linked"), [(False, False), (True, False), (False, True)]
)
def test_hf_inputs_pin_every_revision_before_downloading(
    tmp_path, monkeypatch, missing_dataset, linked
):
    """Model and dataset snapshots use exact commits; an unresolved input copies no bytes."""
    hub = pytest.importorskip("huggingface_hub")
    from modelopt_launcher.inputs import resolve_local

    def info(name, *, repo_type, revision):
        if missing_dataset and repo_type == "dataset":
            raise ValueError("Dataset unavailable")
        return SimpleNamespace(sha=("a" if repo_type == "model" else "b") * 40)

    def download(name, *, repo_type, revision, local_dir=None, cache_dir=None, **kwargs):
        path = Path(local_dir) if local_dir else Path(cache_dir) / repo_type / revision
        path.mkdir(parents=True)
        data = json.dumps({"kind": repo_type, "commit": revision})
        if linked:
            blob = tmp_path / "blob"
            blob.write_text(data)
            (path / "data.json").symlink_to(blob)
        else:
            (path / "data.json").write_text(data)
        return str(path)

    monkeypatch.setattr(hub, "HfApi", lambda **_: SimpleNamespace(repo_info=info))
    monkeypatch.setattr(hub, "snapshot_download", download)
    request = {
        "schema_version": 2,
        "hf_local": str(tmp_path / "cache"),
        "inputs": {
            "/inputs/model": {"ref": "Org/Model@branch"},
            "/inputs/data": {"ref": "Org/Data", "kind": "dataset"},
        },
    }
    if missing_dataset:
        with pytest.raises(ValueError):
            resolve_local(request)
        assert not (tmp_path / "cache").exists()
    elif linked:
        with pytest.raises(ValueError, match="self-contained"):
            resolve_local(request)
    else:
        result = resolve_local(request)
        assert result["status"] == "ready"
        for item, kind, sha in zip(result["inputs"], ("model", "dataset"), ("a", "b")):
            assert item["version"] == sha * 40
            assert json.loads((Path(item["path"]) / "data.json").read_text()) == {
                "kind": kind,
                "commit": sha * 40,
            }


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


@pytest.mark.parametrize("tracked", [False, True])
def test_named_dag_real_cli_and_export(tmp_path, tracked):
    """The portable spec survives export, branches concurrently and joins real outputs."""
    exported = tmp_path / "resolved.yaml"
    example, profile = _EXAMPLE, _PROFILE
    if tracked:
        pytest.importorskip("sqlalchemy")
        mlflow = pytest.importorskip("mlflow.tracking")
        uri = "sqlite:///" + str(tmp_path / "tracking.db")
        client = mlflow.MlflowClient(tracking_uri=uri)
        client.create_experiment(
            "ModelOpt workflows", artifact_location=(tmp_path / "mlruns").as_uri()
        )
        data = yaml.safe_load((_LAUNCHER / "examples/smoke/artifact_workflow.yaml").read_text())
        data["parameters"]["model_name"] = "test-checkpoint"
        example, profile = tmp_path / "workflow.yaml", tmp_path / "profile.yaml"
        example.write_text(yaml.safe_dump(data))
        profile.write_text(
            yaml.safe_dump(
                {
                    "executor": "local",
                    "tracking_uri": uri,
                    "artifact_store": str(tmp_path / "store"),
                }
            )
        )
    result = _cli(
        [
            "--yaml",
            str(example),
            "--profile",
            str(profile),
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
    result = _cli(["--yaml", str(exported), "--profile", str(profile), "--yes"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    manifests = list((tmp_path / "experiments").rglob("metadata.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "succeeded"
    assert set(manifest["tasks"]) == {"produce", "accuracy", "speed", "report"}
    assert manifest["tasks"]["accuracy"]["depends_on"] == ["produce"]
    summary = json.loads(Path(manifest["outputs"]["summary"]).read_text())
    assert summary == {"synthetic": True, "accuracy": 11, "speed": 11}
    if tracked:
        root = Path(manifest["workspace"]["host_path"]) / ".artifacts"
        produced = json.loads((root / "produce.json").read_text())
        model = produced["outputs"]["model"]
        assert model["model_uri"] == "models:/test-checkpoint/1"
        for task in ("accuracy", "speed"):
            consumed = json.loads((root / f"{task}.json").read_text())
            run = client.get_run(consumed["run_id"])
            assert run.info.status == "FINISHED"
            assert run.inputs.model_inputs[0].model_id == model["model_id"]
            assert consumed["inputs"][0]["model_uri"] == model["model_uri"]
        version = client.get_model_version("test-checkpoint", "1")
        assert version.run_id == produced["run_id"]
        from modelopt_launcher.inputs import resolve_local

        resolved = resolve_local(
            {
                "schema_version": 2,
                "host": None,
                "hf_local": str(tmp_path / "cache"),
                "tracking_uri": uri,
                "inputs": {"/inputs/model": {"ref": model["model_uri"]}},
            }
        )
        snapshot = Path(resolved["inputs"][0]["path"])
        assert json.loads((snapshot / "checkpoint.json").read_text())["value"] == 11
        assert resolved["inputs"][0]["model_id"] == model["model_id"]


@pytest.mark.parametrize("failure", ["command", "missing_output", "multiline", "publication"])
def test_failed_producer_blocks_consumer(tmp_path, failure):
    """A missing output or a failed producer prevents downstream side effects."""
    workflow: dict = {
        "api_version": "modelopt/v2",
        "name": "failure_test",
        "tasks": {
            "produce": {
                "inline": {
                    "command": "false",
                    "missing_output": "true",
                    "multiline": 'false\nprintf test > "$RESULT"',
                    "publication": 'touch "$RESULT"',
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
    profile = _PROFILE
    if failure == "publication":
        workflow["tasks"]["produce"]["outputs"]["checkpoint"]["publish"] = {
            "ref": "test/conflict",
            "version": "v1",
        }
        store = tmp_path / "store/test/conflict/v1"
        store.mkdir(parents=True)
        (store / "manifest.json").write_text("{}")
        profile = tmp_path / "profile.yaml"
        profile.write_text(
            yaml.safe_dump({"executor": "local", "artifact_store": str(tmp_path / "store")})
        )
    spec.write_text(yaml.safe_dump(workflow))
    result = _cli(["--yaml", str(spec), "--profile", str(profile), "--yes"], tmp_path)
    assert result.returncode != 0
    assert not list(tmp_path.rglob("consumer-ran"))
    manifests = list((tmp_path / "experiments").rglob("metadata.json"))
    assert len(manifests) == 1  # Failure retains the run identity and graph.
    assert json.loads(manifests[0].read_text())["status"] == "failed"


@pytest.mark.parametrize("fail_rank", [False, True])
def test_publication_waits_for_every_payload_rank(tmp_path, fail_rank):
    """A leader publishes once; any failed rank prevents a successful artifact manifest."""
    output = tmp_path / "workspace/outputs/train/model"
    output.mkdir(parents=True)
    request = {
        "task": "train",
        "workspace": str(tmp_path / "workspace"),
        "workers": 2,
        "command": 'mkdir -p "$RESULT"; '
        + ('if [ "$SLURM_PROCID" = 1 ]; then exit 7; fi; ' if fail_rank else "")
        + 'echo "$SLURM_PROCID" > "$RESULT/rank-$SLURM_PROCID.txt"',
        "inputs": {},
        "assets": [],
        "artifact_store": str(tmp_path / "store"),
        "outputs": {"model": {"path": "model", "publish": {"ref": "test/ranks", "version": "v1"}}},
    }
    processes = []
    try:
        for rank in range(2):
            env = dict(os.environ, SLURM_PROCID=str(rank), RESULT=str(output))
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(_LAUNCHER / "common/workflow_artifacts.py"),
                        json.dumps(request),
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        codes = []
        for process in processes:
            process.communicate(timeout=10)
            codes.append(process.returncode)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert all(code != 0 for code in codes) if fail_rank else codes == [0, 0]
    published = tmp_path / "store/test/ranks/v1/data"
    if fail_rank:
        assert not published.exists()
    else:
        assert sorted(path.name for path in published.iterdir()) == ["rank-0.txt", "rank-1.txt"]


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
