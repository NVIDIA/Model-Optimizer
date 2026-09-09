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

"""Behavioral tests for the concise Puzzletron recipe and site contract."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import yaml

from examples.puzzletron import puzzletron as public_cli
from puzzletron_orchestrator import recipe_config
from puzzletron_orchestrator.recipe_config import (
    bundle_for_run_root,
    materialize_resolved_bundle,
    recipe_template,
    resolve_recipe_run,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _stable_source_identity(monkeypatch):
    monkeypatch.setattr(
        recipe_config,
        "_code_revision",
        lambda: {"revision": "a" * 40, "dirty": False},
    )


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _site(tmp_path: Path, *, max_nodes: int = 1, mode: str = "reusable_allocation") -> Path:
    return _write_yaml(
        tmp_path / "site.yaml",
        {
            "schema_version": 1,
            "site": {
                "kind": "slurm",
                "environment": {"repository": str(REPOSITORY_ROOT), "venv": ".venv"},
                "paths": {"hf_home": str(tmp_path / "hf")},
                "slurm": {"account": "test", "partition": "test"},
            },
            "resources": {
                "selected": {
                    "mode": mode,
                    "gpus_per_node": 8,
                    "max_nodes": max_nodes,
                }
            },
        },
    )


def _recipe(tmp_path: Path, **updates) -> Path:
    payload = recipe_template(resource_profile="selected", run_root=str(tmp_path / "run"))
    payload.update(updates)
    return _write_yaml(tmp_path / "recipe.yaml", payload)


def test_resolved_bundle_preserves_integrity_and_source_provenance(tmp_path):
    resolved = resolve_recipe_run(_recipe(tmp_path), _site(tmp_path))

    assert resolved.plan["execution_mode"] == "reusable_allocation"
    assert resolved.plan["stages"][-1]["stage_id"] == "post.params-90.best"
    assert resolved.provenance["route"]["value"] == "qwen3.5-0.8b/vlm-pruning/smoke"
    prerun = resolved.runner["runner"]["execution_contract"]["prerun_commands"]
    assert "_assert_worker_source" in prerun[-1]

    bundle = materialize_resolved_bundle(resolved, activate=True)
    assert bundle_for_run_root(tmp_path / "run") == bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["code"]["controller"]["revision"] == "a" * 40
    assert manifest["code"]["worker"]["revision"] == "a" * 40

    (bundle / "execution.yaml").write_text("execution: {}\n")
    with pytest.raises(RuntimeError, match="changed after sealing"):
        bundle_for_run_root(tmp_path / "run")


def test_bundle_identity_includes_authored_input_locations(tmp_path):
    recipe = recipe_template(resource_profile="selected", run_root=str(tmp_path / "run"))
    site = yaml.safe_load(_site(tmp_path).read_text())

    first = resolve_recipe_run(
        _write_yaml(tmp_path / "first.recipe.yaml", recipe),
        _write_yaml(tmp_path / "first.site.yaml", site),
    )
    second = resolve_recipe_run(
        _write_yaml(tmp_path / "second.recipe.yaml", recipe),
        _write_yaml(tmp_path / "second.site.yaml", site),
    )

    assert first.experiment == second.experiment
    assert first.bundle_id != second.bundle_id


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"resource_profiel": "selected"}, "did you mean"),
        (
            {"advanced": {"experiment": {"model.force_hf": False}}},
            "Duplicate no-op advanced value",
        ),
        ({"data": {"path": "/prepared/data"}}, "requires explicit immutable data values"),
        (
            {"advanced": {"experiment": {"depth_importance.eval_samples": 3}}},
            "targets inactive stage",
        ),
    ],
)
def test_closed_recipe_schema_rejects_unknown_duplicate_or_unused_values(tmp_path, update, message):
    payload = recipe_template(resource_profile="selected", run_root=str(tmp_path / "run"))
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        resolve_recipe_run(_write_yaml(tmp_path / "recipe.yaml", payload), _site(tmp_path))


@pytest.mark.parametrize(
    ("advanced", "message"),
    [
        ({"experiment": {"puzzle_dir": "/tmp/other"}}, "owned by the selected route"),
        ({"experiment": {"model": {"force_hf": False}}}, "owned by the selected route"),
        (
            {"execution": {"defaults": {"gpus_per_node": 4}}},
            "cannot override site-owned fields",
        ),
    ],
)
def test_recipe_cannot_override_route_or_site_owned_identity(tmp_path, advanced, message):
    recipe = recipe_template(resource_profile="selected", run_root=str(tmp_path / "run"))
    recipe["advanced"] = advanced
    with pytest.raises(ValueError, match=message):
        resolve_recipe_run(_write_yaml(tmp_path / "recipe.yaml", recipe), _site(tmp_path))


def test_duplicate_yaml_keys_report_the_source_line(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "schema_version: 1\nname: first\nname: second\nmodel: qwen3.5-0.8b\n"
        "workflow: vlm-pruning\nmode: smoke\nrun_root: run\nresource_profile: selected\n"
    )
    with pytest.raises(ValueError, match="Duplicate YAML key 'name' at line 3"):
        resolve_recipe_run(recipe, _site(tmp_path))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            lambda site: site["site"]["environment"].update({"setup_env": 1}),
            "site.environment.setup_env",
        ),
        (
            lambda site: site["site"]["environment"].update(
                {"prerun_commands": ["export HF_HOME=/other/cache"]}
            ),
            "assigns HF_HOME",
        ),
        (
            lambda site: site["site"]["environment"].update({"container_mounts": "/data:/data"}),
            "container_mounts is unused",
        ),
        (
            lambda site: site["site"]["environment"].update({"source_revision": "main"}),
            "source_revision must be a full immutable Git commit",
        ),
    ],
)
def test_closed_site_schema_rejects_invalid_or_ignored_values(tmp_path, update, message):
    site = yaml.safe_load(_site(tmp_path).read_text())
    update(site)
    with pytest.raises((TypeError, ValueError), match=message):
        resolve_recipe_run(_recipe(tmp_path), _write_yaml(tmp_path / "site.yaml", site))


def test_baremetal_site_uses_the_same_recipe_contract(tmp_path):
    site = {
        "schema_version": 1,
        "site": {
            "kind": "baremetal",
            "environment": {"repository": str(REPOSITORY_ROOT), "venv": ".venv"},
            "paths": {"hf_home": str(tmp_path / "hf")},
            "baremetal": {"hosts": [{"hostname": "worker-a", "gpus": 8}]},
        },
        "resources": {"selected": {"mode": "per_attempt", "gpus_per_node": 8, "max_nodes": 1}},
    }
    resolved = resolve_recipe_run(_recipe(tmp_path), _write_yaml(tmp_path / "site.yaml", site))
    assert resolved.plan["runner_kind"] == "baremetal"
    assert resolved.runner["runner"]["inventory"]["hosts"] == [{"hostname": "worker-a", "gpus": 8}]


def test_environment_defaults_do_not_change_recipe_identity(tmp_path, monkeypatch):
    recipe = _recipe(tmp_path)
    site = _site(tmp_path)
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "first-hidden-value")
    first = resolve_recipe_run(recipe, site)
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "second-hidden-value")
    second = resolve_recipe_run(recipe, site)
    assert first.experiment == second.experiment
    assert first.bundle_id == second.bundle_id


def test_worker_revision_and_dirty_state_are_enforced(tmp_path, monkeypatch):
    site = yaml.safe_load(_site(tmp_path).read_text())
    site["site"]["environment"]["repository"] = str(tmp_path / "worker")
    site_path = _write_yaml(tmp_path / "site.yaml", site)
    monkeypatch.setattr(
        recipe_config,
        "_repository_revision",
        lambda _path: {"revision": "b" * 40, "dirty": False},
    )
    resolved = resolve_recipe_run(_recipe(tmp_path), site_path)
    assert resolved.code["worker"]["revision"] == "b" * 40
    assert resolved.experiment["vlm_smoke_evaluation"]["evaluator_revision"] == "b" * 40

    monkeypatch.setattr(
        recipe_config,
        "_repository_revision",
        lambda _path: {
            "revision": "b" * 40,
            "dirty": True,
            "working_tree_sha256": "c" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="source state changed"):
        recipe_config._assert_worker_source(
            str(tmp_path / "worker"), {"revision": "b" * 40, "dirty": False}
        )

    packaged_repository = tmp_path / "image" / "src" / "modelopt"
    packaged_repository.mkdir(parents=True)
    (packaged_repository.parents[1] / "modelopt_revision").write_text("d" * 40 + "\n")
    assert recipe_config._source_identity.repository_revision(packaged_repository) == {
        "revision": "d" * 40,
        "dirty": False,
    }


def test_source_identity_ignores_lfs_materialization_but_detects_code_edits(tmp_path):
    repository = tmp_path / "worker"
    repository.mkdir()
    (repository / ".gitattributes").write_text("report.html filter=lfs diff=lfs -text\n")
    (repository / "report.html").write_text(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{'a' * 64}\nsize 12\n"
    )
    source = repository / "worker.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PuzzleTron test",
            "-c",
            "user.email=puzzletron@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=repository,
        check=True,
    )
    (repository / "report.html").write_text("<html>materialized report</html>\n")
    assert recipe_config._repository_revision(repository)["dirty"] is False
    source.write_text("VALUE = 2\n")
    detected = recipe_config._repository_revision(repository)
    assert detected["dirty"] is True
    assert detected["working_tree_sha256"]


def test_synthetic_multinode_topology_is_validated_against_site_capacity(tmp_path):
    payload = recipe_template(
        model="qwen3.5-4b",
        workflow="vlm-pruning",
        mode="campaign",
        resource_profile="selected",
        run_root=str(tmp_path / "large-run"),
    )
    payload["data"] = {"path": str(tmp_path / "data"), "revision": "fixture-revision"}
    payload["advanced"] = {
        "execution": {
            "stages": {
                "post.candidate-evaluation.screening_kd": {
                    "instances": 2,
                    "parallel": {
                        "tp": 8,
                        "pp": 4,
                        "cp": 1,
                        "dp_shard": 2,
                        "dp_replicate": 2,
                        "ep": 2,
                    },
                }
            }
        }
    }
    recipe = _write_yaml(tmp_path / "large.yaml", payload)
    resolved = resolve_recipe_run(recipe, _site(tmp_path, max_nodes=64, mode="per_attempt"))
    kd = next(
        stage
        for stage in resolved.plan["stages"]
        if stage["stage_id"] == "post.candidate-evaluation.screening_kd"
    )
    assert (kd["total_gpus"], kd["nodes"]) == (256, 32)
    with pytest.raises(ValueError, match=r"provides at most 31 node.*screening_kd"):
        resolve_recipe_run(recipe, _site(tmp_path, max_nodes=31, mode="per_attempt"))


def test_every_checked_in_recipe_resolves_to_one_catalog_route(tmp_path):
    recipe_paths = sorted((REPOSITORY_ROOT / "examples/puzzletron/configs/recipes").glob("*.yaml"))
    resolved_routes = []
    for index, source in enumerate(recipe_paths):
        payload = yaml.safe_load(source.read_text())
        payload["run_root"] = str(tmp_path / f"run-{index}")
        payload["resource_profile"] = "selected"
        if "data" in payload:
            payload["data"] = {
                "path": str(tmp_path / f"dataset-{index}"),
                "revision": f"fixture-revision-{index}",
            }
        resolved = resolve_recipe_run(
            _write_yaml(tmp_path / f"recipe-{index}.yaml", payload),
            _site(tmp_path, max_nodes=64, mode="per_attempt"),
        )
        resolved_routes.append(resolved.route.route_id)
    assert Counter(resolved_routes) == Counter(route.route_id for route in recipe_config.ROUTES)


def test_cli_validate_explain_dry_run_and_inspect_share_one_contract(tmp_path, monkeypatch, capsys):
    recipe = _recipe(tmp_path)
    site = _site(tmp_path)
    assert public_cli.main(["validate", str(recipe), "--site", str(site)]) == 0
    assert "valid: my-puzzletron-run" in capsys.readouterr().out
    assert public_cli.main(["explain", str(recipe), "--site", str(site)]) == 0
    assert "route: qwen3.5-0.8b/vlm-pruning/smoke" in capsys.readouterr().out
    assert public_cli.main(["dry-run", str(recipe), "--site", str(site), "--color", "never"]) == 0
    assert "dry-run only; no jobs will be submitted" in capsys.readouterr().err

    resolved = resolve_recipe_run(recipe, site)
    materialize_resolved_bundle(resolved, activate=True)
    assert public_cli.main(["inspect", str(tmp_path / "run")]) == 0
    assert f"bundle: {resolved.bundle_id}" in capsys.readouterr().out


def test_launch_and_resume_delegate_the_same_sealed_inputs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(public_cli.orchestrate, "main", lambda argv: calls.append(argv) or 0)
    recipe = _recipe(tmp_path)
    site = _site(tmp_path)
    assert public_cli.main(["launch", str(recipe), "--site", str(site)]) == 0
    assert public_cli.main(["resume", str(tmp_path / "run")]) == 0

    def inputs(argv):
        return tuple(
            argv[argv.index(flag) + 1] for flag in ("--experiment", "--runner", "--execution")
        )

    bundle = bundle_for_run_root(tmp_path / "run")
    assert (
        inputs(calls[0])
        == inputs(calls[1])
        == (
            str(bundle / "experiment.runtime.yaml"),
            str(bundle / "runner.yaml"),
            str(bundle / "execution.yaml"),
        )
    )
    assert all(call[call.index("--stage") + 1] == "full" for call in calls)


def test_concurrent_activation_binds_exactly_one_bundle(tmp_path, monkeypatch):
    site = _site(tmp_path)
    first = resolve_recipe_run(_recipe(tmp_path), site)
    second = resolve_recipe_run(
        _recipe(tmp_path, advanced={"experiment": {"pruning.eval_samples": 3}}), site
    )
    materialize_resolved_bundle(first, activate=False)
    materialize_resolved_bundle(second, activate=False)
    barrier = Barrier(2)
    link = os.link

    def synchronized_link(source, destination):
        barrier.wait(timeout=5)
        link(source, destination)

    monkeypatch.setattr(recipe_config.os, "link", synchronized_link)

    def activate(resolved):
        try:
            materialize_resolved_bundle(resolved, activate=True)
        except ValueError:
            return "conflict", resolved.bundle_id
        return "bound", resolved.bundle_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(activate, (first, second)))
    assert Counter(status for status, _ in outcomes) == Counter({"bound": 1, "conflict": 1})
    bound_id = next(bundle_id for status, bundle_id in outcomes if status == "bound")
    active = json.loads((tmp_path / "run/orchestration/current_bundle.json").read_text())
    assert active["bundle_id"] == bound_id
