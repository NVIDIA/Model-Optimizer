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

"""Tests for the repository-owned Puzzletron GPU image and workflow contract."""

import importlib.util
import json
import re

import pytest
import yaml

import noxfile


class _RecordingSession:
    def __init__(self):
        self.env = {}
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _load_image_resolver(project_root_path):
    resolver_path = project_root_path / "examples/puzzletron/ci/resolve_ci_image.py"
    spec = importlib.util.spec_from_file_location("puzzletron_ci_image", resolver_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_image_uses_the_recorded_immutable_base(project_root_path):
    environment_path = project_root_path / "examples/puzzletron/ci_environment.json"
    dockerfile_path = project_root_path / "examples/puzzletron/ci/Dockerfile"
    environment = json.loads(environment_path.read_text())
    dockerfile = dockerfile_path.read_text()

    base_image = environment["gpu_image"]["base_image"]
    assert re.fullmatch(r"nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}", base_image)
    assert f"FROM {base_image}" in dockerfile
    assert "ENV PUZZLETRON_REQUIREMENTS=/opt/puzzletron/requirements.txt" in dockerfile
    assert "RUN bash /opt/puzzletron/setup_env.sh --deps" in dockerfile
    assert 'python -m pip install "/opt/modelopt-dependencies[hf,puzzletron,dev-test]"' in (
        dockerfile
    )
    assert "bash /opt/puzzletron/setup_env.sh --verify" in dockerfile

    resolver = _load_image_resolver(project_root_path)
    assert resolver.__all__ == ["resolve_image_reference", "validate_repository_contract"]
    resolver.validate_repository_contract(project_root_path)


@pytest.mark.parametrize(
    "image",
    [
        "nvcr.io/nvidia/modelopt/puzzletron:latest",
        "docker.io/nvidia/modelopt/puzzletron@sha256:" + "a" * 64,
        "nvcr.io/nvidia/modelopt/puzzletron@sha256:" + "A" * 64,
        "nvcr.io/nvidia/modelopt/puzzletron@sha256:" + "a" * 63,
        "nvcr.io/nvidia//puzzletron@sha256:" + "a" * 64,
    ],
)
def test_gpu_image_resolver_rejects_mutable_or_malformed_references(project_root_path, image):
    resolver = _load_image_resolver(project_root_path)
    with pytest.raises(ValueError, match="immutable nvcr.io digest"):
        resolver.resolve_image_reference(image)


def test_gpu_image_resolver_returns_the_digest_cache_key(project_root_path):
    resolver = _load_image_resolver(project_root_path)
    digest = "a" * 64
    image = f"nvcr.io/nvidia/modelopt/puzzletron@sha256:{digest}"

    assert resolver.resolve_image_reference(image) == (image, digest)


def test_gpu_image_resolver_reports_an_unconfigured_repository_variable(
    project_root_path, monkeypatch, capsys
):
    resolver = _load_image_resolver(project_root_path)
    monkeypatch.chdir(project_root_path)
    monkeypatch.delenv("PUZZLETRON_GPU_CI_IMAGE", raising=False)

    assert resolver.main() == 0
    captured = capsys.readouterr()
    assert captured.out == "configured=false\n"
    assert "::warning::PUZZLETRON_GPU_CI_IMAGE is not configured" in captured.err


def test_gpu_image_resolver_reports_a_configured_immutable_image(
    project_root_path, monkeypatch, capsys
):
    resolver = _load_image_resolver(project_root_path)
    monkeypatch.chdir(project_root_path)
    digest = "a" * 64
    image = f"nvcr.io/nvidia/modelopt/puzzletron@sha256:{digest}"
    monkeypatch.setenv("PUZZLETRON_GPU_CI_IMAGE", image)

    assert resolver.main() == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "configured=true",
        f"image={image}",
        f"cache_key={digest}",
    ]
    assert captured.err == ""


def test_gpu_image_resolver_rejects_a_malformed_configured_image(
    project_root_path, monkeypatch, capsys
):
    resolver = _load_image_resolver(project_root_path)
    monkeypatch.chdir(project_root_path)
    monkeypatch.setenv("PUZZLETRON_GPU_CI_IMAGE", "nvcr.io/nvidia/puzzletron:latest")

    assert resolver.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "::error::PUZZLETRON_GPU_CI_IMAGE must be an immutable nvcr.io digest\n"
    )


def test_runtime_modelopt_install_cannot_resolve_dependencies(project_root_path):
    setup_script = (project_root_path / "examples/puzzletron/ci/setup_env.sh").read_text()

    assert "python -m pip install --no-build-isolation --no-deps -e" in setup_script


def test_gpu_workflow_routes_the_pinned_image_to_the_existing_nox_session(project_root_path):
    workflow_path = project_root_path / ".github/workflows/puzzletron_gpu_tests.yml"
    workflow_text = workflow_path.read_text()
    workflow = yaml.safe_load(workflow_text)

    assert workflow["on"]["push"]["branches"] == ["pull-request/[0-9]+"]
    jobs = workflow["jobs"]
    assert "secrets" not in jobs["pr-gate"]
    assert jobs["resolve-image"]["outputs"]["configured"] == (
        "${{ steps.image.outputs.configured }}"
    )
    assert jobs["gpu-puzzletron"]["if"] == "needs.resolve-image.outputs.configured == 'true'"
    assert jobs["gpu-puzzletron"]["container"]["image"] == (
        "${{ needs.resolve-image.outputs.image }}"
    )
    assert "credentials" not in jobs["gpu-puzzletron"]["container"]
    assert "NGC_API_KEY" not in workflow_text
    assert jobs["gpu-puzzletron"]["steps"][-1]["run"] == "nox -s gpu_puzzletron"

    assert jobs["resolve-image"]["permissions"]["contents"] == "read"
    resolve_steps = jobs["resolve-image"]["steps"]
    assert resolve_steps[0]["uses"] == "actions/checkout@v6"
    assert resolve_steps[0]["with"]["persist-credentials"] is False
    resolve_step = resolve_steps[1]
    assert resolve_step["run"] == (
        'python examples/puzzletron/ci/resolve_ci_image.py >> "${GITHUB_OUTPUT}"'
    )
    assert "PUZZLETRON_GPU_CI_IMAGE" in resolve_step["env"]
    assert jobs["gpu-puzzletron"]["timeout-minutes"] == 50

    required_steps = jobs["gpu-puzzletron-required-check"]["steps"]
    unconfigured_step = next(
        step
        for step in required_steps
        if step["name"] == "Report unconfigured Puzzletron GPU image"
    )
    assert " ".join(unconfigured_step["if"].split()) == (
        "${{ needs.pr-gate.outputs.run_tests == 'true' && "
        "needs.resolve-image.result == 'success' && "
        "needs.resolve-image.outputs.configured == 'false' }}"
    )
    failure_step = next(
        step
        for step in required_steps
        if step["name"] == "Required Puzzletron GPU tests did not succeed"
    )
    assert " ".join(failure_step["if"].split()) == (
        "${{ needs.pr-gate.result != 'success' || "
        "(needs.pr-gate.outputs.run_tests == 'true' && "
        "(needs.resolve-image.result != 'success' || "
        "(needs.resolve-image.outputs.configured != 'false' && "
        "needs.resolve-image.outputs.configured != 'true') || "
        "(needs.resolve-image.outputs.configured == 'true' && "
        "needs.gpu-puzzletron.result != 'success'))) }}"
    )


def test_gpu_contract_routes_all_ci_recipe_changes_through_cpu_tests(project_root_path):
    unit_workflow = yaml.safe_load(
        (project_root_path / ".github/workflows/unit_tests.yml").read_text()
    )
    steps = unit_workflow["jobs"]["check-file-changes"]["steps"]
    puzzletron_step = next(step for step in steps if step.get("id") == "puzzletron_changed")
    puzzletron_paths = puzzletron_step["with"]["files"]

    assert "examples/puzzletron/ci/**" in puzzletron_paths.splitlines()


def test_gpu_nox_session_uses_the_checked_out_environment_contract():
    session = _RecordingSession()

    noxfile.gpu_puzzletron.func(session)

    assert session.env["PUZZLETRON_CI_ENVIRONMENT"] == str(
        noxfile.PUZZLETRON_V2_CI_ENVIRONMENT_PATH
    )
    assert session.calls[0] == (
        ("bash", "examples/puzzletron/ci/setup_env.sh", "--modelopt"),
        {"external": True},
    )
    assert [args[:2] for args, _kwargs in session.calls] == [
        ("bash", "examples/puzzletron/ci/setup_env.sh"),
        ("python", "-c"),
        ("python", "-m"),
    ]
