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


def test_runtime_modelopt_install_cannot_resolve_dependencies(project_root_path):
    setup_script = (project_root_path / "examples/puzzletron/ci/setup_env.sh").read_text()

    assert "python -m pip install --no-build-isolation --no-deps -e" in setup_script


def test_gpu_workflow_routes_the_pinned_image_to_the_existing_nox_session(project_root_path):
    workflow_path = project_root_path / ".github/workflows/puzzletron_gpu_tests.yml"
    workflow = yaml.safe_load(workflow_path.read_text())

    assert workflow["on"]["push"]["branches"] == ["pull-request/[0-9]+"]
    jobs = workflow["jobs"]
    assert "secrets" not in jobs["pr-gate"]
    assert jobs["gpu-puzzletron"]["container"]["image"] == (
        "${{ needs.resolve-image.outputs.image }}"
    )
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
