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

"""Tests for the repository-owned Puzzletron image and workflow contract."""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml

import noxfile


def test_canonical_image_is_the_only_install_recipe(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()

    assert not (puzzletron_root / "ci/Dockerfile").exists()
    assert not (puzzletron_root / "ci/setup_env.sh").exists()
    base_image = environment["gpu_image"]["base_image"]
    assert re.fullmatch(r"nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}", base_image)
    assert f"FROM {base_image}" in dockerfile
    assert '"causal-conv1d==${causal_conv1d_version}"' in dockerfile
    assert 'checkout --detach "${mamba_ssm_revision}"' in dockerfile
    assert "sha256sum --check --strict" in dockerfile
    assert 'apply "/opt/puzzletron/patches/${mamba_ssm_patch}"' in dockerfile
    grouped_gemm_ref = (
        '"${grouped_gemm_distribution} @ git+${grouped_gemm_repository}@${grouped_gemm_revision}"'
    )
    assert grouped_gemm_ref in dockerfile
    assert '"flash-linear-attention[cuda]==${linear_attention_version}"' in dockerfile
    assert "VLLM_USE_PRECOMPILED" not in dockerfile
    assert 'export TORCH_CUDA_ARCH_LIST="${recorded_cuda_architectures}"' in dockerfile
    assert 'export TORCH_CUDA_ARCH_LIST="${grouped_gemm_cuda_architectures}"' in dockerfile
    assert "ENV TORCH_CUDA_ARCH_LIST=" not in dockerfile
    assert "ENV FORCE_CUDA=" not in dockerfile
    assert "ENV MODEL_OPT_ROOT=" not in dockerfile
    assert "ENV PUZZLETRON_VLLM_ANYMODEL=1" in dockerfile
    revision_arg = dockerfile.index("ARG MODELOPT_REVISION")
    assert revision_arg > dockerfile.index(grouped_gemm_ref)
    assert revision_arg < dockerfile.index("COPY modelopt /opt/puzzletron/src/modelopt/modelopt")
    examples_package_copy = (
        "COPY examples/__init__.py /opt/puzzletron/src/modelopt/examples/__init__.py"
    )
    assert examples_package_copy in dockerfile
    assert dockerfile.index(examples_package_copy) < dockerfile.index(
        "RUN python -m pip install --no-build-isolation --no-deps -e"
    )

    resolver = _load_image_resolver(project_root_path)
    resolver.validate_repository_contract(project_root_path)

    mamba_source = environment["runtime_image"]["mamba_ssm"]
    patch_path = puzzletron_root / "patches" / mamba_source["compatibility_patch"]
    patch_bytes = patch_path.read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest() == mamba_source["compatibility_patch_sha256"]
    changed_lines = [
        line
        for line in patch_bytes.decode().splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- "))
    ]
    assert changed_lines == [
        '-    "tilelang==0.1.8",',
        '+    "tilelang==0.1.9",',
        '-        "tilelang==0.1.8",',
        '+        "tilelang==0.1.9",',
    ]


@pytest.mark.parametrize(
    "required_line",
    [
        "ENV PYTHONPATH=/opt/puzzletron/src/modelopt\n",
        '"causal-conv1d==${causal_conv1d_version}"',
        'apply "/opt/puzzletron/patches/${mamba_ssm_patch}"',
        '"${grouped_gemm_distribution} @ git+${grouped_gemm_repository}@${grouped_gemm_revision}"',
    ],
)
def test_repository_contract_rejects_recipe_drift(project_root_path, tmp_path, required_line):
    repository_root = tmp_path / "repository"
    puzzletron_root = repository_root / "examples/puzzletron"
    puzzletron_root.mkdir(parents=True)
    shutil.copy(
        project_root_path / "examples/puzzletron/ci_environment.json",
        puzzletron_root / "ci_environment.json",
    )
    dockerfile_path = puzzletron_root / "Dockerfile"
    shutil.copy(project_root_path / "examples/puzzletron/Dockerfile", dockerfile_path)
    dockerfile = dockerfile_path.read_text()
    assert required_line in dockerfile
    dockerfile_path.write_text(dockerfile.replace(required_line, ""))

    resolver = _load_image_resolver(project_root_path)
    with pytest.raises(ValueError, match="missing recorded contract lines"):
        resolver.validate_repository_contract(repository_root)


def test_standalone_verifier_prefers_the_baked_examples_package(project_root_path, tmp_path):
    image_root = tmp_path / "image-root"
    baked_examples = image_root / "examples"
    baked_puzzletron = baked_examples / "puzzletron"
    baked_puzzletron.mkdir(parents=True)
    shutil.copy(project_root_path / "examples/__init__.py", baked_examples / "__init__.py")
    shutil.copy(
        project_root_path / "examples/puzzletron/ci_environment.py",
        baked_puzzletron / "ci_environment.py",
    )

    shadow_examples = tmp_path / "site-packages/examples"
    shadow_examples.mkdir(parents=True)
    (shadow_examples / "__init__.py").write_text(
        "raise RuntimeError('third-party examples package was imported')\n"
    )

    verifier = project_root_path / "examples/puzzletron/ci/verify_image_environment.py"
    environment = project_root_path / "examples/puzzletron/ci_environment.json"
    subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--environment",
            str(environment),
            "--profile",
            "runtime",
            "--manifest-only",
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(image_root), str(tmp_path / "site-packages")]),
        },
    )


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
def test_image_resolver_rejects_mutable_or_malformed_references(project_root_path, image):
    resolver = _load_image_resolver(project_root_path)
    with pytest.raises(ValueError, match="immutable nvcr.io digest"):
        resolver.resolve_image_reference(image)


def test_image_resolver_cli_emits_the_image_and_digest_cache_key(
    project_root_path, monkeypatch, capsys
):
    resolver = _load_image_resolver(project_root_path)
    digest = "a" * 64
    image = f"nvcr.io/nvidia/modelopt/puzzletron@sha256:{digest}"
    monkeypatch.chdir(project_root_path)
    monkeypatch.setenv("PUZZLETRON_GPU_CI_IMAGE", image)

    assert resolver.main() == 0
    assert capsys.readouterr().out.splitlines() == [f"image={image}", f"cache_key={digest}"]


def test_gpu_nox_session_overlays_before_verification_and_lifecycle():
    events = []

    class RecordingSession:
        def run(self, *args):
            events.append(args)

    noxfile.gpu_puzzletron.func(RecordingSession())

    install = next(event for event in events if event[:4] == ("python", "-m", "pip", "install"))
    assert "--no-build-isolation" in install
    assert "--no-deps" in install
    assert install[-2:] == ("-e", ".[hf,puzzletron,dev-test]")

    verify = (
        "python",
        "-m",
        "examples.puzzletron.ci.verify_image_environment",
        "--environment",
        "examples/puzzletron/ci_environment.json",
        "--profile",
        "ci",
    )
    lifecycle = next(event for event in events if event[:3] == ("python", "-m", "pytest"))
    assert events.index(install) < events.index(verify) < events.index(lifecycle)
    lifecycle_test = (
        "tests/gpu/torch/puzzletron/test_puzzletron.py::"
        "test_tiny_qwen_campaign_uses_current_public_route"
    )
    assert lifecycle_test in lifecycle


def test_image_workflow_builds_once_and_exercises_lifecycle_ci(project_root_path):
    workflow_path = project_root_path / ".github/workflows/puzzletron_runtime_image.yml"
    workflow = yaml.safe_load(workflow_path.read_text())

    assert workflow["on"]["push"]["branches"] == ["pull-request/[0-9]+"]
    assert "schedule" not in workflow["on"]
    jobs = workflow["jobs"]
    watched_files = jobs["pr-gate"]["with"]["files"].splitlines()
    assert ".dockerignore" in watched_files
    for image_input in (
        "LICENSE_HEADER",
        "README.md",
        "examples/__init__.py",
        "examples/puzzletron/**",
        "modelopt/**",
        "modelopt_recipes/**",
        "noxfile.py",
        "puzzletron_orchestrator/**",
        "puzzletron_setup/**",
        "pyproject.toml",
        "tests/conftest.py",
        "tests/_test_utils/torch/puzzletron/**",
        "tests/gpu_vllm/torch/puzzletron/test_calc_runtime_stats.py",
    ):
        assert image_input in watched_files

    build_job = jobs["build-runtime-image"]
    assert build_job["timeout-minutes"] == 180
    assert set(build_job["needs"]) == {"pr-gate", "dependency-metadata-preflight"}
    checkout = next(
        step for step in build_job["steps"] if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["persist-credentials"] is False
    build_command = next(
        step["run"] for step in build_job["steps"] if "docker build" in step.get("run", "")
    )
    assert build_command.count("docker build") == 1
    assert "--file examples/puzzletron/Dockerfile" in build_command
    assert "python /opt/puzzletron/verify_image_environment.py" in build_command
    assert '"${GITHUB_WORKSPACE}:/qualification/source:ro"' in build_command
    assert "--workdir /opt/puzzletron/src/modelopt" in build_command
    assert "--workdir /qualification/source" in build_command
    assert "python -P -m pytest" in build_command
    assert "PYTHONPATH=/qualification/source:/qualification/source/tests" in build_command
    assert "Path(modelopt.__file__).resolve().is_relative_to(root)" in build_command
    assert "/qualification/source/tests/unit/torch/puzzletron" in build_command
    assert "tests/gpu_vllm/torch/puzzletron/test_calc_runtime_stats.py" in build_command
    assert "--gpus device=0" in build_command
    assert "nox -s gpu_puzzletron" in build_command
    metadata_preflight = jobs["dependency-metadata-preflight"]
    assert metadata_preflight["runs-on"] == "ubuntu-latest"
    assert "gpu" not in metadata_preflight["runs-on"]
    metadata_command = next(
        step["run"]
        for step in metadata_preflight["steps"]
        if "preflight_dependency_metadata" in step.get("run", "")
    )
    assert "ci_environment.json" in metadata_command
    _assert_required_check(
        jobs["runtime-image-required-check"],
        required_dependencies={
            "pr-gate",
            "dependency-metadata-preflight",
            "build-runtime-image",
        },
        required_results={
            "pr-gate",
            "dependency-metadata-preflight",
            "build-runtime-image",
        },
    )

    assert workflow["permissions"] == {"contents": "read"}
    for job in jobs.values():
        for permission in job.get("permissions", workflow["permissions"]).values():
            assert permission != "write"
        assert "secrets." not in json.dumps(job)
        for step in job.get("steps", []):
            action = step.get("uses", "")
            assert "docker/login-action" not in action
            assert "docker/build-push-action" not in action
            assert step.get("with", {}).get("push") is not True
            command = step.get("run", "")
            for publication_operation in (
                "docker push",
                "docker image push",
                "buildx build --push",
                "oras push",
                "skopeo copy",
            ):
                assert publication_operation not in command


def test_gpu_workflow_consumes_one_immutable_image(project_root_path):
    workflow_path = project_root_path / ".github/workflows/puzzletron_gpu_tests.yml"
    workflow = yaml.safe_load(workflow_path.read_text())

    assert workflow["on"]["push"]["branches"] == ["pull-request/[0-9]+"]
    jobs = workflow["jobs"]
    assert "secrets" not in jobs["pr-gate"]
    assert jobs["gpu-puzzletron"]["container"]["image"] == (
        "${{ needs.resolve-image.outputs.image }}"
    )
    container_env = jobs["gpu-puzzletron"]["container"]["env"]
    assert container_env["PUZZLETRON_ROOT"] == "${{ github.workspace }}"
    assert container_env["PYTHONPATH"] == "${{ github.workspace }}"
    lifecycle = next(
        step
        for step in jobs["gpu-puzzletron"]["steps"]
        if "nox -s gpu_puzzletron" in step.get("run", "")
    )
    assert lifecycle["run"] == "nox -s gpu_puzzletron"

    resolve_step = next(
        step for step in jobs["resolve-image"]["steps"] if step.get("id") == "image"
    )
    assert resolve_step["run"] == (
        'python examples/puzzletron/ci/resolve_ci_image.py >> "${GITHUB_OUTPUT}"'
    )
    assert "PUZZLETRON_GPU_CI_IMAGE" in resolve_step["env"]
    _assert_required_check(
        jobs["gpu-puzzletron-required-check"],
        required_dependencies={"pr-gate", "resolve-image", "gpu-puzzletron"},
        required_results={"pr-gate", "resolve-image", "gpu-puzzletron"},
    )


def test_documentation_has_no_parallel_manual_install_path(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    readme = (puzzletron_root / "README.md").read_text()
    image_readme = (puzzletron_root / "ci/README.md").read_text()

    assert "### Manual environment construction" not in readme
    assert "setup_env.sh" not in readme
    assert "--file examples/puzzletron/Dockerfile" in readme
    assert "verify_image_environment.py" in readme
    assert "--file examples/puzzletron/Dockerfile" in image_readme
    assert "verify_image_environment.py" in image_readme

    puzzletron_docs = list((puzzletron_root / "docs").glob("**/*.md"))
    assert puzzletron_docs
    for documentation_path in puzzletron_docs:
        assert "pip install -r examples/puzzletron/requirements.txt" not in (
            documentation_path.read_text()
        )


def test_image_excludes_checked_in_reports(project_root_path):
    dockerignore = (project_root_path / ".dockerignore").read_text().splitlines()

    assert "examples/puzzletron/reports" in dockerignore


def test_cpu_contract_lane_watches_all_image_contract_inputs(project_root_path):
    workflow_path = project_root_path / ".github/workflows/unit_tests.yml"
    workflow = yaml.safe_load(workflow_path.read_text())

    # PyYAML applies YAML 1.1 boolean resolution to GitHub's unquoted `on` key.
    push_paths = workflow[True]["push"]["paths"]
    changed_files_step = next(
        step
        for step in workflow["jobs"]["check-file-changes"]["steps"]
        if step.get("id") == "puzzletron_changed"
    )
    pull_request_paths = changed_files_step["with"]["files"].splitlines()

    for image_contract_input in (
        ".dockerignore",
        ".github/workflows/puzzletron_gpu_tests.yml",
        ".github/workflows/puzzletron_runtime_image.yml",
        "examples/__init__.py",
    ):
        assert image_contract_input in push_paths
    assert "examples/puzzletron/**" in push_paths
    for image_contract_input in (
        ".dockerignore",
        ".github/workflows/puzzletron_gpu_tests.yml",
        ".github/workflows/puzzletron_runtime_image.yml",
        "examples/__init__.py",
        "examples/puzzletron/Dockerfile",
    ):
        assert image_contract_input in pull_request_paths


def _assert_required_check(job, *, required_dependencies, required_results):
    assert set(job["needs"]) == required_dependencies
    assert "always()" in job["if"]
    assert "startsWith(github.ref, 'refs/heads/pull-request/')" in job["if"]
    failure_step = next(step for step in job["steps"] if step.get("run") == "exit 1")
    for result in required_results:
        assert f"needs.{result}.result != 'success'" in failure_step["if"]


def _load_image_resolver(project_root_path):
    resolver_path = project_root_path / "examples/puzzletron/ci/resolve_ci_image.py"
    spec = importlib.util.spec_from_file_location("puzzletron_ci_image", resolver_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
