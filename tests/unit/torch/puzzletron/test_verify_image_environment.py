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

"""Behavioral tests for the Puzzletron image environment verifier."""

import copy
import json
from importlib import metadata
from types import SimpleNamespace

import pytest

from examples.puzzletron.ci import verify_image_environment


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "https://github.com/example/vllm.git", "must use"),
        ("commit", "feature/add_anymodel_to_vllm", "full Git revision"),
    ],
)
def test_manifest_rejects_mutable_or_unapproved_vllm_source(
    project_root_path, field, value, message
):
    environment = copy.deepcopy(_load_environment(project_root_path))
    environment["vllm"][field] = value

    with pytest.raises(ValueError, match=message):
        verify_image_environment.validate_environment_contract(environment)


def test_runtime_verifier_reports_a_package_version_mismatch(project_root_path):
    environment = _load_environment(project_root_path)
    versions = _version_catalog(environment)
    versions["flash-linear-attention"] = "0.5.2"

    with pytest.raises(RuntimeError, match="flash-linear-attention"):
        verify_image_environment.verify_installed_environment(
            environment,
            package_version=_version_lookup(versions),
            source_verifier=lambda *_args: None,
            module_importer=lambda _name: object(),
            python_version=environment["python"],
            torch_cuda=environment["gpu_image"]["torch_cuda"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "https://github.com/example/mamba.git", "must use"),
        ("commit", "v2.3.2.post1", "full Git revision"),
        ("compatibility_patch", "../unreviewed.patch", "safe patch filename"),
        ("compatibility_patch_sha256", "not-a-digest", "declare a SHA-256"),
    ],
)
def test_manifest_rejects_unpinned_mamba_source_or_patch(project_root_path, field, value, message):
    environment = _load_environment(project_root_path)
    environment["runtime_image"]["mamba_ssm"][field] = value

    with pytest.raises(ValueError, match=message):
        verify_image_environment.validate_environment_contract(environment)


def test_runtime_verifier_reports_a_mamba_version_mismatch(project_root_path):
    environment = _load_environment(project_root_path)
    versions = _version_catalog(environment)
    versions["mamba-ssm"] = "2.3.1"

    with pytest.raises(RuntimeError, match="mamba-ssm"):
        verify_image_environment.verify_installed_environment(
            environment,
            package_version=_version_lookup(versions),
            source_verifier=lambda *_args: None,
            module_importer=lambda _name: object(),
            python_version=environment["python"],
            torch_cuda=environment["gpu_image"]["torch_cuda"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("decord_wheel_url", "https://example.com/decord.whl", "x86_64 decord wheel"),
        ("decord_wheel_sha256", "not-a-digest", "decord wheel checksum"),
        ("nltk_data_commit", "gh-pages", "NLTK data revision"),
        ("nltk_resource_sha256", {"punkt": "0" * 64}, "checksum every NLTK resource"),
    ],
)
def test_manifest_rejects_unpinned_worker_assets(project_root_path, field, value, message):
    environment = _load_environment(project_root_path)
    environment["gpu_image"][field] = value

    with pytest.raises(ValueError, match=message):
        verify_image_environment.validate_environment_contract(environment)


def test_verifier_applies_worker_contract(project_root_path, tmp_path):
    environment = _load_environment(project_root_path)
    sources = []
    imports = []
    resources = []
    version_queries = []
    lmms_root = tmp_path / "lmms_eval"
    for task_config in environment["lmms_eval"]["task_configs"]:
        path = lmms_root / task_config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def import_worker_module(name):
        imports.append(name)
        if name == "lmms_eval":
            return SimpleNamespace(__path__=[str(lmms_root)])
        if name == "nltk":
            return SimpleNamespace(data=SimpleNamespace(find=resources.append))
        return object()

    verify_image_environment.verify_installed_environment(
        environment,
        package_version=_version_lookup(_version_catalog(environment), version_queries),
        source_verifier=lambda package, source: sources.append((package, source)),
        module_importer=import_worker_module,
        python_version=environment["python"],
        torch_cuda=environment["gpu_image"]["torch_cuda"],
    )

    assert version_queries == [
        "torch",
        "torchvision",
        "transformers",
        "lmms-eval",
        "nemo-automodel",
        "aiperf",
        "decord",
        "langdetect",
        "nltk",
        "nox",
        "causal-conv1d",
        "flash-linear-attention",
        "nv-grouped-gemm",
        "mamba-ssm",
        "tilelang",
    ]
    assert sources == [
        ("lmms-eval", environment["lmms_eval"]),
        ("nemo-automodel", environment["nemo_automodel"]),
        ("nv-grouped-gemm", environment["runtime_image"]["grouped_gemm"]),
        ("vllm", environment["vllm"]),
    ]
    assert imports == [
        "aiperf",
        "causal_conv1d",
        "decord",
        "fla",
        "grouped_gemm",
        "langdetect",
        "lmms_eval",
        "mamba_ssm",
        "modelopt",
        "nemo_automodel",
        "nltk",
        "puzzletron_orchestrator",
        "puzzletron_setup",
        "tilelang",
        "torch",
        "transformers",
        "vllm",
    ]
    assert resources == ["tokenizers/punkt", "tokenizers/punkt_tab"]


def test_verifier_rejects_a_cuda_mismatch(project_root_path):
    environment = _load_environment(project_root_path)

    with pytest.raises(RuntimeError, match="CUDA mismatch"):
        verify_image_environment.verify_installed_environment(
            environment,
            package_version=_version_lookup(_version_catalog(environment)),
            source_verifier=lambda *_args: None,
            module_importer=lambda _name: object(),
            python_version=environment["python"],
            torch_cuda="0.0",
        )


def _load_environment(project_root_path):
    path = project_root_path / "examples/puzzletron/ci_environment.json"
    return json.loads(path.read_text())


def _version_catalog(environment):
    return {
        "torch": environment["torch"],
        "torchvision": environment["torchvision"],
        "transformers": environment["transformers"],
        "lmms-eval": environment["lmms_eval"]["base_version"],
        "nemo-automodel": environment["nemo_automodel"]["base_version"],
        "aiperf": environment["gpu_image"]["aiperf"],
        "decord": environment["gpu_image"]["decord"],
        "langdetect": environment["gpu_image"]["langdetect"],
        "nltk": environment["gpu_image"]["nltk"],
        "nox": environment["gpu_image"]["nox"],
        "causal-conv1d": environment["runtime_image"]["causal_conv1d"],
        "flash-linear-attention": environment["runtime_image"]["flash_linear_attention"],
        "nv-grouped-gemm": environment["runtime_image"]["grouped_gemm"]["base_version"],
        "mamba-ssm": environment["runtime_image"]["mamba_ssm"]["base_version"],
        "tilelang": environment["runtime_image"]["tilelang"],
    }


def _version_lookup(versions, queries=None):
    def lookup(package):
        if queries is not None:
            queries.append(package)
        if package not in versions:
            raise metadata.PackageNotFoundError(package)
        return versions[package]

    return lookup
