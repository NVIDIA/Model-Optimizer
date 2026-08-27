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
            "runtime",
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
            "runtime",
            package_version=_version_lookup(versions),
            source_verifier=lambda *_args: None,
            module_importer=lambda _name: object(),
            python_version=environment["python"],
            torch_cuda=environment["gpu_image"]["torch_cuda"],
        )


@pytest.mark.parametrize(
    (
        "profile",
        "expected_version_queries",
        "expected_sources",
        "expected_imports",
        "torch_cuda",
    ),
    [
        (
            "cpu",
            ["torch", "torchvision", "transformers", "lmms-eval", "nemo-automodel"],
            [("lmms-eval", "lmms_eval"), ("nemo-automodel", "nemo_automodel")],
            [],
            "not-installed",
        ),
        (
            "ci",
            [
                "torch",
                "torchvision",
                "transformers",
                "lmms-eval",
                "nemo-automodel",
                "aiperf",
                "nox",
            ],
            [("lmms-eval", "lmms_eval"), ("nemo-automodel", "nemo_automodel")],
            [],
            None,
        ),
        (
            "runtime",
            [
                "torch",
                "torchvision",
                "transformers",
                "lmms-eval",
                "nemo-automodel",
                "aiperf",
                "nox",
                "causal-conv1d",
                "flash-linear-attention",
                "nv-grouped-gemm",
                "mamba-ssm",
                "tilelang",
            ],
            [
                ("lmms-eval", "lmms_eval"),
                ("nemo-automodel", "nemo_automodel"),
                ("nv-grouped-gemm", "grouped_gemm"),
                ("vllm", "vllm"),
            ],
            ["causal_conv1d", "fla", "grouped_gemm", "mamba_ssm", "tilelang", "vllm"],
            None,
        ),
    ],
)
def test_verifier_applies_each_profile_contract(
    project_root_path,
    profile,
    expected_version_queries,
    expected_sources,
    expected_imports,
    torch_cuda,
):
    environment = _load_environment(project_root_path)
    sources = []
    imports = []
    version_queries = []
    torch_cuda = environment["gpu_image"]["torch_cuda"] if torch_cuda is None else torch_cuda

    verify_image_environment.verify_installed_environment(
        environment,
        profile,
        package_version=_version_lookup(_version_catalog(environment), version_queries),
        source_verifier=lambda package, source: sources.append((package, source)),
        module_importer=lambda name: imports.append(name),
        python_version=environment["python"],
        torch_cuda=torch_cuda,
    )

    assert version_queries == expected_version_queries
    assert sources == [
        (
            package,
            environment["runtime_image"][source_key]
            if source_key == "grouped_gemm"
            else environment[source_key],
        )
        for package, source_key in expected_sources
    ]
    assert imports == expected_imports


@pytest.mark.parametrize("profile", ["ci", "runtime"])
def test_gpu_profiles_reject_a_cuda_mismatch(project_root_path, profile):
    environment = _load_environment(project_root_path)

    with pytest.raises(RuntimeError, match="CUDA mismatch"):
        verify_image_environment.verify_installed_environment(
            environment,
            profile,
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
