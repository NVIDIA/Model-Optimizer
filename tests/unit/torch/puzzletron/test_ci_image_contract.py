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

"""Tests for the initial repository-owned Puzzletron image recipe."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import yaml


def test_image_recipe_records_pinned_environment(project_root_path):
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
    assert 'export TORCH_CUDA_ARCH_LIST="${recorded_cuda_architectures}"' in dockerfile
    assert 'export TORCH_CUDA_ARCH_LIST="${grouped_gemm_cuda_architectures}"' in dockerfile
    assert "VLLM_USE_PRECOMPILED" not in dockerfile
    assert "ENV TORCH_CUDA_ARCH_LIST=" not in dockerfile
    assert "ENV FORCE_CUDA=" not in dockerfile

    revision_arg = dockerfile.index("ARG MODELOPT_REVISION")
    assert revision_arg > dockerfile.index(grouped_gemm_ref)
    assert revision_arg < dockerfile.index("COPY modelopt /opt/puzzletron/src/modelopt/modelopt")
    assert "COPY examples/__init__.py /opt/puzzletron/src/modelopt/examples/__init__.py" in (
        dockerfile
    )

    mamba_source = environment["runtime_image"]["mamba_ssm"]
    patch_bytes = (puzzletron_root / "patches" / mamba_source["compatibility_patch"]).read_bytes()
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

    subprocess.run(
        [
            sys.executable,
            str(project_root_path / "examples/puzzletron/ci/verify_image_environment.py"),
            "--environment",
            str(project_root_path / "examples/puzzletron/ci_environment.json"),
            "--manifest-only",
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(image_root), str(tmp_path / "site-packages")]),
        },
    )


def test_image_excludes_checked_in_reports(project_root_path):
    dockerignore = (project_root_path / ".dockerignore").read_text().splitlines()

    assert "examples/puzzletron/reports" in dockerignore


def test_cpu_contract_lane_watches_image_recipe_inputs(project_root_path):
    workflow = yaml.safe_load((project_root_path / ".github/workflows/unit_tests.yml").read_text())

    # PyYAML applies YAML 1.1 boolean resolution to GitHub's unquoted `on` key.
    push_paths = workflow[True]["push"]["paths"]
    changed_files_step = next(
        step
        for step in workflow["jobs"]["check-file-changes"]["steps"]
        if step.get("id") == "puzzletron_changed"
    )
    pull_request_paths = changed_files_step["with"]["files"].splitlines()

    for image_input in (".dockerignore", "examples/__init__.py"):
        assert image_input in push_paths
        assert image_input in pull_request_paths
    assert "examples/puzzletron/**" in push_paths
    assert "examples/puzzletron/Dockerfile" in pull_request_paths
