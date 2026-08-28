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

"""Tests for the canonical repository-owned Puzzletron worker image."""

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
    requirements = (puzzletron_root / "requirements.txt").read_text()

    assert not (puzzletron_root / "ci/Dockerfile").exists()
    assert not (puzzletron_root / "ci/setup_env.sh").exists()
    base_image = environment["gpu_image"]["base_image"]
    assert re.fullmatch(r"nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}", base_image)
    assert f"FROM {base_image}" in dockerfile
    assert "ARG TARGETPLATFORM" in dockerfile
    assert 'test "${TARGETPLATFORM}" = "linux/amd64"' in dockerfile
    assert (
        "COPY examples/puzzletron/requirements.txt /opt/puzzletron/requirements.txt" in dockerfile
    )
    assert "COPY examples/puzzletron/ci_environment.json /opt/puzzletron/ci_environment.json" in (
        dockerfile
    )
    assert 'python "${PUZZLETRON_VERIFY_SCRIPT}"' in dockerfile

    video_decoder = environment["gpu_image"]["video_decoder"]
    assert (
        f'{video_decoder["distribution"]}=={video_decoder["version"]}; platform_system == "Linux"'
    ) in requirements
    lmms_source = environment["lmms_eval"]
    assert (
        f"-e git+{lmms_source['repository']}@{lmms_source['commit']}#egg=lmms-eval" in requirements
    )
    assert '"langdetect==${langdetect_version}"' in dockerfile
    assert '"nltk==${nltk_version}"' in dockerfile
    assert "nltk_data/${nltk_data_commit}/packages/tokenizers/${nltk_resource}.zip" in dockerfile
    assert (
        'echo "${nltk_resource_sha256}  ${nltk_archive}" | sha256sum --check --strict' in dockerfile
    )


def test_mamba_compatibility_patch_matches_the_manifest(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())

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


def test_worker_documentation_has_one_install_recipe(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment_guide = (puzzletron_root / "docs/environment_setup.md").read_text()
    worker_section = environment_guide.split("## Worker environment", maxsplit=1)[1]

    assert "../Dockerfile" in worker_section
    assert "../ci/README.md" in worker_section
    for manual_install_command in (
        "apt-get install",
        "git clone",
        "python -m pip install",
    ):
        assert manual_install_command not in worker_section

    checkpoint_guide = (puzzletron_root / "docs/checkpoint_evaluation.md").read_text()
    assert "pip install -r examples/puzzletron/requirements.txt" not in checkpoint_guide


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
