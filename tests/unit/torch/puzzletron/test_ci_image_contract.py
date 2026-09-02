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

"""Tests for the repository-owned Puzzletron worker image."""

import hashlib
import json
import re

import yaml


def test_image_recipe_records_pinned_environment(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()

    base_image = environment["gpu_image"]["base_image"]
    assert re.fullmatch(r"nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}", base_image)
    assert f"FROM {base_image}" in dockerfile
    assert "ARG TARGETPLATFORM" in dockerfile
    assert 'test "${TARGETPLATFORM}" = "linux/amd64"' in dockerfile
    assert "> /opt/puzzletron/modelopt_revision" in dockerfile
    assert (
        "COPY examples/puzzletron/ci_environment.json /opt/puzzletron/ci_environment.json"
        in dockerfile
    )
    assert "for module in" in dockerfile
    assert "nltk.data.find" in dockerfile
    assert ".lmms_eval.required_paths[]" in dockerfile

    vcs_sources = [
        environment["lmms_eval"],
        environment["nemo_automodel"],
        environment["vllm"],
        environment["runtime_image"]["grouped_gemm"],
        environment["runtime_image"]["mamba_ssm"],
    ]
    assert all(re.fullmatch(r"[0-9a-f]{40}", source["commit"]) for source in vcs_sources)

    nltk_resources = environment["gpu_image"]["nltk_resources"]
    nltk_checksums = environment["gpu_image"]["nltk_resource_sha256"]
    assert set(nltk_checksums) == set(nltk_resources)
    assert all(re.fullmatch(r"[0-9a-f]{64}", checksum) for checksum in nltk_checksums.values())

    assert "nltk_data/$(pin gpu_image.nltk_data_commit)/packages/tokenizers" in dockerfile
    assert (
        'echo "${nltk_resource_sha256}  ${nltk_archive}" | sha256sum --check --strict' in dockerfile
    )


def test_mamba_compatibility_patch_is_limited_to_the_tilelang_pin(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()

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
    assert '"$(pin runtime_image.mamba_ssm.repository)" /tmp/mamba-ssm' in dockerfile
    assert 'git -C /tmp/mamba-ssm checkout --detach "$(pin runtime_image.mamba_ssm.commit)"' in (
        dockerfile
    )
    assert 'test "$(git -C /tmp/mamba-ssm rev-parse HEAD)" = \\' in dockerfile


def test_lmms_eval_compatibility_patch_only_reconciles_wandb(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()

    lmms_source = environment["lmms_eval"]
    patch = puzzletron_root / "patches" / lmms_source["compatibility_patch"]
    patch_bytes = patch.read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest() == lmms_source["compatibility_patch_sha256"]
    changed_lines = [
        line
        for line in patch_bytes.decode().splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- "))
    ]
    assert changed_lines == [
        '-    "wandb==0.25.0",',
        '+    "wandb>=0.28.0,<0.30",',
    ]
    assert lmms_source["compatibility_patch_files"] == ["pyproject.toml"]
    assert '"$(pin lmms_eval.repository)" "${LMMS_EVAL_ROOT}"' in dockerfile
    assert 'git -C "${LMMS_EVAL_ROOT}" checkout --detach "$(pin lmms_eval.commit)"' in dockerfile
    assert 'git -C "${LMMS_EVAL_ROOT}" apply --check' in dockerfile
    assert 'python -m pip install -e "${LMMS_EVAL_ROOT}[qwen]"' in dockerfile


def test_image_checks_native_lmms_eval_contract(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()

    assert environment["lmms_eval"]["base_version"] == "0.7.2"
    assert set(environment["lmms_eval"]["required_paths"]) >= {
        "models/chat/qwen3_5.py",
        "tasks/realworldqa/realworldqa.yaml",
        "tasks/videomme/videomme.yaml",
        "tasks/omni_bench/_default_template_yaml",
    }
    assert "python -m pip check" in dockerfile
    assert "verify_lmms_eval_revision" in dockerfile


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

    assert "examples/__init__.py" in push_paths
    assert "examples/__init__.py" in pull_request_paths
    assert "examples/puzzletron/**" in push_paths
    assert "examples/puzzletron/Dockerfile" in pull_request_paths
