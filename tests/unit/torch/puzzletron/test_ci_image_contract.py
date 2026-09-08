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
import subprocess

import yaml


def test_image_recipe_records_pinned_environment(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()
    requirements = (puzzletron_root / "requirements.txt").read_text().splitlines()

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
    for package in ("aiohttp", "antlr4-python3-runtime", "math-verify", "ray"):
        assert f"{package}=={environment[package]}" in requirements

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


def test_lmms_eval_compatibility_patch_reconciles_worker_dependencies(project_root_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    dockerfile = (puzzletron_root / "Dockerfile").read_text()

    lmms_source = environment["lmms_eval"]
    patch = puzzletron_root / "patches" / lmms_source["compatibility_patch"]
    patch_bytes = patch.read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest() == lmms_source["compatibility_patch_sha256"]
    assert lmms_source["compatibility_patch_files"] == [
        "lmms_eval/loggers/wandb_logger.py",
        "lmms_eval/models/simple/vllm.py",
        "lmms_eval/tasks/emma/utils.py",
        "lmms_eval/tasks/mathvision/eval_utils.py",
        "lmms_eval/tasks/stare/utils.py",
        "pyproject.toml",
    ]
    assert lmms_source["compatibility_patch_context_lines"] == 0
    assert '"$(pin lmms_eval.repository)" "${LMMS_EVAL_ROOT}"' in dockerfile
    assert 'git -C "${LMMS_EVAL_ROOT}" checkout --detach "$(pin lmms_eval.commit)"' in dockerfile
    assert 'git -C "${LMMS_EVAL_ROOT}" apply --unidiff-zero --check' in dockerfile
    assert 'git -C "${LMMS_EVAL_ROOT}" apply --unidiff-zero "/opt/puzzletron/patches/' in dockerfile
    assert 'python -m pip install -e "${LMMS_EVAL_ROOT}[qwen]"' in dockerfile


def test_lmms_eval_vllm_patch_preserves_task_sampling(project_root_path, tmp_path):
    puzzletron_root = project_root_path / "examples/puzzletron"
    environment = json.loads((puzzletron_root / "ci_environment.json").read_text())
    patch_text = (
        puzzletron_root / "patches" / environment["lmms_eval"]["compatibility_patch"]
    ).read_text()
    marker = "diff --git a/lmms_eval/models/simple/vllm.py b/lmms_eval/models/simple/vllm.py"
    start = patch_text.index(marker)
    end = patch_text.index("\ndiff --git ", start + len(marker)) + 1
    vllm_patch = tmp_path / "vllm.patch"
    vllm_patch.write_text(patch_text[start:end])

    correct_sampling = (
        "                    sampling_params = "
        "SamplingParams(**self._build_sampling_params_dict(gen_kwargs))"
    )
    undefined_overwrite = "                sampling_params = SamplingParams(**params)"

    def write_fixture(root):
        source = root / "lmms_eval/models/simple/vllm.py"
        source.parent.mkdir(parents=True)
        lines = ["# pinned upstream fixture"] * 522
        lines[474] = (
            '                    gen_kwargs["max_new_tokens"] = '
            'self._select_max_new_tokens(gen_kwargs.get("max_new_tokens"))'
        )
        lines[477] = correct_sampling
        lines[520] = undefined_overwrite
        lines[521] = (
            '                self._write_watchdog_heartbeat("chat_start", '
            "batch_idx=batch_idx, batch_requests=batch_requests)"
        )
        source.write_text("\n".join(lines) + "\n")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        return source

    exact_checkout = tmp_path / "exact"
    exact_checkout.mkdir()
    exact_source = write_fixture(exact_checkout)
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(vllm_patch)],
        cwd=exact_checkout,
        check=True,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", str(vllm_patch)],
        cwd=exact_checkout,
        check=True,
    )
    patched_source = exact_source.read_text()
    assert correct_sampling in patched_source
    assert undefined_overwrite not in patched_source


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
