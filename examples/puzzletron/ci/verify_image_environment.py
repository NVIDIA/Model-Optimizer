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

"""Validate and verify the repository-owned Puzzletron image environments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import Version

from examples.puzzletron.ci_environment import verify_installed_vcs_source

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["validate_environment_contract", "verify_installed_environment"]

_APPROVED_REPOSITORIES = {
    "grouped_gemm": "https://github.com/fanshiqing/grouped_gemm.git",
    "lmms_eval": "https://github.com/EvolvingLMMs-Lab/lmms-eval.git",
    "mamba_ssm": "https://github.com/state-spaces/mamba.git",
    "nemo_automodel": "https://github.com/Separius/Automodel.git",
    "vllm": "https://github.com/Separius/vllm.git",
}
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BASE_IMAGE_PATTERN = re.compile(r"nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}")
_UNSET = object()


def validate_environment_contract(environment: dict[str, Any]) -> None:
    """Reject mutable or unexpected repositories before trusting the manifest."""

    if environment.get("schema_version") != 1:
        raise ValueError("Puzzletron image environment schema_version must be 1")
    if environment.get("scope") != "puzzletron_v2_worker_ci":
        raise ValueError("Puzzletron image environment has an unexpected scope")

    base_image = environment.get("gpu_image", {}).get("base_image", "")
    if not _BASE_IMAGE_PATTERN.fullmatch(base_image):
        raise ValueError("Puzzletron image base must be an immutable NVIDIA CUDA digest")

    sources = {
        "grouped_gemm": (environment.get("runtime_image") or {}).get("grouped_gemm") or {},
        "mamba_ssm": (environment.get("runtime_image") or {}).get("mamba_ssm") or {},
        **{key: environment.get(key) or {} for key in ("lmms_eval", "nemo_automodel", "vllm")},
    }
    for key, approved_repository in _APPROVED_REPOSITORIES.items():
        source = sources[key]
        if source.get("repository") != approved_repository:
            raise ValueError(f"Puzzletron image source {key!r} must use {approved_repository!r}")
        if not _REVISION_PATTERN.fullmatch(str(source.get("commit", ""))):
            raise ValueError(f"Puzzletron image source {key!r} must use a full Git revision")

    if sources["grouped_gemm"].get("distribution") != "nv-grouped-gemm":
        raise ValueError("Puzzletron grouped_gemm source must declare nv-grouped-gemm")

    runtime_image = environment.get("runtime_image") or {}
    for key in ("causal_conv1d", "flash_linear_attention", "tilelang"):
        version = runtime_image.get(key, "")
        parsed_version = Version(str(version))
        if str(parsed_version) != version or parsed_version.local is not None:
            raise ValueError(f"Puzzletron runtime package {key!r} must use an exact public version")
    mamba_source = runtime_image.get("mamba_ssm") or {}
    mamba_version = mamba_source.get("base_version", "")
    parsed_mamba_version = Version(str(mamba_version))
    if str(parsed_mamba_version) != mamba_version or parsed_mamba_version.local is not None:
        raise ValueError("Puzzletron runtime package 'mamba_ssm' must use an exact public version")
    if not re.fullmatch(
        r"[A-Za-z0-9._-]+\.patch", str(mamba_source.get("compatibility_patch", ""))
    ):
        raise ValueError("Puzzletron mamba_ssm compatibility patch must use a safe patch filename")
    if not _SHA256_PATTERN.fullmatch(str(mamba_source.get("compatibility_patch_sha256", ""))):
        raise ValueError("Puzzletron mamba_ssm compatibility patch must declare a SHA-256")
    for key in ("grouped_gemm_cuda_arch_list", "torch_cuda_arch_list"):
        if not re.fullmatch(r"[0-9.]+(?:;[0-9.]+)*", runtime_image.get(key, "")):
            raise ValueError(f"Puzzletron runtime image must declare explicit {key}")
    gpu_image = environment.get("gpu_image") or {}
    decord_version = gpu_image.get("decord", "")
    decord_wheel_url = gpu_image.get("decord_wheel_url", "")
    if not re.fullmatch(
        rf"https://files\.pythonhosted\.org/.*/decord-{re.escape(decord_version)}-"
        r"py3-none-manylinux2010_x86_64\.whl",
        decord_wheel_url,
    ):
        raise ValueError("Puzzletron worker image must pin the Linux x86_64 decord wheel")
    if not re.fullmatch(r"[0-9a-f]{64}", gpu_image.get("decord_wheel_sha256", "")):
        raise ValueError("Puzzletron worker image must pin the decord wheel checksum")
    if gpu_image.get("nltk_resources") != ["punkt", "punkt_tab"]:
        raise ValueError("Puzzletron worker image must declare the required NLTK resources")
    if not _REVISION_PATTERN.fullmatch(str(gpu_image.get("nltk_data_commit", ""))):
        raise ValueError("Puzzletron worker image must pin the NLTK data revision")
    nltk_resource_sha256 = gpu_image.get("nltk_resource_sha256")
    if not isinstance(nltk_resource_sha256, dict) or set(nltk_resource_sha256) != set(
        gpu_image["nltk_resources"]
    ):
        raise ValueError("Puzzletron worker image must checksum every NLTK resource")
    if not all(_SHA256_PATTERN.fullmatch(str(value)) for value in nltk_resource_sha256.values()):
        raise ValueError("Puzzletron NLTK resource checksums must be SHA-256 values")
    task_configs = environment.get("lmms_eval", {}).get("task_configs")
    if not isinstance(task_configs, list) or not task_configs:
        raise ValueError("Puzzletron worker image must declare LMMS-Eval task configs")
    for task_config in task_configs:
        if not re.fullmatch(r"tasks/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.yaml", str(task_config)):
            raise ValueError(
                "Puzzletron LMMS-Eval task configs must use safe package-relative paths"
            )


def _expected_versions(environment: dict[str, Any]) -> dict[str, str]:
    return {
        "python": environment["python"],
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
        environment["runtime_image"]["grouped_gemm"]["distribution"]: environment["runtime_image"][
            "grouped_gemm"
        ]["base_version"],
        "mamba-ssm": environment["runtime_image"]["mamba_ssm"]["base_version"],
        "tilelang": environment["runtime_image"]["tilelang"],
    }


def verify_installed_environment(
    environment: dict[str, Any],
    *,
    package_version: Callable[[str], str] = metadata.version,
    source_verifier: Callable[[str, dict[str, Any]], None] = verify_installed_vcs_source,
    module_importer: Callable[[str], Any] = import_module,
    python_version: str | None = None,
    torch_cuda: object = _UNSET,
) -> None:
    """Verify package, VCS, CUDA, and runtime invariants."""

    validate_environment_contract(environment)

    expected = _expected_versions(environment)
    actual = {
        "python": python_version or f"{sys.version_info.major}.{sys.version_info.minor}",
        **{
            package: Version(package_version(package)).public
            for package in expected
            if package != "python"
        },
    }
    mismatches = {
        package: (actual[package], expected_version)
        for package, expected_version in expected.items()
        if actual[package] != expected_version
    }
    if mismatches:
        raise RuntimeError(f"Pinned Puzzletron image mismatch: {mismatches}")

    sources = {
        "lmms-eval": environment["lmms_eval"],
        "nemo-automodel": environment["nemo_automodel"],
        environment["runtime_image"]["grouped_gemm"]["distribution"]: environment["runtime_image"][
            "grouped_gemm"
        ],
        "vllm": environment["vllm"],
    }
    for package, source in sources.items():
        source_verifier(package, source)

    if torch_cuda is _UNSET:
        torch_cuda = module_importer("torch").version.cuda
    expected_cuda = environment["gpu_image"]["torch_cuda"]
    if torch_cuda != expected_cuda:
        raise RuntimeError(
            f"Pinned Puzzletron CUDA mismatch: actual={torch_cuda!r}, expected={expected_cuda!r}"
        )

    imported = {
        module: module_importer(module)
        for module in (
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
        )
    }
    lmms_roots = tuple(Path(path) for path in imported["lmms_eval"].__path__)
    for task_config in environment["lmms_eval"]["task_configs"]:
        if not any((root / task_config).is_file() for root in lmms_roots):
            raise RuntimeError(f"Pinned LMMS-Eval task config is missing: {task_config}")
    for resource in environment["gpu_image"]["nltk_resources"]:
        imported["nltk"].data.find(f"tokenizers/{resource}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    validate_environment_contract(environment)
    if not args.manifest_only:
        verify_installed_environment(environment)


if __name__ == "__main__":
    main()
