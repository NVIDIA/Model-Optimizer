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
_REQUIRED_MODULES = (
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


def _environment_sources(environment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runtime_image = environment.get("runtime_image") or {}
    return {
        "grouped_gemm": runtime_image.get("grouped_gemm") or {},
        "lmms_eval": environment.get("lmms_eval") or {},
        "mamba_ssm": runtime_image.get("mamba_ssm") or {},
        "nemo_automodel": environment.get("nemo_automodel") or {},
        "vllm": environment.get("vllm") or {},
    }


def _require_exact_public_version(value: object, name: str) -> None:
    version = str(value or "")
    parsed = Version(version)
    if str(parsed) != version or parsed.local is not None:
        raise ValueError(f"Puzzletron runtime package {name!r} must use an exact public version")


def _validate_image_identity(environment: dict[str, Any]) -> None:
    gpu_image = environment.get("gpu_image") or {}
    if not _BASE_IMAGE_PATTERN.fullmatch(gpu_image.get("base_image", "")):
        raise ValueError("Puzzletron image base must be an immutable NVIDIA CUDA digest")
    if gpu_image.get("platform") != "linux/amd64":
        raise ValueError("Puzzletron worker image platform must be linux/amd64")


def _validate_pinned_sources(environment: dict[str, Any]) -> None:
    sources = _environment_sources(environment)
    for name, approved_repository in _APPROVED_REPOSITORIES.items():
        source = sources[name]
        if source.get("repository") != approved_repository:
            raise ValueError(f"Puzzletron image source {name!r} must use {approved_repository!r}")
        if not _REVISION_PATTERN.fullmatch(str(source.get("commit", ""))):
            raise ValueError(f"Puzzletron image source {name!r} must use a full Git revision")

    if sources["grouped_gemm"].get("distribution") != "nv-grouped-gemm":
        raise ValueError("Puzzletron grouped_gemm source must declare nv-grouped-gemm")


def _validate_cuda_extensions(environment: dict[str, Any]) -> None:
    runtime_image = environment.get("runtime_image") or {}
    for name in ("causal_conv1d", "flash_linear_attention", "tilelang"):
        _require_exact_public_version(runtime_image.get(name), name)

    mamba_source = runtime_image.get("mamba_ssm") or {}
    _require_exact_public_version(mamba_source.get("base_version"), "mamba_ssm")
    if not re.fullmatch(
        r"[A-Za-z0-9._-]+\.patch", str(mamba_source.get("compatibility_patch", ""))
    ):
        raise ValueError("Puzzletron mamba_ssm compatibility patch must use a safe patch filename")
    if not _SHA256_PATTERN.fullmatch(str(mamba_source.get("compatibility_patch_sha256", ""))):
        raise ValueError("Puzzletron mamba_ssm compatibility patch must declare a SHA-256")

    for name in ("grouped_gemm_cuda_arch_list", "torch_cuda_arch_list"):
        if not re.fullmatch(r"[0-9.]+(?:;[0-9.]+)*", runtime_image.get(name, "")):
            raise ValueError(f"Puzzletron runtime image must declare explicit {name}")


def _validate_worker_assets(environment: dict[str, Any]) -> None:
    gpu_image = environment.get("gpu_image") or {}
    video_decoder = gpu_image.get("video_decoder") or {}
    if video_decoder.get("distribution") != "eva-decord":
        raise ValueError("Puzzletron worker image must use the Linux eva-decord distribution")
    if video_decoder.get("version") != "0.6.1":
        raise ValueError("Puzzletron worker image must pin eva-decord 0.6.1")

    resources = gpu_image.get("nltk_resources")
    if resources != ["punkt", "punkt_tab"]:
        raise ValueError("Puzzletron worker image must declare the required NLTK resources")
    if not _REVISION_PATTERN.fullmatch(str(gpu_image.get("nltk_data_commit", ""))):
        raise ValueError("Puzzletron worker image must pin the NLTK data revision")
    checksums = gpu_image.get("nltk_resource_sha256")
    if not isinstance(checksums, dict) or set(checksums) != set(resources):
        raise ValueError("Puzzletron worker image must checksum every NLTK resource")
    if not all(_SHA256_PATTERN.fullmatch(str(value)) for value in checksums.values()):
        raise ValueError("Puzzletron NLTK resource checksums must be SHA-256 values")

    task_configs = (environment.get("lmms_eval") or {}).get("task_configs")
    if not isinstance(task_configs, list) or not task_configs:
        raise ValueError("Puzzletron worker image must declare LMMS-Eval task configs")
    for task_config in task_configs:
        if not re.fullmatch(r"tasks/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.yaml", str(task_config)):
            raise ValueError(
                "Puzzletron LMMS-Eval task configs must use safe package-relative paths"
            )


def validate_environment_contract(environment: dict[str, Any]) -> None:
    """Validate the immutable inputs and worker assets recorded by the manifest."""

    if environment.get("schema_version") != 1:
        raise ValueError("Puzzletron image environment schema_version must be 1")
    if environment.get("scope") != "puzzletron_v2_worker_ci":
        raise ValueError("Puzzletron image environment has an unexpected scope")
    _validate_image_identity(environment)
    _validate_pinned_sources(environment)
    _validate_cuda_extensions(environment)
    _validate_worker_assets(environment)


def _expected_versions(environment: dict[str, Any]) -> dict[str, str]:
    return {
        "python": environment["python"],
        "torch": environment["torch"],
        "torchvision": environment["torchvision"],
        "transformers": environment["transformers"],
        "lmms-eval": environment["lmms_eval"]["base_version"],
        "nemo-automodel": environment["nemo_automodel"]["base_version"],
        "aiperf": environment["gpu_image"]["aiperf"],
        environment["gpu_image"]["video_decoder"]["distribution"]: environment["gpu_image"][
            "video_decoder"
        ]["version"],
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


def _verify_package_versions(
    environment: dict[str, Any],
    package_version: Callable[[str], str],
    python_version: str,
) -> None:
    expected = _expected_versions(environment)
    actual = {
        "python": python_version,
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


def _verify_vcs_sources(
    environment: dict[str, Any],
    source_verifier: Callable[[str, dict[str, Any]], None],
) -> None:
    runtime_image = environment["runtime_image"]
    sources = {
        "lmms-eval": environment["lmms_eval"],
        "nemo-automodel": environment["nemo_automodel"],
        runtime_image["grouped_gemm"]["distribution"]: runtime_image["grouped_gemm"],
        "vllm": environment["vllm"],
    }
    for package, source in sources.items():
        source_verifier(package, source)


def _verify_cuda_version(environment: dict[str, Any], actual_cuda: object) -> None:
    expected_cuda = environment["gpu_image"]["torch_cuda"]
    if actual_cuda != expected_cuda:
        raise RuntimeError(
            f"Pinned Puzzletron CUDA mismatch: actual={actual_cuda!r}, expected={expected_cuda!r}"
        )


def _verify_runtime_assets(environment: dict[str, Any], imported: dict[str, Any]) -> None:
    lmms_roots = tuple(Path(path) for path in imported["lmms_eval"].__path__)
    for task_config in environment["lmms_eval"]["task_configs"]:
        if not any((root / task_config).is_file() for root in lmms_roots):
            raise RuntimeError(f"Pinned LMMS-Eval task config is missing: {task_config}")

    for resource in environment["gpu_image"]["nltk_resources"]:
        imported["nltk"].data.find(f"tokenizers/{resource}")


def verify_installed_environment(
    environment: dict[str, Any],
    *,
    package_version: Callable[[str], str] = metadata.version,
    source_verifier: Callable[[str, dict[str, Any]], None] = verify_installed_vcs_source,
    module_importer: Callable[[str], Any] = import_module,
    python_version: str | None = None,
    torch_cuda: object = _UNSET,
) -> None:
    """Verify the installed packages, sources, CUDA ABI, and runtime assets."""

    validate_environment_contract(environment)
    _verify_package_versions(
        environment,
        package_version,
        python_version or f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    _verify_vcs_sources(environment, source_verifier)

    if torch_cuda is _UNSET:
        torch_cuda = module_importer("torch").version.cuda
    _verify_cuda_version(environment, torch_cuda)

    imported = {module: module_importer(module) for module in _REQUIRED_MODULES}
    _verify_runtime_assets(environment, imported)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    if args.manifest_only:
        validate_environment_contract(environment)
        print("Puzzletron image manifest: OK")
        return
    verify_installed_environment(environment)
    print("Puzzletron worker environment: OK")


if __name__ == "__main__":
    main()
