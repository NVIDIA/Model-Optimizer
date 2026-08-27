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

"""Check exact VCS package metadata before allocating a GPU image builder."""

from __future__ import annotations

import argparse
import ast
import json
import re
from http import HTTPStatus
from http.client import HTTPSConnection
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from examples.puzzletron.ci.verify_image_environment import validate_environment_contract

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["validate_pinned_metadata"]

_GITHUB_REPOSITORY = re.compile(r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)\.git")
_REVISION = re.compile(r"[0-9a-f]{40}")
_RAW_GITHUB_HOST = "raw.githubusercontent.com"


def _raw_url(source: dict[str, Any]) -> str:
    repository = str(source.get("repository", ""))
    match = _GITHUB_REPOSITORY.fullmatch(repository)
    revision = str(source.get("commit", ""))
    metadata_path = str(source.get("metadata_path", ""))
    if match is None or not _REVISION.fullmatch(revision):
        raise ValueError(f"unsupported pinned metadata source: {repository!r}@{revision!r}")
    if metadata_path not in {"pyproject.toml", "setup.py"}:
        raise ValueError(f"unsupported package metadata path: {metadata_path!r}")
    return (
        "https://raw.githubusercontent.com/"
        f"{match.group('owner')}/{match.group('repo')}/{revision}/{metadata_path}"
    )


def _read_setup_name(text: str) -> str:
    tree = ast.parse(text)
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "setup":
            continue
        name = next((keyword.value for keyword in node.keywords if keyword.arg == "name"), None)
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            return name.value
        if isinstance(name, ast.Name) and name.id in constants:
            return constants[name.id]
    raise ValueError("setup.py does not declare a statically inspectable distribution name")


def _fetch_url(url: str) -> str:
    """Fetch metadata only from the raw GitHub host emitted by ``_raw_url``."""

    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != _RAW_GITHUB_HOST
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"unsupported pinned metadata URL: {url!r}")
    with HTTPSConnection(_RAW_GITHUB_HOST, timeout=30) as connection:
        connection.request("GET", parsed.path)
        response = connection.getresponse()
        if response.status != HTTPStatus.OK:
            raise ValueError(f"pinned metadata request failed with HTTP status {response.status}")
        return response.read().decode()


def _parse_metadata(metadata_path: str, text: str) -> tuple[str, list[str]]:
    if metadata_path == "setup.py":
        return _read_setup_name(text), []
    project = tomllib.loads(text).get("project") or {}
    name = project.get("name")
    if not isinstance(name, str):
        raise ValueError("pyproject.toml does not declare project.name")
    dependencies = project.get("dependencies") or []
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) for dependency in dependencies
    ):
        raise ValueError("pyproject.toml project.dependencies must be a list of strings")
    return name, dependencies


def _exact_versions(requirements: Iterable[Requirement]) -> set[Version]:
    return {
        Version(specifier.version)
        for requirement in requirements
        for specifier in requirement.specifier
        if specifier.operator in {"==", "==="} and "*" not in specifier.version
    }


def _validate_dependency_compatibility(dependencies: Iterable[str]) -> None:
    requirements: dict[str, list[Requirement]] = {}
    for dependency in dependencies:
        requirement = Requirement(dependency)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        requirements.setdefault(canonicalize_name(requirement.name), []).append(requirement)

    for name, package_requirements in requirements.items():
        for version in _exact_versions(package_requirements):
            incompatible = [
                str(requirement)
                for requirement in package_requirements
                if version not in requirement.specifier
            ]
            if incompatible:
                constraints = sorted(str(requirement) for requirement in package_requirements)
                raise ValueError(
                    f"incompatible exact dependency pin for {name!r}: "
                    f"{version} does not satisfy {constraints}"
                )


def validate_pinned_metadata(
    environment: dict[str, Any],
    *,
    fetch_text: Callable[[str], str] | None = None,
) -> None:
    """Validate VCS distribution names and directly declared exact-pin compatibility."""
    validate_environment_contract(environment)
    if fetch_text is None:
        fetch_text = _fetch_url

    sources = {
        "grouped_gemm": environment["runtime_image"]["grouped_gemm"],
        "lmms_eval": environment["lmms_eval"],
        "nemo_automodel": environment["nemo_automodel"],
    }
    dependencies = []
    for key, source in sources.items():
        url = _raw_url(source)
        actual_name, source_dependencies = _parse_metadata(source["metadata_path"], fetch_text(url))
        expected_name = source["distribution"]
        if canonicalize_name(actual_name) != canonicalize_name(expected_name):
            raise ValueError(
                f"pinned source {key!r} declares distribution {actual_name!r}, "
                f"not {expected_name!r}"
            )
        dependencies.extend(source_dependencies)
    _validate_dependency_compatibility(dependencies)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_pinned_metadata(json.loads(args.environment.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
