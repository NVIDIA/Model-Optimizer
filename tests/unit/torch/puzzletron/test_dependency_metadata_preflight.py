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

"""Tests for the CPU-only pinned dependency metadata preflight."""

import json
from pathlib import Path

import pytest

from examples.puzzletron.ci import preflight_dependency_metadata

_PROJECT_ROOT = Path(__file__).parents[4]


def _environment():
    return json.loads((_PROJECT_ROOT / "examples/puzzletron/ci_environment.json").read_text())


def _metadata(environment, *, grouped_name="nv_grouped_gemm", lmms_wandb="wandb>=0.16.0"):
    grouped_metadata_path = environment["runtime_image"]["grouped_gemm"]["metadata_path"]
    sources = {
        grouped_metadata_path: f'''
PACKAGE_NAME = "{grouped_name}"
setup(name=PACKAGE_NAME)
''',
        "lmms": f'''
[project]
name = "lmms_eval"
dependencies = ["{lmms_wandb}"]
''',
        "automodel": """
[project]
name = "nemo-automodel"
dependencies = ["wandb>=0.28.0"]
""",
    }

    def fetch(url):
        if "grouped_gemm" in url:
            return sources[grouped_metadata_path]
        if "lmms-eval" in url:
            return sources["lmms"]
        return sources["automodel"]

    return fetch


def test_metadata_fetch_uses_fixed_https_host_and_timeout(monkeypatch):
    calls = []

    class Response:
        status = 200

        def read(self):
            return b"metadata"

    class Connection:
        def __init__(self, host, *, timeout):
            calls.append(("connect", host, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def request(self, method, path):
            calls.append(("request", method, path))

        def getresponse(self):
            return Response()

    monkeypatch.setattr(preflight_dependency_metadata, "HTTPSConnection", Connection)
    url = "https://raw.githubusercontent.com/owner/repository/revision/pyproject.toml"

    assert preflight_dependency_metadata._fetch_url(url) == "metadata"
    assert calls == [
        ("connect", "raw.githubusercontent.com", 30),
        ("request", "GET", "/owner/repository/revision/pyproject.toml"),
    ]

    with pytest.raises(ValueError, match="unsupported pinned metadata URL"):
        preflight_dependency_metadata._fetch_url("https://example.com/pyproject.toml")


def test_preflight_accepts_pinned_distribution_names_and_compatible_dependencies():
    environment = _environment()

    preflight_dependency_metadata.validate_pinned_metadata(
        environment, fetch_text=_metadata(environment)
    )


def test_preflight_rejects_vcs_reference_name_mismatch():
    environment = _environment()

    with pytest.raises(ValueError, match="declares distribution 'grouped_gemm'"):
        preflight_dependency_metadata.validate_pinned_metadata(
            environment,
            fetch_text=_metadata(environment, grouped_name="grouped_gemm"),
        )


def test_preflight_rejects_incompatible_exact_dependency_pin():
    environment = _environment()

    with pytest.raises(ValueError, match="incompatible exact dependency pin for 'wandb'"):
        preflight_dependency_metadata.validate_pinned_metadata(
            environment,
            fetch_text=_metadata(environment, lmms_wandb="wandb==0.25.0"),
        )
