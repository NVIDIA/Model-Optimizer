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

"""Tests for Puzzletron CI environment provenance checks."""

import json
import subprocess

import pytest

from examples.puzzletron import ci_environment


class _Distribution:
    def __init__(self, payload: dict | list | str | None):
        self.payload = payload

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        if isinstance(self.payload, str) or self.payload is None:
            return self.payload
        return json.dumps(self.payload)


_EXPECTED_SOURCE = {
    "repository": "https://github.com/Separius/Automodel.git",
    "commit": "b22cd029d806197e249f2cc4a42c5de91713b772",
}


def _pep610_source(repository: str, commit: str) -> dict:
    return {
        "url": repository,
        "vcs_info": {"vcs": "git", "commit_id": commit},
    }


def test_pep610_exact_source_is_accepted(monkeypatch):
    monkeypatch.setattr(
        ci_environment.metadata,
        "distribution",
        lambda _package: _Distribution(
            _pep610_source(_EXPECTED_SOURCE["repository"], _EXPECTED_SOURCE["commit"])
        ),
    )

    ci_environment.verify_installed_vcs_source("nemo-automodel", _EXPECTED_SOURCE)


@pytest.mark.parametrize(
    ("repository", "commit"),
    [
        ("https://github.com/example/Automodel.git", _EXPECTED_SOURCE["commit"]),
        (_EXPECTED_SOURCE["repository"], "0" * 40),
    ],
    ids=("repository", "commit"),
)
def test_pep610_vcs_source_mismatch_is_rejected(monkeypatch, repository, commit):
    monkeypatch.setattr(
        ci_environment.metadata,
        "distribution",
        lambda _package: _Distribution(_pep610_source(repository, commit)),
    )

    with pytest.raises(RuntimeError, match="source mismatch"):
        ci_environment.verify_installed_vcs_source("nemo-automodel", _EXPECTED_SOURCE)


def test_editable_pinned_dependency_must_be_clean(monkeypatch):
    monkeypatch.setattr(
        ci_environment.metadata,
        "distribution",
        lambda _package: _Distribution(
            {"url": "file:///src/automodel", "dir_info": {"editable": True}}
        ),
    )
    outputs = iter(
        [
            "https://github.com/Separius/Automodel.git\n",
            "b22cd029d806197e249f2cc4a42c5de91713b772\n",
            " M nemo_automodel/model.py\n",
        ]
    )

    def check_output(*_args, **kwargs):
        assert kwargs["timeout"] == ci_environment._GIT_TIMEOUT_SECONDS
        return next(outputs)

    monkeypatch.setattr(ci_environment.subprocess, "check_output", check_output)

    with pytest.raises(RuntimeError, match="dependency 'nemo-automodel' is dirty"):
        ci_environment.verify_installed_vcs_source(
            "nemo-automodel",
            _EXPECTED_SOURCE,
        )


def test_missing_package_has_package_named_error(monkeypatch):
    def missing(package: str):
        raise ci_environment.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(ci_environment.metadata, "distribution", missing)

    with pytest.raises(RuntimeError, match="dependency 'nemo-automodel' is not installed"):
        ci_environment.verify_installed_vcs_source("nemo-automodel", _EXPECTED_SOURCE)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (None, "no direct URL metadata"),
        ("{", "malformed direct URL metadata"),
        ([], "invalid direct URL metadata"),
        ({"url": _EXPECTED_SOURCE["repository"], "vcs_info": "git"}, "invalid direct URL metadata"),
    ],
    ids=("absent", "malformed-json", "non-mapping", "invalid-vcs-info"),
)
def test_invalid_direct_url_metadata_has_package_named_error(monkeypatch, payload, reason):
    monkeypatch.setattr(
        ci_environment.metadata,
        "distribution",
        lambda _package: _Distribution(payload),
    )

    with pytest.raises(RuntimeError, match=rf"dependency 'nemo-automodel'.*{reason}"):
        ci_environment.verify_installed_vcs_source("nemo-automodel", _EXPECTED_SOURCE)


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, ["git"]),
        subprocess.TimeoutExpired(["git"], 10),
        OSError("git unavailable"),
    ],
    ids=("failure", "timeout", "os-error"),
)
def test_editable_git_inspection_failure_has_package_named_error(monkeypatch, error):
    monkeypatch.setattr(
        ci_environment.metadata,
        "distribution",
        lambda _package: _Distribution(
            {"url": "file:///src/automodel", "dir_info": {"editable": True}}
        ),
    )

    def fail(*_args, **kwargs):
        assert kwargs["timeout"] == ci_environment._GIT_TIMEOUT_SECONDS
        raise error

    monkeypatch.setattr(ci_environment.subprocess, "check_output", fail)

    with pytest.raises(RuntimeError, match="dependency 'nemo-automodel'.*could not be inspected"):
        ci_environment.verify_installed_vcs_source("nemo-automodel", _EXPECTED_SOURCE)
