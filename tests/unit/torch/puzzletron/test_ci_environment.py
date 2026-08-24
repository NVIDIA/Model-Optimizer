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
import sys
from importlib import metadata

import pytest

import noxfile
from examples.puzzletron import ci_environment


class _Distribution:
    def __init__(self, payload: dict):
        self.payload = payload

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
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
    monkeypatch.setattr(
        ci_environment.subprocess,
        "check_output",
        lambda *_args, **_kwargs: next(outputs),
    )

    with pytest.raises(RuntimeError, match="dependency 'nemo-automodel' is dirty"):
        ci_environment.verify_installed_vcs_source(
            "nemo-automodel",
            _EXPECTED_SOURCE,
        )


def test_nox_verifier_executes_scalar_version_and_exact_vcs_checks(monkeypatch):
    lmms_source = {
        "base_version": "7.8.9",
        "repository": "https://example.test/lmms-eval.git",
        "commit": "1" * 40,
    }
    automodel_source = {
        "base_version": "4.5.6",
        "repository": "https://example.test/Automodel.git",
        "commit": "2" * 40,
    }
    expected_versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": "1.2.3",
        "torchvision": "2.3.4",
        "transformers": "3.4.5",
        "lmms-eval": lmms_source["base_version"],
        "nemo-automodel": automodel_source["base_version"],
    }
    monkeypatch.setattr(
        noxfile,
        "PUZZLETRON_V2_CI_ENVIRONMENT",
        {
            **expected_versions,
            "lmms_eval": lmms_source,
            "nemo_automodel": automodel_source,
        },
    )
    monkeypatch.setattr(noxfile, "PUZZLETRON_V2_LMMS_SOURCE", lmms_source)
    monkeypatch.setattr(noxfile, "PUZZLETRON_V2_AUTOMODEL_SOURCE", automodel_source)
    monkeypatch.setattr(metadata, "version", lambda package: expected_versions[package])
    vcs_calls = []
    monkeypatch.setattr(
        ci_environment,
        "verify_installed_vcs_source",
        lambda package, source: vcs_calls.append((package, source)),
    )

    class ExecutingSession:
        def run(self, python, flag, script):
            assert (python, flag) == ("python", "-c")
            exec(compile(script, "<nox-verifier>", "exec"), {})

    noxfile._verify_puzzletron_v2_environment(ExecutingSession())

    assert vcs_calls == [
        ("lmms-eval", lmms_source),
        ("nemo-automodel", automodel_source),
    ]


def test_puzzletron_nox_session_verifies_environment_before_pytest(monkeypatch):
    events = []

    class RecordingSession:
        def install(self, *args):
            events.append(("install", args))

        def run(self, *args):
            events.append(("run", args))

    monkeypatch.setattr(
        noxfile,
        "_verify_puzzletron_v2_environment",
        lambda session: events.append(("verify", session)),
    )
    session = RecordingSession()

    noxfile.puzzletron_v2.func(session)

    verify_index = events.index(("verify", session))
    pytest_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "run" and event[1][:3] == ("python", "-m", "pytest")
    )
    assert verify_index < pytest_index
