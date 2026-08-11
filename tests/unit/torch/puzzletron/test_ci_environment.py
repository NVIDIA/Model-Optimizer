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

import pytest

from examples.puzzletron import ci_environment


class _Distribution:
    def __init__(self, payload: dict):
        self.payload = payload

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return json.dumps(self.payload)


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
            {
                "repository": "https://github.com/Separius/Automodel.git",
                "commit": "b22cd029d806197e249f2cc4a42c5de91713b772",
            },
        )
