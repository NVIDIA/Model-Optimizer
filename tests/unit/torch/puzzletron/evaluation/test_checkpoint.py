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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.puzzletron.evaluation import checkpoint


def _distribution(provenance):
    return SimpleNamespace(read_text=lambda filename: json.dumps(provenance))


def test_verify_lmms_eval_revision_accepts_pinned_vcs_install(monkeypatch):
    provenance = {"vcs_info": {"commit_id": checkpoint.LMMS_EVAL_REVISION}}
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )

    assert checkpoint.verify_lmms_eval_revision() == checkpoint.LMMS_EVAL_REVISION


def test_verify_lmms_eval_revision_accepts_clean_pinned_editable_checkout(monkeypatch, tmp_path):
    provenance = {"dir_info": {"editable": True}, "url": tmp_path.as_uri()}
    responses = [
        SimpleNamespace(stdout=f"{checkpoint.LMMS_EVAL_REVISION}\n"),
        SimpleNamespace(stdout=""),
    ]
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )
    monkeypatch.setattr(checkpoint.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0))

    assert checkpoint.verify_lmms_eval_revision() == checkpoint.LMMS_EVAL_REVISION


def test_verify_lmms_eval_revision_rejects_dirty_editable_checkout(monkeypatch, tmp_path):
    provenance = {"dir_info": {"editable": True}, "url": tmp_path.as_uri()}
    responses = [
        SimpleNamespace(stdout=f"{checkpoint.LMMS_EVAL_REVISION}\n"),
        SimpleNamespace(stdout=" M task.yaml\n"),
    ]
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )
    monkeypatch.setattr(checkpoint.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0))

    with pytest.raises(RuntimeError, match="contains local changes"):
        checkpoint.verify_lmms_eval_revision()


@pytest.mark.parametrize(
    "provenance",
    [
        {"dir_info": {"editable": True}, "url": "https://example.com/lmms-eval"},
        {"dir_info": {"editable": False}, "url": Path.cwd().as_uri()},
    ],
)
def test_verify_lmms_eval_revision_rejects_unverifiable_editable_install(monkeypatch, provenance):
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )

    with pytest.raises(RuntimeError, match="found unknown"):
        checkpoint.verify_lmms_eval_revision()
