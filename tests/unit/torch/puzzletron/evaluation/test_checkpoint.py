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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from examples.puzzletron.evaluation import checkpoint


def _distribution(provenance):
    return SimpleNamespace(read_text=lambda filename: json.dumps(provenance))


def test_load_runner_restores_import_state(monkeypatch):
    package = "modelopt.torch.puzzletron.evaluation"
    original = ModuleType(package)
    monkeypatch.setitem(sys.modules, package, original)
    for name in (
        "modelopt.torch.puzzletron.orchestration.mesh",
        "modelopt.torch.puzzletron.evaluation.lmms",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    runner = checkpoint._load_runner()

    assert callable(runner)
    assert sys.modules[package] is original
    assert "modelopt.torch.puzzletron.orchestration.mesh" not in sys.modules
    assert "modelopt.torch.puzzletron.evaluation.lmms" not in sys.modules


def test_verify_lmms_eval_revision_rejects_unpatched_vcs_install(monkeypatch):
    provenance = {"vcs_info": {"commit_id": checkpoint.LMMS_EVAL_REVISION}}
    monkeypatch.setattr(checkpoint, "_imported_lmms_eval_revision", lambda: None)
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )

    with pytest.raises(RuntimeError, match="requires a verified editable"):
        checkpoint.verify_lmms_eval_revision()


def test_verify_lmms_eval_revision_accepts_clean_pinned_editable_checkout(monkeypatch, tmp_path):
    provenance = {"dir_info": {"editable": True}, "url": tmp_path.as_uri()}
    monkeypatch.setattr(checkpoint, "_imported_lmms_eval_revision", lambda: None)
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )
    monkeypatch.setattr(
        checkpoint.ci_environment, "verify_installed_vcs_source", lambda _package, _source: None
    )

    assert checkpoint.verify_lmms_eval_revision() == checkpoint.LMMS_EVAL_REVISION


@pytest.mark.parametrize("distribution_available", [False, True])
def test_verify_lmms_eval_revision_accepts_clean_imported_source_checkout(
    monkeypatch, tmp_path, distribution_available
):
    package = tmp_path / "lmms_eval"
    package.mkdir()
    (tmp_path / ".git").mkdir()

    def distribution(_name):
        if not distribution_available:
            raise checkpoint.importlib.metadata.PackageNotFoundError
        return _distribution(None)

    monkeypatch.setattr(checkpoint.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(
        checkpoint.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    monkeypatch.setattr(
        checkpoint.ci_environment,
        "verify_vcs_checkout",
        lambda _checkout, _package, _source: checkpoint.LMMS_EVAL_REVISION,
    )

    assert checkpoint.verify_lmms_eval_revision() == checkpoint.LMMS_EVAL_REVISION


def test_verify_lmms_eval_revision_rejects_dirty_editable_checkout(monkeypatch, tmp_path):
    provenance = {"dir_info": {"editable": True}, "url": tmp_path.as_uri()}
    monkeypatch.setattr(checkpoint, "_imported_lmms_eval_revision", lambda: None)
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )
    monkeypatch.setattr(
        checkpoint.ci_environment,
        "verify_installed_vcs_source",
        lambda _package, _source: (_ for _ in ()).throw(
            RuntimeError("compatibility patch files differ")
        ),
    )

    with pytest.raises(RuntimeError, match="compatibility patch files differ"):
        checkpoint.verify_lmms_eval_revision()


@pytest.mark.parametrize(
    "provenance",
    [
        {"dir_info": {"editable": True}, "url": "https://example.com/lmms-eval"},
        {"dir_info": {"editable": False}, "url": Path.cwd().as_uri()},
    ],
)
def test_verify_lmms_eval_revision_rejects_unverifiable_editable_install(monkeypatch, provenance):
    monkeypatch.setattr(checkpoint, "_imported_lmms_eval_revision", lambda: None)
    monkeypatch.setattr(
        checkpoint.importlib.metadata, "distribution", lambda _name: _distribution(provenance)
    )
    monkeypatch.setattr(
        checkpoint.ci_environment,
        "verify_installed_vcs_source",
        lambda _package, _source: (_ for _ in ()).throw(RuntimeError("source mismatch")),
    )

    with pytest.raises(RuntimeError, match="source mismatch"):
        checkpoint.verify_lmms_eval_revision()
