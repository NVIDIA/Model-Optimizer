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

"""Tests for fail-closed Puzzletron stage terminal states."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import modelopt.torch.puzzletron.stage_runner as stage_runner
import modelopt.torch.puzzletron.stages.convert as convert_stages
from examples.puzzletron import main as puzzletron_main
from examples.puzzletron.main import _completion_is_valid, _validate_worker_result
from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.stage_runner import StageResult, run_stage


def _config(tmp_path, **sections):
    return {"puzzle_dir": str(tmp_path), "experiment": {"dir": str(tmp_path)}, **sections}


def _write_successful_convert_result(tmp_path, config):
    artifact = tmp_path / "ckpts" / "teacher" / "config.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    return manifest, StageResult("convert", "success", manifest_path, "done")


def _write_imported_convert_result(tmp_path, config, *, sha256=None):
    artifact = tmp_path / "ckpts" / "teacher" / "config.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = StageManifest(stage="convert", status="imported", config=config)
    manifest_path = tmp_path / "manifests" / "convert.json"
    payload = manifest.to_dict()
    payload["output_inventory"] = [
        {
            "path": "ckpts/teacher/config.json",
            "size": artifact.stat().st_size,
            "sha256": sha256 or hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    ]
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(payload) + "\n")
    return StageResult("convert", "imported", manifest_path, "imported")


def test_required_stage_missing_handler_fails_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_runner, "_resolve_capabilities", lambda _config: None)

    with pytest.raises(RuntimeError, match="required stage 'convert' has no registered handler"):
        run_stage(_config(tmp_path), "convert", handlers={})

    assert not (tmp_path / "manifests" / "convert.json").exists()


def test_optional_missing_handler_writes_typed_optional_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_runner, "_resolve_capabilities", lambda _config: None)

    result = run_stage(
        _config(tmp_path, aiperf={"enabled": True}),
        "aiperf",
        handlers={},
    )

    payload = json.loads(result.manifest_path.read_text())
    assert result.status == payload["status"] == "skipped"
    assert result.skip_reason == payload["skip_reason"] == "optional"


def test_explicitly_disabled_stage_does_not_call_handler(tmp_path):
    def unexpected_handler(_config, _manifest):
        raise AssertionError("disabled stage handler must not run")

    config = _config(tmp_path, aiperf={"enabled": False})
    result = run_stage(config, "aiperf", handlers={"aiperf": unexpected_handler})

    payload = json.loads(result.manifest_path.read_text())
    assert result.status == "skipped"
    assert payload["skip_reason"] == "disabled"
    assert _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")


def test_existing_teacher_checkpoint_completes_convert_successfully(tmp_path, monkeypatch):
    teacher_dir = tmp_path / "ckpts" / "teacher"
    teacher_dir.mkdir(parents=True)
    config = _config(tmp_path, model={"source": "unused"})
    teacher_config = SimpleNamespace(architectures=["AnyModel"])
    descriptor = SimpleNamespace()

    monkeypatch.setattr(convert_stages, "_register_automodel_config_aliases", lambda: None)
    monkeypatch.setattr(convert_stages, "_is_complete_checkpoint", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        convert_stages,
        "_descriptor_checkpoint_layout_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        convert_stages.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: teacher_config,
    )
    monkeypatch.setattr(
        convert_stages,
        "resolve_descriptor_from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(descriptor=descriptor),
    )
    monkeypatch.setattr(
        convert_stages,
        "model_identity",
        lambda _config: SimpleNamespace(value="teacher"),
    )

    result = convert_stages.convert_stage(config, StageManifest(stage="convert", config=config))
    payload = json.loads(result.manifest_path.read_text())

    assert result.status == payload["status"] == "success"
    assert payload["outputs"]["skipped"] is True


def test_worker_rejects_success_without_required_artifact(tmp_path):
    config = _config(tmp_path)
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    result = StageResult("convert", "success", manifest_path, "done")

    with pytest.raises(RuntimeError, match="ckpts/teacher/config.json"):
        _validate_worker_result(config, result)


def test_worker_rejects_width_success_without_activation_artifacts(tmp_path):
    config = _config(tmp_path)
    manifest = StageManifest(stage="width_importance", config=config)
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "width_importance.json"
    write_stage_manifest(manifest_path, manifest)
    result = StageResult("width_importance", "success", manifest_path, "done")

    with pytest.raises(RuntimeError, match="canonical artifact validation"):
        _validate_worker_result(config, result)


def test_worker_rejects_convert_without_config_dependent_library(tmp_path):
    config = _config(tmp_path, vllm_stats={"enabled": True})
    artifact = tmp_path / "ckpts" / "teacher" / "config.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = StageManifest(stage="convert", config=config)
    manifest.complete()
    manifest_path = tmp_path / "manifests" / "convert.json"
    write_stage_manifest(manifest_path, manifest)
    result = StageResult("convert", "success", manifest_path, "done")

    with pytest.raises(RuntimeError, match="subblock_library.json"):
        _validate_worker_result(config, result)


def test_worker_rejects_result_stage_disagreement(tmp_path):
    config = _config(tmp_path)
    _manifest, result = _write_successful_convert_result(tmp_path, config)

    with pytest.raises(RuntimeError, match="returned result for stage 'sort'"):
        _validate_worker_result(
            config,
            StageResult("sort", "success", result.manifest_path, "done"),
            expected_stage="convert",
        )


def test_worker_rejects_manifest_path_disagreement(tmp_path):
    config = _config(tmp_path)
    _manifest, _result = _write_successful_convert_result(tmp_path, config)
    wrong_path = tmp_path / "manifests" / "other.json"
    with pytest.raises(RuntimeError, match="returned manifest path"):
        _validate_worker_result(
            config,
            StageResult("convert", "success", wrong_path, "done"),
            expected_stage="convert",
        )


def test_worker_rejects_manifest_stage_disagreement(tmp_path):
    config = _config(tmp_path)
    manifest, result = _write_successful_convert_result(tmp_path, config)
    manifest.stage = "sort"
    write_stage_manifest(result.manifest_path, manifest)
    with pytest.raises(RuntimeError, match="manifest identifies stage 'sort'"):
        _validate_worker_result(config, result, expected_stage="convert")


def test_worker_rejects_result_status_disagreement(tmp_path):
    config = _config(tmp_path)
    _manifest, successful_result = _write_successful_convert_result(tmp_path, config)
    result = StageResult(
        "convert",
        "imported",
        successful_result.manifest_path,
        "done",
    )

    with pytest.raises(RuntimeError, match="result status 'imported' disagrees"):
        _validate_worker_result(config, result)


def test_worker_rejects_skip_reason_disagreement(tmp_path):
    config = _config(tmp_path, aiperf={"enabled": False})
    manifest = StageManifest(
        stage="aiperf",
        status="skipped",
        skip_reason="disabled",
        config=config,
    )
    manifest_path = tmp_path / "manifests" / "aiperf.json"
    write_stage_manifest(manifest_path, manifest)
    result = StageResult("aiperf", "skipped", manifest_path, "done", "optional")

    with pytest.raises(RuntimeError, match="result skip reason 'optional' disagrees"):
        _validate_worker_result(config, result)


def test_worker_accepts_imported_manifest_with_required_artifact(tmp_path):
    config = _config(tmp_path)
    result = _write_imported_convert_result(tmp_path, config)

    _validate_worker_result(config, result)


def test_imported_completion_rejects_forged_artifact_digest(tmp_path):
    config = _config(tmp_path)
    _write_imported_convert_result(tmp_path, config, sha256="0" * 64)

    assert not _completion_is_valid(config, tmp_path / "config.yaml", "convert")


def test_resume_rejects_untyped_or_now_enabled_skip(tmp_path):
    config = _config(tmp_path, aiperf={"enabled": False})
    manifest_path = tmp_path / "manifests" / "aiperf.json"
    manifest = StageManifest(stage="aiperf", status="skipped", config=config)
    write_stage_manifest(manifest_path, manifest)
    assert not _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")

    manifest.skip_reason = "disabled"
    write_stage_manifest(manifest_path, manifest)
    assert _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")

    config["aiperf"]["enabled"] = True
    assert not _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")


def test_resume_rechecks_canonical_artifact_contract(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manifest = StageManifest(stage="width_importance", config=config)
    manifest.complete()
    write_stage_manifest(tmp_path / "manifests" / "width_importance.json", manifest)
    monkeypatch.setattr(puzzletron_main, "check_marker", lambda *_args, **_kwargs: True)

    assert not _completion_is_valid(config, tmp_path / "config.yaml", "width_importance")
