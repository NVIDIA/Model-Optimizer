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

import json
from types import SimpleNamespace

import pytest

import modelopt.torch.puzzletron.stage_runner as stage_runner
import modelopt.torch.puzzletron.stages.convert as convert_stages
import modelopt.torch.puzzletron.stages.diagnostics as diagnostic_stages
import modelopt.torch.puzzletron.stages.future as future_stages
import modelopt.torch.puzzletron.stages.pipeline as pipeline_stages
from examples.puzzletron import main as puzzletron_main
from examples.puzzletron import tokenize_data as tokenize_data_module
from examples.puzzletron.main import _completion_is_valid, _validate_worker_result
from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.stage_runner import StageResult, run_stage
from modelopt.torch.puzzletron.stages.graph import StageSkipReason


@pytest.mark.parametrize(
    ("stage", "sections"),
    [("convert", {}), ("aiperf", {"aiperf": {"enabled": True}})],
)
def test_enabled_stage_missing_handler_fails_without_manifest(
    tmp_path, monkeypatch, stage, sections
):
    monkeypatch.setattr(stage_runner, "_resolve_capabilities", lambda _config: None)

    with pytest.raises(RuntimeError, match=rf"enabled stage '{stage}' has no registered handler"):
        run_stage(_config(tmp_path, **sections), stage, handlers={})

    assert not (tmp_path / "manifests" / f"{stage}.json").exists()


def test_explicitly_disabled_stage_does_not_call_handler(tmp_path):
    def unexpected_handler(_config, _manifest):
        raise AssertionError("disabled stage handler must not run")

    config = _config(tmp_path, aiperf={"enabled": False})
    result = run_stage(config, "aiperf", handlers={"aiperf": unexpected_handler})

    payload = json.loads(result.manifest_path.read_text())
    assert result.status == "skipped"
    assert payload["skip_reason"] == "disabled"
    assert _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")


@pytest.mark.parametrize(
    ("stage", "section", "handler"),
    [
        ("aiperf", "aiperf", future_stages.aiperf_stage),
        ("zero_shot_evaluation", "zero_shot_evaluation", future_stages.evaluation_stage),
        (
            "global_distillation_sanity",
            "global_distillation_sanity",
            future_stages.distillation_overfit_stage,
        ),
        ("width_sanity", "width_sanity", diagnostic_stages.activation_diagnostic_stage),
        (
            "bypass_diagnostic",
            "bypass_diagnostic",
            diagnostic_stages.bypass_diagnostic_stage,
        ),
        ("bypass", "bypass", pipeline_stages.bypass_stage),
    ],
)
def test_disabled_stage_handlers_record_canonical_skip_reason(
    tmp_path, monkeypatch, stage, section, handler
):
    runtime_config = SimpleNamespace(puzzle_dir=str(tmp_path))
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.pipeline_config.load_runtime_hydra_config",
        lambda _config: runtime_config,
    )
    monkeypatch.setattr(
        pipeline_stages,
        "load_runtime_hydra_config",
        lambda _config: runtime_config,
    )
    monkeypatch.setattr(pipeline_stages, "ensure_scoring_parent", lambda *_args, **_kwargs: None)
    config = _config(tmp_path, **{section: {"enabled": False}})
    manifest = StageManifest(stage=stage, config=config)

    result = handler(config, manifest)

    payload = json.loads(result.manifest_path.read_text())
    assert result.status == "skipped"
    assert result.skip_reason == StageSkipReason.DISABLED.value
    assert payload["skip_reason"] == StageSkipReason.DISABLED.value


def test_direct_tokenize_data_stage_runs_when_enabled(tmp_path, monkeypatch, write_token_cache):
    output = tmp_path / "dataset_cache" / "train.tokens"
    config = _config(
        tmp_path,
        dataset_path="dataset",
        convert={"teacher_dir": str(tmp_path / "ckpts" / "teacher")},
        tokenize_data={
            "enabled": True,
            "caches": [
                {
                    "output": str(output),
                    "split": "train",
                    "num_samples": 1,
                    "seq_length": 8,
                    "shuffle_seed": 1,
                }
            ],
        },
    )

    def build_cache(command, *, check):
        assert check
        configured_cache = config["tokenize_data"]["caches"][0]
        expected_options = {
            "--dataset-path": config["dataset_path"],
            "--tokenizer-path": config["convert"]["teacher_dir"],
            "--output": configured_cache["output"],
            "--split": configured_cache["split"],
            "--num-samples": str(configured_cache["num_samples"]),
            "--seq-length": str(configured_cache["seq_length"]),
            "--shuffle-seed": str(configured_cache["shuffle_seed"]),
        }
        assert {
            option: command[command.index(option) + 1] for option in expected_options
        } == expected_options
        write_token_cache(config, configured_cache)

    monkeypatch.setattr(tokenize_data_module.subprocess, "run", build_cache)

    result = tokenize_data_module.tokenize_data_stage(config)
    _validate_worker_result(config, result)

    assert result.status == "success"
    assert result.skip_reason is None


def test_enabled_tokenize_data_stage_rejects_empty_success_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    config = _config(
        tmp_path,
        dataset_path="dataset",
        convert={"teacher_dir": str(tmp_path / "ckpts" / "teacher")},
        tokenize_data={"enabled": True, "caches": []},
    )
    manifest = StageManifest(stage="tokenize_data", config=config)
    manifest.complete(outputs={"caches": []})
    manifest_path = tmp_path / "manifests" / "tokenize_data.json"
    write_stage_manifest(manifest_path, manifest)
    assert manifest_path.exists()

    assert not stage_is_complete(config, "tokenize_data")
    assert not _completion_is_valid(config, tmp_path / "config.yaml", "tokenize_data")


@pytest.mark.parametrize(
    ("stage", "config_sections", "manifest_fields"),
    [
        ("convert", {}, {}),
        (
            "aiperf",
            {"aiperf": {"enabled": False}},
            {"status": "skipped", "skip_reason": "disabled"},
        ),
    ],
    ids=("success", "skipped"),
)
@pytest.mark.parametrize("recorded_stage", [None, "other"], ids=("missing", "mismatched"))
def test_terminal_consumers_reject_wrong_stage_identity(
    tmp_path,
    monkeypatch,
    write_terminal_manifest,
    stage,
    config_sections,
    manifest_fields,
    recorded_stage,
):
    config = _config(tmp_path, **config_sections)
    write_terminal_manifest(tmp_path, stage, config=config, **manifest_fields)
    if stage == "convert":
        artifact = tmp_path / "ckpts" / "teacher" / "config.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}\n")
        monkeypatch.setattr(puzzletron_main, "check_marker", lambda *_args, **_kwargs: True)

    manifest_path = tmp_path / "manifests" / f"{stage}.json"
    payload = json.loads(manifest_path.read_text())
    if recorded_stage is None:
        payload.pop("stage")
    else:
        payload["stage"] = recorded_stage
    manifest_path.write_text(json.dumps(payload) + "\n")

    assert not stage_is_complete(config, stage)
    assert not _completion_is_valid(config, tmp_path / "config.yaml", stage)


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


def test_resume_rejects_skip_without_reason(tmp_path):
    config = _config(tmp_path, aiperf={"enabled": False})
    manifest_path = tmp_path / "manifests" / "aiperf.json"
    manifest = StageManifest(stage="aiperf", status="skipped", config=config)
    write_stage_manifest(manifest_path, manifest)

    assert not _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")


def test_resume_accepts_current_disabled_skip(tmp_path):
    config = _config(tmp_path, aiperf={"enabled": False})
    manifest_path = tmp_path / "manifests" / "aiperf.json"
    manifest = StageManifest(
        stage="aiperf",
        status="skipped",
        skip_reason="disabled",
        config=config,
    )
    write_stage_manifest(manifest_path, manifest)

    assert _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")


def test_resume_rejects_disabled_skip_after_stage_is_enabled(tmp_path):
    config = _config(tmp_path, aiperf={"enabled": True})
    manifest_path = tmp_path / "manifests" / "aiperf.json"
    manifest = StageManifest(
        stage="aiperf",
        status="skipped",
        skip_reason="disabled",
        config=config,
    )
    write_stage_manifest(manifest_path, manifest)

    assert not _completion_is_valid(config, tmp_path / "config.yaml", "aiperf")


def test_resume_rechecks_canonical_artifact_contract(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manifest = StageManifest(stage="width_importance", config=config)
    manifest.complete()
    write_stage_manifest(tmp_path / "manifests" / "width_importance.json", manifest)
    monkeypatch.setattr(puzzletron_main, "check_marker", lambda *_args, **_kwargs: True)

    assert not _completion_is_valid(config, tmp_path / "config.yaml", "width_importance")


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
