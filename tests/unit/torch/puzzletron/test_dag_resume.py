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

import hashlib
import json
from pathlib import Path

import pytest

from examples.puzzletron.acceptance_resume import (
    build_payload,
    check_marker,
    check_marker_details,
    marker_path,
    source_identity,
    write_marker,
)
from examples.puzzletron.main import _completion_is_valid, _resume_kwargs
from modelopt.torch.puzzletron.manifest import (
    StageManifest,
    semantic_stage_config,
    write_stage_manifest,
)
from modelopt.torch.puzzletron.stages.graph import STAGE_REGISTRY


_PRE_V3_COMPLETION_MARKER = """{
  "config": "/historical/campaign/config.yaml",
  "depth": null,
  "mode": "convert",
  "required_artifacts": {},
  "source_identity": "pre-v3-static-source-identity",
  "upstream_identities": {},
  "version": 2,
  "width": null
}
"""


def _config(root: Path, **sections: dict) -> dict:
    return {
        "puzzle_dir": str(root),
        "convert": {},
        "prepare_data": {"enabled": True},
        "activation": {},
        "sort": {},
        "build_library": {},
        "scoring": {},
        "mip": {},
        **sections,
    }


def _write_completion(root: Path, stage: str, identity: str) -> Path:
    config_path = root / "completion-test-config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("static: true\n")
    stage_config = {"fixture_identity": identity}
    _write_stage_output(root, stage, stage_config)
    payload = build_payload(
        root=root,
        config=config_path,
        mode=stage,
        width=None,
        depth=None,
        required_patterns=(f"manifests/{stage}.json",),
        stage_config=stage_config,
        source_roots=(),
    )
    return write_marker(root, stage, payload)


def _write_stage_output(root: Path, stage: str, stage_config: dict) -> None:
    manifest = StageManifest(stage=stage, config={stage: stage_config})
    manifest.complete(outputs={"summary": "ok"})
    write_stage_manifest(root / "manifests" / f"{stage}.json", manifest)


def test_resume_records_every_selected_mip_parent(tmp_path: Path) -> None:
    config = _config(tmp_path, depth={"enabled": True}, vllm_stats={"enabled": False})

    kwargs = _resume_kwargs(config, tmp_path / "config.yaml", "mip")

    assert tuple(kwargs["upstream_markers"]) == ("depth", "scoring", "build_library")


def test_resume_selects_sort_or_bypass_for_library(tmp_path: Path) -> None:
    disabled = _config(
        tmp_path,
        bypass={"enabled": False, "use_nested_bypassed_checkpoint_for_scoring": True},
    )
    enabled = _config(
        tmp_path,
        bypass={"enabled": True, "use_nested_bypassed_checkpoint_for_scoring": True},
    )

    disabled_kwargs = _resume_kwargs(disabled, tmp_path / "config.yaml", "build_library")
    enabled_kwargs = _resume_kwargs(enabled, tmp_path / "config.yaml", "build_library")

    assert tuple(disabled_kwargs["upstream_markers"]) == ("sort",)
    assert tuple(enabled_kwargs["upstream_markers"]) == ("bypass",)


def test_resume_selects_inline_or_explicit_vllm_producer(tmp_path: Path) -> None:
    inline = _config(tmp_path, depth={"enabled": True}, vllm_stats={"enabled": False})
    explicit = _config(tmp_path, depth={"enabled": True}, vllm_stats={"enabled": True})

    inline_kwargs = _resume_kwargs(inline, tmp_path / "config.yaml", "mip")
    explicit_kwargs = _resume_kwargs(explicit, tmp_path / "config.yaml", "mip")

    assert tuple(inline_kwargs["upstream_markers"]) == ("depth", "scoring", "build_library")
    assert tuple(explicit_kwargs["upstream_markers"]) == ("depth", "scoring", "vllm_stats")


def test_explicit_vllm_stats_is_a_build_library_resume_parent(tmp_path: Path) -> None:
    inline = _config(tmp_path, vllm_stats={"enabled": False})
    explicit = _config(tmp_path, vllm_stats={"enabled": True})

    assert tuple(_resume_kwargs(inline, tmp_path / "config.yaml", "build_library")["upstream_markers"]) == (
        "sort",
    )
    assert tuple(_resume_kwargs(explicit, tmp_path / "config.yaml", "build_library")["upstream_markers"]) == (
        "sort",
        "vllm_stats",
    )


def test_vllm_completion_stales_when_runtime_aggregate_is_deleted(tmp_path: Path) -> None:
    config = _config(tmp_path, vllm_stats={"enabled": True})
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    _write_completion(tmp_path, "convert", "convert-v1")
    _write_stage_output(tmp_path, "vllm_stats", config["vllm_stats"])
    stats_path = tmp_path / "subblock_stats.json"
    stats_path.write_text("[{}]\n")
    kwargs = _resume_kwargs(config, config_path, "vllm_stats")
    write_marker(tmp_path, "vllm_stats", build_payload(**kwargs))

    assert _completion_is_valid(config, config_path, "vllm_stats")
    stats_path.unlink()
    assert not _completion_is_valid(config, config_path, "vllm_stats")


def test_vllm_completion_uses_configured_runtime_aggregate_filename(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        vllm_stats={"enabled": True},
        calc_subblock_stats={"subblock_stats_filename": "runtime/custom_stats.json"},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    _write_completion(tmp_path, "convert", "convert-v1")
    _write_stage_output(tmp_path, "vllm_stats", config["vllm_stats"])
    stats_path = tmp_path / "runtime/custom_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text("[{}]\n")
    kwargs = _resume_kwargs(config, config_path, "vllm_stats")
    write_marker(tmp_path, "vllm_stats", build_payload(**kwargs))

    assert _completion_is_valid(config, config_path, "vllm_stats")
    stats_path.unlink()
    assert not _completion_is_valid(config, config_path, "vllm_stats")


def test_build_library_completion_uses_selected_custom_vllm_aggregate(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        vllm_stats={"enabled": True},
        calc_subblock_stats={"subblock_stats_filename": "runtime/custom_stats.json"},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    _write_completion(tmp_path, "convert", "convert-v1")
    _write_completion(tmp_path, "sort", "sort-v1")
    _write_stage_output(tmp_path, "vllm_stats", config["vllm_stats"])
    stats_path = tmp_path / "runtime/custom_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text("[{}]\n")
    vllm_kwargs = _resume_kwargs(config, config_path, "vllm_stats")
    write_marker(tmp_path, "vllm_stats", build_payload(**vllm_kwargs))
    _write_stage_output(tmp_path, "build_library", config["build_library"])
    (tmp_path / "replacement_library.json").write_text("{}\n")
    (tmp_path / "candidate_library.json").write_text("{}\n")
    build_kwargs = _resume_kwargs(config, config_path, "build_library")
    write_marker(tmp_path, "build_library", build_payload(**build_kwargs))

    assert _completion_is_valid(config, config_path, "build_library")
    stats_path.unlink()
    assert not _completion_is_valid(config, config_path, "build_library")


def test_resume_keeps_disabled_and_skipped_fixed_parents(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        activation_diagnostic={"enabled": False},
        bypass_overfit={"enabled": True},
    )
    manifest = StageManifest(stage="activation_diagnostic", status="skipped")
    write_stage_manifest(tmp_path / "manifests" / "activation_diagnostic.json", manifest)

    kwargs = _resume_kwargs(config, tmp_path / "config.yaml", "bypass_overfit")

    assert tuple(kwargs["upstream_markers"]) == ("activation_diagnostic",)


def test_every_registry_stage_projects_its_own_section_but_not_report_settings() -> None:
    config = {
        stage: {"semantic_value": stage}
        for stage in STAGE_REGISTRY
    }
    config["report"] = {"theme": "light"}

    for stage in STAGE_REGISTRY:
        baseline = semantic_stage_config(config, stage)
        changed_stage = {**config, stage: {"semantic_value": f"changed-{stage}"}}
        changed_report = {**config, "report": {"theme": "dark"}}

        assert semantic_stage_config(changed_stage, stage) != baseline
        assert semantic_stage_config(changed_report, stage) == baseline


@pytest.mark.parametrize(
    "section",
    ("prepare_data", "convert", "dataset_path", "model", "data", "dataset"),
)
def test_prepare_data_identity_includes_every_tokenizer_and_data_input(
    section: str,
) -> None:
    config = {
        "prepare_data": {"caches": [{"split": "train"}]},
        "convert": {"teacher_dir": "/models/teacher-a"},
        "dataset_path": "/datasets/source-a",
        "model": {"source": "/models/source-a"},
        "data": {"content_field": "messages"},
        "dataset": {"split": "train"},
        "report": {"theme": "light"},
    }
    baseline = StageManifest(stage="prepare_data", config=config).semantic_config_identity
    changed = {**config, section: {"changed": True}}
    report_changed = {**config, "report": {"theme": "dark"}}

    assert StageManifest(stage="prepare_data", config=changed).semantic_config_identity != baseline
    assert (
        StageManifest(stage="prepare_data", config=report_changed).semantic_config_identity
        == baseline
    )


@pytest.mark.parametrize(
    ("stage", "handler_section"),
    (
        ("activation", "pruning"),
        ("build_library", "build_replacement_library"),
        ("build_library", "calc_subblock_stats"),
        ("build_library", "library"),
        ("scoring", "replacement_scoring"),
    ),
)
def test_handler_section_changes_stale_consumer_but_report_changes_do_not(
    tmp_path: Path,
    stage: str,
    handler_section: str,
) -> None:
    root = tmp_path / "campaign"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    config = _config(
        root,
        **{
            handler_section: {"semantic_value": "original"},
            "report": {"theme": "light"},
        },
    )
    _write_stage_output(root, stage, {})
    kwargs = {
        "root": root,
        "config": config_path,
        "mode": stage,
        "width": None,
        "depth": None,
        "required_patterns": (f"manifests/{stage}.json",),
        "stage_config": semantic_stage_config(config, stage),
        "source_roots": (),
    }
    marker = write_marker(root, stage, build_payload(**kwargs))

    report_changed = {**config, "report": {"theme": "dark"}}
    assert check_marker(
        marker,
        **{**kwargs, "stage_config": semantic_stage_config(report_changed, stage)},
    )

    handler_changed = {**config, handler_section: {"semantic_value": "changed"}}
    result = check_marker_details(
        marker,
        **{**kwargs, "stage_config": semantic_stage_config(handler_changed, stage)},
    )
    assert not result.valid
    assert "changed relevant stage config" in result.stale_reasons


def test_report_source_change_does_not_stale_expensive_stage(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    report_source = source_root / "report.py"
    report_source.write_text("FIRST = True\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("activation:\n  samples: 8\n")
    _write_stage_output(root, "activation", {"samples": 8})
    kwargs = {
        "root": root,
        "config": config_path,
        "mode": "activation",
        "width": None,
        "depth": None,
        "required_patterns": ("manifests/activation.json",),
        "stage_config": {"samples": 8},
        "source_roots": (source_root,),
    }
    marker = write_marker(root, "activation", build_payload(**kwargs))

    report_source.write_text("FIRST = False\n")

    assert check_marker(marker, **kwargs)
    assert check_marker_details(marker, **kwargs).validation_mode == "semantic-v3"


def test_relevant_config_and_upstream_changes_have_stale_reasons(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("scoring:\n  samples: 8\n")
    _write_stage_output(root, "scoring", {"samples": 8})
    upstream = _write_completion(root, "build_library", "build-v1")
    kwargs = {
        "root": root,
        "config": config_path,
        "mode": "scoring",
        "width": None,
        "depth": None,
        "required_patterns": ("manifests/scoring.json",),
        "upstream_markers": {"build_library": upstream},
        "stage_config": {"samples": 8},
        "source_roots": (),
    }
    marker = write_marker(root, "scoring", build_payload(**kwargs))

    changed_config = {**kwargs, "stage_config": {"samples": 16}}
    assert check_marker_details(marker, **changed_config).stale_reasons == (
        "changed relevant stage config",
    )

    _write_completion(root, "build_library", "build-v2")
    assert check_marker_details(marker, **kwargs).stale_reasons == (
        "changed selected upstream identity: build_library",
    )


@pytest.mark.parametrize("mutated_parent_evidence", ("artifact", "manifest"))
def test_child_revalidates_current_parent_evidence_without_marker_rewrite(
    tmp_path: Path,
    mutated_parent_evidence: str,
) -> None:
    root = tmp_path / "campaign"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    parent_artifact = root / "artifacts" / "build_library.json"
    parent_artifact.parent.mkdir(parents=True)
    parent_artifact.write_text('{"value": 1}\n')
    _write_stage_output(root, "build_library", {})
    parent_kwargs = {
        "root": root,
        "config": config_path,
        "mode": "build_library",
        "width": None,
        "depth": None,
        "required_patterns": (
            "artifacts/build_library.json",
            "manifests/build_library.json",
        ),
        "stage_config": {},
        "source_roots": (),
    }
    parent_marker = write_marker(root, "build_library", build_payload(**parent_kwargs))
    _write_stage_output(root, "scoring", {})
    child_kwargs = {
        "root": root,
        "config": config_path,
        "mode": "scoring",
        "width": None,
        "depth": None,
        "required_patterns": ("manifests/scoring.json",),
        "upstream_markers": {"build_library": parent_marker},
        "stage_config": {},
        "source_roots": (),
    }
    child_marker = write_marker(root, "scoring", build_payload(**child_kwargs))

    if mutated_parent_evidence == "artifact":
        parent_artifact.write_text('{"value": 2}\n')
    else:
        parent_manifest = root / "manifests" / "build_library.json"
        payload = json.loads(parent_manifest.read_text())
        payload["semantic_identity"] = "mutated-without-marker-rewrite"
        parent_manifest.write_text(json.dumps(payload))

    result = check_marker_details(child_marker, **child_kwargs)

    assert not result.valid
    assert result.stale_reasons == (
        "changed selected upstream identity: build_library",
    )


@pytest.mark.parametrize("parent_evidence", ("artifact", "manifest"))
@pytest.mark.parametrize("change", ("mutated", "deleted"))
def test_child_build_rejects_parent_changed_before_identity_collection(
    tmp_path: Path,
    parent_evidence: str,
    change: str,
) -> None:
    root = tmp_path / "campaign"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    parent_artifact = root / "artifacts" / "build_library.json"
    parent_artifact.parent.mkdir(parents=True)
    parent_artifact.write_text('{"value": 1}\n')
    _write_stage_output(root, "build_library", {})
    parent_manifest = root / "manifests" / "build_library.json"
    parent_marker = write_marker(
        root,
        "build_library",
        build_payload(
            root=root,
            config=config_path,
            mode="build_library",
            width=None,
            depth=None,
            required_patterns=(
                "artifacts/build_library.json",
                "manifests/build_library.json",
            ),
            stage_config={},
            source_roots=(),
        ),
    )
    changed_path = parent_artifact if parent_evidence == "artifact" else parent_manifest
    if change == "deleted":
        changed_path.unlink()
    elif parent_evidence == "artifact":
        changed_path.write_text('{"value": 2}\n')
    else:
        payload = json.loads(changed_path.read_text())
        payload["semantic_identity"] = "mutated-before-child-build"
        changed_path.write_text(json.dumps(payload))

    _write_stage_output(root, "scoring", {})
    with pytest.raises(ValueError, match="selected upstream unverifiable: build_library"):
        build_payload(
            root=root,
            config=config_path,
            mode="scoring",
            width=None,
            depth=None,
            required_patterns=("manifests/scoring.json",),
            upstream_markers={"build_library": parent_marker},
            stage_config={},
            source_roots=(),
        )


@pytest.mark.parametrize("parent_version", ("legacy-v2", "incomplete-v3"))
def test_unverifiable_parent_cannot_seed_or_validate_v3_child_after_evidence_mutation(
    tmp_path: Path,
    parent_version: str,
) -> None:
    root = tmp_path / "campaign"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("static: true\n")
    parent_artifact = root / "artifacts" / "convert.json"
    parent_artifact.parent.mkdir(parents=True)
    parent_artifact.write_text('{"value": 1}\n')
    if parent_version == "legacy-v2":
        parent_payload = {
            "version": 2,
            "mode": "convert",
            "source_identity": "historical",
            "required_artifacts": {"artifacts/convert.json": []},
            "upstream_identities": {},
        }
    else:
        parent_payload = {
            "version": 3,
            "mode": "convert",
            "completion_identity": "convert_incomplete",
        }
    parent_marker = write_marker(root, "convert", parent_payload)

    _write_stage_output(root, "prepare_data", {})
    child_kwargs = {
        "root": root,
        "config": config_path,
        "mode": "prepare_data",
        "width": None,
        "depth": None,
        "required_patterns": ("manifests/prepare_data.json",),
        "upstream_markers": {"convert": parent_marker},
        "stage_config": {},
        "source_roots": (),
    }
    with pytest.raises(ValueError, match="selected upstream unverifiable: convert"):
        build_payload(**child_kwargs)

    historical_child = write_marker(
        root,
        "prepare_data",
        build_payload(**{**child_kwargs, "upstream_markers": {}}),
    )
    parent_artifact.write_text('{"value": 2}\n')

    result = check_marker_details(historical_child, **child_kwargs)

    assert not result.valid
    assert result.stale_reasons == ("selected upstream unverifiable: convert",)


def test_missing_output_and_incompatible_manifest_identity_have_reasons(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("activation: {}\n")
    _write_stage_output(root, "activation", {})
    kwargs = {
        "root": root,
        "config": config_path,
        "mode": "activation",
        "width": None,
        "depth": None,
        "required_patterns": ("manifests/activation.json",),
        "stage_config": {},
        "source_roots": (),
    }
    marker = write_marker(root, "activation", build_payload(**kwargs))

    manifest_path = root / "manifests" / "activation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["semantic_identity"] = "stage_incompatible"
    manifest_path.write_text(json.dumps(manifest))
    assert check_marker_details(marker, **kwargs).stale_reasons == (
        "incompatible semantic identity",
    )

    manifest_path.unlink()
    assert check_marker_details(marker, **kwargs).stale_reasons == (
        "missing output: manifests/activation.json",
    )


def test_legacy_version_2_marker_remains_checkable(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    output = root / "artifact.json"
    output.parent.mkdir()
    output.write_text('{"ok": true}\n')
    config_path = tmp_path / "config.yaml"
    config_path.write_text("convert: {}\n")
    kwargs = {
        "root": root,
        "config": config_path,
        "mode": "convert",
        "width": None,
        "depth": None,
        "required_patterns": ("artifact.json",),
        "source_roots": (),
    }
    stat = output.stat()
    legacy = {
        "version": 2,
        "mode": "convert",
        "width": None,
        "depth": None,
        "source_identity": source_identity(config_path, source_roots=()),
        "config": str(config_path.resolve()),
        "required_artifacts": {
            "artifact.json": [
                {
                    "path": "artifact.json",
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            ]
        },
        "upstream_identities": {},
    }
    marker = write_marker(root, "convert", legacy)

    result = check_marker_details(marker, **kwargs)

    assert result.valid
    assert result.validation_mode == "legacy-v2"
    assert result.stale_reasons == ()


def test_static_pre_v3_marker_reports_implementation_source_staleness(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("convert: {}\n")
    marker = marker_path(tmp_path, "convert", None, None)
    marker.parent.mkdir(parents=True)
    marker.write_text(_PRE_V3_COMPLETION_MARKER)

    result = check_marker_details(
        marker,
        root=tmp_path,
        config=config_path,
        mode="convert",
        width=None,
        depth=None,
        source_roots=(),
    )

    assert not result.valid
    assert result.validation_mode == "legacy-v2"
    assert "changed implementation/source identity" in result.stale_reasons
