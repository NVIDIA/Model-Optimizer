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

"""Focused tests for fail-closed mask-reuse candidate publication."""

import importlib.util
import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)

_SCRIPT_PATH = Path(__file__).parents[5] / "examples/vllm_serve/calibrate_mask_reuse.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_mask_reuse_cli", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
calibrate_mask_reuse_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate_mask_reuse_cli)


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _base_args(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_root / "model.safetensors").write_bytes(b"weights")
    checkpoint = create_checkpoint_manifest(checkpoint_root, model="test-model")

    captures = tmp_path / "captures.jsonl"
    captures.write_text("unused\n", encoding="utf-8")
    vanilla = tmp_path / "vanilla.json"
    vanilla.write_text('{"vanilla":true}\n', encoding="utf-8")
    topology = tmp_path / "topology.json"
    topology.write_text('{"anchors":[0],"nearest":{"0":0}}\n', encoding="utf-8")
    artifacts = {}
    for name in ("calibration_plan", "family_registry", "grouped_fit", "outer_report"):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"artifact":"{name}"}}\n', encoding="utf-8")
        artifacts[name] = path
    capture_manifest = tmp_path / "capture-manifest.json"
    capture_manifest.write_bytes(
        _canonical(
            {
                "capture_manifest_schema_version": 2,
                "capture_protocol": "modelopt_vllm_mask_reuse_target_sparsity_v2",
                "model": "test-model",
                "checkpoint_manifest_sha256": checkpoint.sha256,
                "checkpoint_manifest_path": str(checkpoint.manifest_path),
                "checkpoint_file_count": checkpoint.file_count,
                "checkpoint_total_size_bytes": checkpoint.total_size_bytes,
                "plan": "test_stride2",
                "fa4_source": "/source",
                "fa4_source_commit": "a" * 40,
                "target_sparsity_hex": [(0.7).hex()],
                "vanilla_threshold_scale_factor": {"formula": "unused"},
                "vanilla_fit_sha256": sha256(b"normalized-fit").hexdigest(),
                "vanilla_config_file_sha256": sha256(vanilla.read_bytes()).hexdigest(),
                "prompt_plan_file_sha256": sha256(b"prompts").hexdigest(),
                "compact_capture_file_sha256": sha256(captures.read_bytes()).hexdigest(),
                "capture_count": 1,
                "candidate_cell_count": 4,
                "captures": [{"candidate_cell_count": 4}],
            }
        )
    )
    policy = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    args = [
        "--checkpoint",
        str(checkpoint_root),
        "--compact-captures",
        str(captures),
        "--capture-manifest",
        str(capture_manifest),
        "--vanilla-config",
        str(vanilla),
        "--topology",
        str(topology),
        "--calibration-plan",
        str(artifacts["calibration_plan"]),
        "--family-registry",
        str(artifacts["family_registry"]),
        "--grouped-fit",
        str(artifacts["grouped_fit"]),
        "--outer-report",
        str(artifacts["outer_report"]),
        "--max-anchor-dropped-mass",
        "0.02",
        "--max-reuse-dropped-mass",
        "0.03",
        "--max-reuse-selection-dropped-mass",
        "0.01",
        "--output-policy",
        str(policy),
        "--output-report",
        str(report),
    ]
    return args, policy, report, captures, vanilla, artifacts


def test_main_verifies_artifacts_and_atomically_writes_candidate(tmp_path, monkeypatch, capsys):
    args, policy_path, report_path, captures, vanilla, artifacts = _base_args(tmp_path)
    source = object()
    monkeypatch.setattr(
        calibrate_mask_reuse_cli, "load_compact_mask_reuse_captures", lambda path: source
    )
    captured = {}
    artifact = {
        "version": 3,
        "promotion_status": "candidate_only",
        "deployment_geometry_validated": False,
        "provenance": {"input_capture_count": 1, "candidate_cell_count": 4},
        "calibration_report": {"promotion": {"eligible": False}},
    }

    def fake_calibrate(compact_source, **kwargs):
        captured["source"] = compact_source
        captured.update(kwargs)
        return artifact

    monkeypatch.setattr(
        calibrate_mask_reuse_cli, "calibrate_compact_mask_reuse_policy", fake_calibrate
    )

    assert calibrate_mask_reuse_cli.main(args) == 0

    expected_policy = _canonical(artifact)
    assert policy_path.read_bytes() == expected_policy
    assert report_path.read_bytes() == _canonical(artifact["calibration_report"])
    assert captured["source"] is source
    assert captured["evidence"] == {
        "calibration_plan_sha256": sha256(artifacts["calibration_plan"].read_bytes()).hexdigest(),
        "family_registry_sha256": sha256(artifacts["family_registry"].read_bytes()).hexdigest(),
        "grouped_fit_sha256": sha256(artifacts["grouped_fit"].read_bytes()).hexdigest(),
        "outer_report_sha256": sha256(artifacts["outer_report"].read_bytes()).hexdigest(),
        "vanilla_fit_sha256": sha256(vanilla.read_bytes()).hexdigest(),
        "reuse_bundle_sha256": sha256(captures.read_bytes()).hexdigest(),
    }
    assert "MASK_REUSE_FA4_CANDIDATE_SHA256=" in capsys.readouterr().out


def test_vanilla_mutation_after_semantic_snapshot_cannot_publish(tmp_path, monkeypatch, capsys):
    args, policy_path, report_path, _, vanilla, _ = _base_args(tmp_path)
    original_payload = vanilla.read_bytes()
    real_evidence_artifacts = calibrate_mask_reuse_cli._evidence_artifacts

    def mutate_after_snapshot(namespace, *, vanilla_fit_sha256):
        vanilla.write_text('{"vanilla":false}\n', encoding="utf-8")
        return real_evidence_artifacts(namespace, vanilla_fit_sha256=vanilla_fit_sha256)

    monkeypatch.setattr(calibrate_mask_reuse_cli, "_evidence_artifacts", mutate_after_snapshot)
    monkeypatch.setattr(
        calibrate_mask_reuse_cli,
        "load_compact_mask_reuse_captures",
        lambda path: object(),
    )

    def fake_calibrate(compact_source, **kwargs):
        assert kwargs["vanilla_calibration"] == {"vanilla": True}
        assert kwargs["evidence"]["vanilla_fit_sha256"] == sha256(original_payload).hexdigest()
        return {
            "promotion_status": "candidate_only",
            "deployment_geometry_validated": False,
            "provenance": {"input_capture_count": 1, "candidate_cell_count": 4},
            "calibration_report": {"promotion": {"eligible": False}},
        }

    monkeypatch.setattr(
        calibrate_mask_reuse_cli,
        "calibrate_compact_mask_reuse_policy",
        fake_calibrate,
    )

    with pytest.raises(SystemExit, match="2"):
        calibrate_mask_reuse_cli.main(args)

    assert "vanilla_fit_sha256 artifact changed during calibration" in capsys.readouterr().err
    assert not policy_path.exists()
    assert not report_path.exists()


def test_main_rejects_capture_manifest_hash_mismatch(tmp_path, capsys):
    args, _, _, captures, _, _ = _base_args(tmp_path)
    captures.write_text("changed\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        calibrate_mask_reuse_cli.main(args)

    assert "compact_capture_file_sha256" in capsys.readouterr().err


def test_load_json_object_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"anchors":[0],"anchors":[1]}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key 'anchors'"):
        calibrate_mask_reuse_cli._load_json_object(path, label="topology")


def test_candidate_publication_rolls_back_report_when_policy_race_wins(tmp_path, monkeypatch):
    policy = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    real_link = os.link
    call_count = 0

    def racing_link(source, target, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            Path(target).write_bytes(b"racer")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(calibrate_mask_reuse_cli.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        calibrate_mask_reuse_cli._publish_candidate_outputs(policy, b"ours", report, b"report")

    assert policy.read_bytes() == b"racer"
    assert not report.exists()


def test_candidate_publication_refuses_existing_output(tmp_path):
    policy = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    policy.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        calibrate_mask_reuse_cli._publish_candidate_outputs(policy, b"ours", report, b"report")

    assert policy.read_bytes() == b"existing"
    assert not report.exists()
