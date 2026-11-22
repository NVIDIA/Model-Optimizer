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
from types import SimpleNamespace

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)

_SCRIPT_PATH = Path(__file__).parents[5] / "examples/vllm_serve/calibrate_mask_reuse.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_mask_reuse_cli", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
calibrate_mask_reuse_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate_mask_reuse_cli)

_TOPOLOGY_SPEC = importlib.util.spec_from_file_location(
    "calibrate_mask_reuse_topology_cli", _SCRIPT_PATH.with_name("calibrate_mask_reuse_topology.py")
)
assert _TOPOLOGY_SPEC is not None and _TOPOLOGY_SPEC.loader is not None
calibrate_mask_reuse_topology_cli = importlib.util.module_from_spec(_TOPOLOGY_SPEC)
_TOPOLOGY_SPEC.loader.exec_module(calibrate_mask_reuse_topology_cli)


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _base_args(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "config.json").write_text(
        '{"num_hidden_layers":1,"num_attention_heads":2}\n', encoding="utf-8"
    )
    (checkpoint_root / "model.safetensors").write_bytes(b"weights")
    checkpoint = create_checkpoint_manifest(
        checkpoint_root, model="test-model", manifest_path=tmp_path / "checkpoint-external.json"
    )

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
                "capture_manifest_schema_version": 4,
                "capture_protocol": "modelopt_vllm_mask_reuse_target_sparsity_v4",
                "model": "test-model",
                "checkpoint_manifest_sha256": checkpoint.sha256,
                "checkpoint_manifest_path": str(checkpoint.manifest_path),
                "checkpoint_file_count": checkpoint.file_count,
                "checkpoint_total_size_bytes": checkpoint.total_size_bytes,
                "plan": "test_stride2",
                "fa4_source": "/source",
                "fa4_source_commit": "a" * 40,
                "fa4_source_git_tree": "b" * 40,
                "fa4_source_git_archive_sha256": "c" * 64,
                "fa4_source_manifest_path": "/source-manifest.json",
                "fa4_source_manifest_sha256": "d" * 64,
                "fa4_source_directory_count": 4,
                "fa4_source_file_count": 2,
                "fa4_source_total_size_bytes": 42,
                "engine_kwargs": {"tensor_parallel_size": 2},
                "dense_shadow_validation_requested": True,
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
        "--diagnostic-final-chunk",
        "--checkpoint",
        str(checkpoint_root),
        "--checkpoint-manifest",
        str(checkpoint.manifest_path),
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
        "--reuse-dropped-mass-report-threshold",
        "0.03",
        "--target-bmm1-skip-ratio",
        "0.10",
        "--output-policy",
        str(policy),
        "--output-report",
        str(report),
    ]
    return args, policy, report, captures, vanilla, artifacts


@pytest.mark.parametrize(("reference", "page"), [(False, 16), (True, 16), (True, 4160)])
def test_main_verifies_artifacts_and_atomically_writes_candidate(
    tmp_path, monkeypatch, capsys, reference, page
):
    args, policy_path, report_path, captures, vanilla, artifacts = _base_args(tmp_path)
    if reference:
        manifest_path = Path(args[args.index("--capture-manifest") + 1])
        manifest = json.loads(manifest_path.read_text())
        manifest["capture_manifest_schema_version"] = 6
        if page != 16:
            manifest["capture_manifest_schema_version"] = 8
            manifest["observed_kv_page_tokens"] = page
        manifest["capture_protocol"] = "modelopt_vllm_mask_reuse_reference_target_sparsity_v1"
        manifest["provider_kind"] = "triton-reference"
        manifest["evidence_level"] = "accuracy-reference-only"
        manifest["dense_shadow_validation_requested"] = False
        for field in tuple(manifest):
            if field.startswith("fa4_source"):
                manifest[field.replace("fa4_source", "provider_source", 1)] = manifest.pop(field)
        manifest_path.write_bytes(_canonical(manifest))
    source = [SimpleNamespace(global_num_heads=2)]
    monkeypatch.setattr(
        calibrate_mask_reuse_cli, "load_compact_mask_reuse_captures", lambda path: source
    )
    captured = {}
    artifact = {
        "version": 3,
        "promotion_status": "candidate_only",
        "deployment_geometry_validated": False,
        "provenance": {"input_capture_count": 1, "candidate_cell_count": 4},
        "calibration_report": {
            "promotion": {"eligible": False},
            "by_bucket": [{"bmm1_skip_objective": {"target_met": True}}],
        },
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
    assert report_path.read_bytes() == _canonical(
        {
            **artifact["calibration_report"],
            "candidate_policy_sha256": sha256(expected_policy).hexdigest(),
        }
    )
    assert captured["source"] is source
    assert captured["evidence"] == {
        "calibration_plan_sha256": sha256(artifacts["calibration_plan"].read_bytes()).hexdigest(),
        "family_registry_sha256": sha256(artifacts["family_registry"].read_bytes()).hexdigest(),
        "grouped_fit_sha256": sha256(artifacts["grouped_fit"].read_bytes()).hexdigest(),
        "outer_report_sha256": sha256(artifacts["outer_report"].read_bytes()).hexdigest(),
        "vanilla_fit_sha256": sha256(vanilla.read_bytes()).hexdigest(),
        "reuse_bundle_sha256": sha256(captures.read_bytes()).hexdigest(),
    }
    assert captured["reuse_dropped_mass_report_threshold"] == 0.03
    assert captured["target_bmm1_skip_ratio"] == 0.10
    assert captured["source_provenance"]["capture_provider"]["provider_kind"] == (
        "triton-reference" if reference else "fa4"
    )
    if page != 16:
        assert captured["source_provenance"]["capture_provider"]["observed_kv_page_tokens"] == page
    assert "MASK_REUSE_FA4_CANDIDATE_SHA256=" in capsys.readouterr().out


@pytest.mark.parametrize(("reference", "page"), [(False, 16), (True, 16), (True, 4160)])
def test_topology_manifest_preserves_provider_and_rejects_mixed_evidence(tmp_path, reference, page):
    args, *_ = _base_args(tmp_path)
    path = Path(args[args.index("--capture-manifest") + 1])
    raw = json.loads(path.read_text())
    raw["capture_manifest_schema_version"] = 7 if reference else 5
    if page != 16:
        raw["capture_manifest_schema_version"] = 9
        raw["observed_kv_page_tokens"] = page
    raw["capture_protocol"] = (
        "modelopt_vllm_mask_reuse_reference_topology_discovery_v1"
        if reference
        else "modelopt_vllm_mask_reuse_topology_discovery_v1"
    )
    raw["capture_mode"] = "topology_discovery"
    raw["max_reuse_span"] = 1
    raw["plan"] = "test_topology_discovery"
    raw["topology_discovery_capture_file_sha256"] = raw.pop("compact_capture_file_sha256")
    fit = {"prefill": {"a": 1, "b": 1}}
    raw["vanilla_fit_sha256"] = calibrate_mask_reuse_topology_cli.canonical_json_sha256(fit)
    if reference:
        for field in tuple(raw):
            if field.startswith("fa4_source"):
                raw[field.replace("fa4_source", "provider_source", 1)] = raw.pop(field)
        raw["provider_kind"] = "triton-reference"
        raw["evidence_level"] = "accuracy-reference-only"
    kwargs = {
        "checkpoint_sha256": raw["checkpoint_manifest_sha256"],
        "model": raw["model"],
        "capture_sha256": raw["topology_discovery_capture_file_sha256"],
        "vanilla_sha256": raw["vanilla_config_file_sha256"],
        "prompt_sha256": raw["prompt_plan_file_sha256"],
        "fit": fit,
    }
    calibrate_mask_reuse_topology_cli._validate_manifest(raw, **kwargs)
    raw["provider_kind"] = "fa4" if reference else "triton-reference"
    with pytest.raises(ValueError, match=r"provider_kind|extra="):
        calibrate_mask_reuse_topology_cli._validate_manifest(raw, **kwargs)


def test_vanilla_mutation_after_semantic_snapshot_cannot_publish(tmp_path, monkeypatch, capsys):
    args, policy_path, report_path, _, vanilla, _ = _base_args(tmp_path)
    original_payload = vanilla.read_bytes()
    real_evidence_artifacts = calibrate_mask_reuse_cli._evidence_artifacts

    def mutate_after_snapshot(namespace, *, vanilla_fit_sha256, checkpoint_sha256):
        vanilla.write_text('{"vanilla":false}\n', encoding="utf-8")
        return real_evidence_artifacts(
            namespace, vanilla_fit_sha256=vanilla_fit_sha256, checkpoint_sha256=checkpoint_sha256
        )

    monkeypatch.setattr(calibrate_mask_reuse_cli, "_evidence_artifacts", mutate_after_snapshot)
    monkeypatch.setattr(
        calibrate_mask_reuse_cli,
        "load_compact_mask_reuse_captures",
        lambda path: [SimpleNamespace(global_num_heads=2)],
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


def test_discovery_artifact_refits_through_real_cli(tmp_path):
    from _test_utils.torch.mask_reuse import FIT, captures, checkpoint, solve

    case = tmp_path / "real"
    case.mkdir()
    cp = checkpoint(case)
    rows = captures(cp, seed=1)
    topology = solve(case, cp, rows, 0.1, "highs")
    inputs = tmp_path / "cli"
    inputs.mkdir()
    args, policy, report, capture_path, vanilla, _ = _base_args(inputs)

    def argument(name, value):
        args[args.index(name) + 1] = str(value)

    argument("--checkpoint", cp.checkpoint_root)
    argument("--checkpoint-manifest", cp.manifest_path)
    topology_path = Path(args[args.index("--topology") + 1])
    topology_path.write_bytes(_canonical(topology))
    args.extend(["--topology-sha256", sha256(topology_path.read_bytes()).hexdigest()])
    vanilla.write_bytes(_canonical(FIT))
    compact = []
    for row in rows:
        compact.append(  # noqa: PERF401 - expand the fixture conversion for readability
            {
                "compact_capture_schema_version": 1,
                **{
                    k: row[k]
                    for k in ("invocation", "geometry", "global_num_heads", "eligible_tiles")
                },
                "anchor_stats_by_layer": {
                    str(a): row["anchor_stats_by_layer"][str(a)] for a in topology["anchors"]
                },
                "consumer_layers": {
                    layer: {
                        "anchor_layer": a,
                        "dropped_mass": row["consumer_candidates_by_layer"][layer][str(a)][
                            "dropped_mass"
                        ],
                    }
                    for layer, a in topology["nearest"].items()
                    if int(layer) != a
                },
            }
        )
    capture_path.write_bytes(b"".join(_canonical(row) for row in compact))
    manifest_path = Path(args[args.index("--capture-manifest") + 1])
    manifest = json.loads(manifest_path.read_bytes())
    cells = len(compact[0]["consumer_layers"]) * 4
    manifest.update(
        checkpoint_manifest_sha256=cp.sha256,
        checkpoint_manifest_path=str(cp.manifest_path),
        compact_capture_file_sha256=sha256(capture_path.read_bytes()).hexdigest(),
        vanilla_config_file_sha256=sha256(vanilla.read_bytes()).hexdigest(),
        capture_count=len(compact),
        candidate_cell_count=cells * len(compact),
        captures=[{"candidate_cell_count": cells} for _ in compact],
    )
    manifest_path.write_bytes(_canonical(manifest))
    assert calibrate_mask_reuse_cli.main(args) == 0
    exported = json.loads(policy.read_bytes())
    assert exported["anchors"] == topology["anchors"]
    assert exported["nearest"] == topology["nearest"]
    assert exported["calibration_report"]["overall"]["bmm1_skip_objective"][
        "all_context_buckets_met"
    ]
    assert report.exists()


@pytest.mark.parametrize("unresolved", [False, True])
def test_failed_selection_publishes_report_without_policy(tmp_path, monkeypatch, unresolved):
    args, policy, report, *_ = _base_args(tmp_path)
    monkeypatch.setattr(
        calibrate_mask_reuse_cli,
        "load_compact_mask_reuse_captures",
        lambda path: [SimpleNamespace(global_num_heads=2)],
    )

    def select(*args, **kwargs):
        if unresolved:
            raise calibrate_mask_reuse_cli.MaskReuseSolverError(
                "time limit", [{"status": "Not Solved"}]
            )
        return {
            "promotion_status": "candidate_only",
            "deployment_geometry_validated": False,
            "provenance": {"input_capture_count": 1, "candidate_cell_count": 4},
            "calibration_report": {
                "by_bucket": [{"bmm1_skip_objective": {"target_met": False, "achieved": 0.08}}]
            },
        }

    monkeypatch.setattr(calibrate_mask_reuse_cli, "calibrate_compact_mask_reuse_policy", select)
    assert calibrate_mask_reuse_cli.main(args) == (3 if unresolved else 2)
    assert report.exists() and not policy.exists()


def test_final_chunk_and_empty_evidence_refused_before_solver(tmp_path, capsys):
    args, policy, report, _, _, artifacts = _base_args(tmp_path)
    args.remove("--diagnostic-final-chunk")
    with pytest.raises(SystemExit):
        calibrate_mask_reuse_cli.main(args)
    assert "whole-prefill" in capsys.readouterr().err
    args.append("--diagnostic-final-chunk")
    artifacts["outer_report"].write_bytes(b"")
    with pytest.raises(SystemExit):
        calibrate_mask_reuse_cli.main(args)
    assert "outer_report" in capsys.readouterr().err
    assert not policy.exists() and not report.exists()


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
