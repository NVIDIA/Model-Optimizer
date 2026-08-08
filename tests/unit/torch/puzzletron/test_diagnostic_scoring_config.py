# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import examples.puzzletron.run_axis_diagnostic_worker as axis_worker
from examples.puzzletron.run_axis_diagnostic_worker import (
    _axes,
    _hidden_widths,
    _validate_worker_topology,
    _worker_config,
)
from modelopt.torch.puzzletron.manifest import StageManifest
from modelopt.torch.puzzletron.stages import diagnostics
from modelopt.torch.puzzletron.stages.diagnostics import _scoring_cfg_for_method


def test_diagnostic_scoring_config_uses_explicit_checkpoint_roles(tmp_path: Path):
    teacher = tmp_path / "teacher"
    sorted_parent = tmp_path / "diagnostic_sorted_teacher"
    method_dir = tmp_path / "method"
    hydra_cfg = OmegaConf.create(
        {
            "teacher_dir": str(teacher),
            "scoring": {"automodel": {}},
        }
    )

    cfg = _scoring_cfg_for_method(
        hydra_cfg,
        method_dir=method_dir,
        scoring_output_dir=method_dir / "outputs",
        parallel=None,
        source_checkpoint_dir=sorted_parent,
        target_teacher_dir=teacher,
    )

    assert cfg.scoring.source_checkpoint_dir == str(sorted_parent)
    assert cfg.scoring.target_teacher_dir == str(teacher)


def test_diagnostic_sorted_parent_runs_the_distributed_sort_on_every_rank(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(diagnostics.dist, "is_master", lambda: False)
    monkeypatch.setattr(diagnostics.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(
        diagnostics,
        "build_sorted_teacher",
        lambda *args, **kwargs: calls.append("sort"),
    )
    monkeypatch.setattr(
        diagnostics,
        "_write_transformed_activation_logs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("non-master wrote logs")),
    )

    diagnostics._build_diagnostic_sorted_parent(
        teacher_dir=tmp_path / "teacher",
        activations_log_dir=tmp_path / "activations",
        transformed_log_dir=tmp_path / "transformed",
        sorted_dir=tmp_path / "sorted",
        descriptor=object(),
        method="activation",
        seed=1,
        selected_passes=("ffn",),
        axis="ffn_intermediate",
        layer_idx=0,
        embedding_widths=(),
    )

    assert calls == ["barrier", "sort", "barrier"]


def test_sort_equivalence_keeps_production_and_reverse_control_tolerances_separate():
    decision = diagnostics._sort_equivalence_decision(
        delta=1.4e-4,
        reverse_delta=1.54e-2,
        tolerance=1.0e-3,
        reverse_tolerance=2.0e-2,
    )

    assert decision == {
        "sorted_passed": True,
        "reverse_passed": True,
        "passed": True,
    }
    assert not diagnostics._sort_equivalence_decision(
        delta=1.4e-4,
        reverse_delta=1.54e-2,
        tolerance=1.0e-3,
        reverse_tolerance=1.0e-3,
    )["passed"]


def test_sort_equivalence_summary_records_blocking_drift(tmp_path: Path):
    scoring_output_dir = tmp_path / "scoring"
    scoring_output_dir.mkdir()
    (scoring_output_dir / "teacher.json").write_text(
        json.dumps({"lm_loss": {"avg": 1.0}}), encoding="utf-8"
    )
    (scoring_output_dir / "sliced_teacher.json").write_text(
        json.dumps({"lm_loss": {"avg": 1.25}}), encoding="utf-8"
    )
    summary_path = tmp_path / "summary.json"

    diagnostics._write_sort_equivalence_summary(
        teacher_dir=tmp_path / "teacher",
        sorted_dir=tmp_path / "sorted",
        reverse_dir=tmp_path / "reverse",
        scoring_output_dir=scoring_output_dir,
        reverse_output_dir=None,
        summary_path=summary_path,
        table_path=tmp_path / "table.md",
        metric="lm_loss",
        include_reverse=False,
        tolerance=0.01,
        reverse_tolerance=0.01,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert summary["delta"] == 0.25
    assert summary["findings"][0]["stage"] == "sort_sanity"


def test_sort_equivalence_uses_master_failure_verdict_on_every_rank(
    monkeypatch, tmp_path: Path
):
    config = {"experiment": {"dir": str(tmp_path)}}
    manifest = StageManifest(stage="sort_sanity", config=config)
    summary_path = tmp_path / "artifacts" / "sort_sanity" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    (tmp_path / "manifests").mkdir()
    finding = {
        "stage": "sort_sanity",
        "message": "sorted teacher drift too large",
        "severity": "error",
    }
    summary_path.write_text(
        json.dumps(
            {
                "passed": False,
                "metric": "lm_loss",
                "delta": 0.25,
                "reverse_delta": 0.0,
                "findings": [finding],
            }
        ),
        encoding="utf-8",
    )
    barriers = []
    monkeypatch.setattr(diagnostics.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(diagnostics.dist, "is_master", lambda: False)
    monkeypatch.setattr(diagnostics.dist, "barrier", lambda: barriers.append("barrier"))
    monkeypatch.setattr(
        diagnostics,
        "_write_sort_equivalence_summary",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("non-master wrote summary")),
    )

    result = diagnostics._finalize_sort_equivalence_stage(
        config,
        manifest,
        teacher_dir=tmp_path / "teacher",
        sorted_dir=tmp_path / "sorted",
        reverse_dir=tmp_path / "reverse",
        scoring_output_dir=tmp_path / "scoring",
        reverse_output_dir=None,
        summary_path=summary_path,
        table_path=summary_path.with_name("table.md"),
        metric="lm_loss",
        include_reverse=False,
        tolerance=0.01,
        reverse_tolerance=0.01,
    )

    assert result.status == "failed"
    assert barriers == ["barrier", "barrier"]
    assert manifest.outputs["verdict"] == "failed"
    assert manifest.outputs["delta"] == 0.25
    assert manifest.outputs["findings"] == [finding]


def test_sort_equivalence_rejects_finalization_after_distributed_cleanup(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(diagnostics.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="active process group"):
        diagnostics._finalize_sort_equivalence_stage(
            {"experiment": {"dir": str(tmp_path)}},
            StageManifest(stage="sort_sanity"),
            teacher_dir=tmp_path / "teacher",
            sorted_dir=tmp_path / "sorted",
            reverse_dir=tmp_path / "reverse",
            scoring_output_dir=tmp_path / "scoring",
            reverse_output_dir=None,
            summary_path=tmp_path / "summary.json",
            table_path=tmp_path / "table.md",
            metric="lm_loss",
            include_reverse=False,
            tolerance=0.01,
            reverse_tolerance=0.01,
        )


def test_axis_worker_preserves_requested_layers_and_targets_per_axis(tmp_path: Path):
    parallel = {
        "tp": 1,
        "cp": 1,
        "pp": 2,
        "ep": 4,
        "dp_shard": 4,
        "dp_replicate": 1,
        "pipeline_schedule": "1f1b",
    }
    config = {
        "experiment": {"dir": str(tmp_path / "run")},
        "clean_config_root": str(tmp_path / "clean"),
        "width_sanity": {
            "layer_count": 3,
            "target_count_per_axis": 2,
            "automodel": {"parallel": parallel},
        },
    }

    worker = _worker_config(config, "kv_groups", tmp_path / "production.yaml")

    assert worker["width_sanity"]["layer_count"] == 3
    assert worker["width_sanity"]["target_count_per_axis"] == 2
    assert worker["width_sanity"]["automodel"]["parallel"] == parallel
    assert not any(
        override.startswith("++width_sanity.automodel.parallel.")
        for override in worker.get("_runtime", {}).get("overrides", [])
    )


def test_axis_worker_excludes_non_sortable_axes_from_width_diagnostics():
    config = {
        "search_space": {
            "axes": {
                "moe_experts": {"enabled": True},
                "moe_top_k": {"enabled": True},
            }
        },
        "width_sanity": {"non_sortable_axes": ["moe_top_k"]},
    }

    assert _axes(config) == ["moe_experts", "hidden_width"]


def test_axis_worker_uses_seven_eighths_and_three_quarters_hidden_widths(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        axis_worker,
        "load_model_config",
        lambda *args, **kwargs: SimpleNamespace(hidden_size=2688),
    )
    config = {
        "teacher_dir": str(tmp_path / "teacher"),
        "embedding_pruning": {"alignment": 128},
    }

    assert _hidden_widths(config) == [2304, 1920]


def test_axis_worker_accepts_pp2_ep2_overlay_on_four_ranks(monkeypatch, tmp_path: Path):
    parallel = {
        "tp": 1,
        "cp": 1,
        "pp": 2,
        "ep": 2,
        "dp_shard": 2,
        "dp_replicate": 1,
    }
    config = {"width_sanity": {"automodel": {"parallel": parallel}}}
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(
        axis_worker,
        "load_runtime_hydra_config",
        lambda _config: OmegaConf.create(
            {"width_sanity": {"automodel": {"parallel": parallel}}}
        ),
    )

    _validate_worker_topology(config, "kv_groups")


def test_axis_worker_accepts_the_stage_owned_super_mesh(monkeypatch) -> None:
    parallel = {
        "tp": 1,
        "cp": 1,
        "pp": 2,
        "ep": 4,
        "dp_shard": 4,
        "dp_replicate": 2,
        "pipeline_schedule": "1f1b",
    }
    config = {"width_sanity": {"automodel": {"parallel": parallel}}}
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setattr(
        axis_worker,
        "load_runtime_hydra_config",
        lambda _config: OmegaConf.create(
            {"width_sanity": {"automodel": {"parallel": parallel}}}
        ),
    )

    _validate_worker_topology(config, "moe_experts")
