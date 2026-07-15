# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

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
        recipe_path=None,
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
