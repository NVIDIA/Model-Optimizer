# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_profile_kd_overfit_worker_uses_canonical_sanity_stage_and_stage_mesh(
    monkeypatch, tmp_path: Path
):
    import examples.puzzletron.run_profile_distillation_overfit_worker as worker

    mesh = {
        "tp": 1,
        "cp": 1,
        "pp": 2,
        "ep": 4,
        "dp_shard": 4,
        "dp_replicate": 1,
        "sequence_parallel": False,
        "pipeline_schedule": "1f1b",
    }
    config = {
        "experiment": {"dir": str(tmp_path)},
        "parallel": {"tp": 1, "cp": 1, "pp": 2, "dp": 1, "ep": 4},
        "global_distillation_sanity": {
            "enabled": True,
            "profile_id": "old-profile",
            "packed_token_cache_path": str(tmp_path / "tokens"),
            "lr": 2.0e-4,
            "automodel": {"parallel": mesh},
        },
        "global_distillation": {"num_best_to_distill": 1},
    }
    manifest = tmp_path / "manifests" / "build_library.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"config": config}))
    captured = {}

    def fake_run_stage(actual_config, stage):
        captured["config"] = actual_config
        captured["stage"] = stage
        return SimpleNamespace(manifest_path=tmp_path / "manifests" / "global_kd.json")

    monkeypatch.setattr(worker, "run_stage", fake_run_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_profile_distillation_overfit_worker.py",
            "--puzzle-dir",
            str(tmp_path),
            "--profile-id",
            "runtime-075",
            "--solution-id",
            "best",
            "--registry-path",
            str(tmp_path / "selected_solutions.json"),
            "--sample-count",
            "16",
            "--sequence-length",
            "8192",
            "--max-steps",
            "128",
            "--local-batch-size",
            "1",
            "--student-torch-dtype",
            "float32",
            "--teacher-torch-dtype",
            "bfloat16",
            "--pp",
            "4",
            "--ep",
            "2",
            "--dp-shard",
            "2",
        ],
    )

    worker.main()

    assert captured["stage"] == "global_distillation_sanity"
    assert "distillation_overfit" not in captured["config"]
    assert "parallel" not in captured["config"]
    sanity = captured["config"]["global_distillation_sanity"]
    assert sanity["profile_id"] == "runtime-075"
    assert sanity["solution_ids"] == ["best"]
    assert sanity["registry_path"] == str(tmp_path / "selected_solutions.json")
    assert (sanity["sample_count"], sanity["sequence_length"], sanity["max_steps"]) == (
        16,
        8192,
        128,
    )
    assert sanity["automodel"]["parallel"] == {
        **mesh,
        "pp": 4,
        "ep": 2,
        "dp_shard": 2,
    }
    assert sanity["lr"] == 2.0e-4
    assert sanity["student_model_kwargs"] == {"torch_dtype": "float32"}
    assert sanity["teacher_model_kwargs"] == {"torch_dtype": "bfloat16"}


def test_profile_kd_overfit_worker_maps_legacy_sanity_parallelism(monkeypatch, tmp_path: Path):
    import examples.puzzletron.run_profile_distillation_overfit_worker as worker

    config = {
        "experiment": {"dir": str(tmp_path)},
        "parallel": {"tp": 1, "cp": 1, "pp": 2, "dp": 1, "ep": 4},
        "global_distillation_sanity": {
            "enabled": True,
            "tp": 1,
            "cp": 1,
            "pp": 2,
            "dp": 1,
            "ep": 4,
            "sequence_parallel": False,
        },
    }
    manifest = tmp_path / "manifests" / "build_library.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"config": config}))
    captured = {}

    def fake_run_stage(actual_config, stage):
        captured["config"] = actual_config
        captured["stage"] = stage
        return SimpleNamespace(manifest_path=tmp_path / "manifests" / "global_kd.json")

    monkeypatch.setattr(worker, "run_stage", fake_run_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_profile_distillation_overfit_worker.py", "--puzzle-dir", str(tmp_path)],
    )

    worker.main()

    assert captured["stage"] == "global_distillation_sanity"
    assert captured["config"]["global_distillation_sanity"]["automodel"]["parallel"] == {
        "tp": 1,
        "cp": 1,
        "pp": 2,
        "ep": 4,
        "dp_shard": 4,
        "dp_replicate": 1,
        "sequence_parallel": False,
        "pipeline_schedule": "1f1b",
    }
