# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

from omegaconf import OmegaConf


def test_profile_evaluation_root_uses_canonical_stage_artifact_directory(tmp_path: Path):
    from examples.puzzletron.run_profile_evaluation_worker import _evaluation_root

    root = _evaluation_root(
        tmp_path,
        profile_id="latency-095",
        eval_samples=128,
        block_size=16384,
    )

    assert root == (
        tmp_path
        / "artifacts/zero_shot_evaluation/profiles/latency-095/text-s128-l16384"
    )


def test_single_gpu_evaluation_recipe_disables_all_parallel_dimensions(tmp_path: Path):
    from examples.puzzletron.run_profile_evaluation_worker import _single_gpu_recipe

    source = tmp_path / "recipe.yaml"
    output = tmp_path / "generated.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "step_scheduler": {"global_batch_size": 4, "local_batch_size": 2},
                "distributed": {
                    "tp_size": 2,
                    "cp_size": 2,
                    "pp_size": 2,
                    "dp_size": 2,
                    "ep_size": 1,
                    "pipeline": {"pp_batch_size": 4},
                },
            }
        ),
        source,
    )

    _single_gpu_recipe(source, output)

    recipe = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
    assert recipe["distributed"] == {
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "pipeline": {"pp_batch_size": 1, "pp_microbatch_size": 1},
    }
    assert recipe["step_scheduler"]["global_batch_size"] == 1
    assert recipe["step_scheduler"]["local_batch_size"] == 1


def test_evaluation_cli_forwards_post_kd_checkpoint_and_output(monkeypatch, tmp_path: Path):
    import examples.puzzletron.run_profile_evaluation_worker as worker

    captured = {}

    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "result.json"

    monkeypatch.setattr(worker, "evaluate_one", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_profile_evaluation_worker.py",
            "--puzzle-dir",
            str(tmp_path),
            "--profile-id",
            "latency-075",
            "--solution-id",
            "h3584-d1",
            "--checkpoint",
            str(tmp_path / "distilled"),
            "--output-dir",
            str(tmp_path / "post-kd"),
        ],
    )

    worker.main()

    assert captured["checkpoint_override"] == tmp_path / "distilled"
    assert captured["output_dir_override"] == tmp_path / "post-kd"


def test_evaluation_merge_cli_does_not_pass_single_worker_overrides(monkeypatch, tmp_path: Path):
    import examples.puzzletron.run_profile_evaluation_worker as worker

    captured = {}

    def fake_merge(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "summary.json"

    monkeypatch.setattr(worker, "merge_results", fake_merge)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_profile_evaluation_worker.py",
            "--puzzle-dir",
            str(tmp_path),
            "--profile-id",
            "latency-075",
            "--merge",
        ],
    )

    worker.main()

    assert set(captured) == {"profile_id", "eval_samples", "block_size"}
