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

"""Tests for VLM post-MIP evaluation adapters."""

import json
from pathlib import Path

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import post_mip, suites
from tests.unit.torch.puzzletron.evaluation.vlm.vlm_test_utils import (
    _use_offline_fakes,
    _write_checkpoint,
    _write_lmms_tasks,
)


def test_post_mip_realworldqa_adapter_runs_pinned_profile(monkeypatch, tmp_path):
    model = _write_checkpoint(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("realworldqa",))
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    output = tmp_path / "output"
    captured = {}

    def fake_runner(checkpoint_path, *, output_root, settings):
        report = json.loads((output / "profile.json").read_text())
        assert report["configured_tasks"] == ["modelopt_vlm_benchmark_realworldqa"]
        assert report["sample_limit"] == 2
        captured.update(
            checkpoint=checkpoint_path,
            output_root=output_root,
            settings=settings,
        )
        result_path = output_root / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n")
        return {
            "metrics": {"modelopt_vlm_benchmark_realworldqa.accuracy": 0.5},
            "result_path": str(result_path),
        }

    monkeypatch.setattr(checkpoint, "run_lmms_eval_checkpoint", fake_runner)
    result = post_mip.evaluate_realworldqa_checkpoint(
        model,
        output_root=output,
        settings={
            "batch_size": 1,
            "timeout_seconds": 900,
            "dtype": "bfloat16",
            "topology": {"tensor_parallel_size": 1},
        },
    )

    assert captured["checkpoint"] == model
    assert captured["output_root"] == output
    assert captured["settings"]["tasks"] == "modelopt_vlm_benchmark_realworldqa"
    assert captured["settings"]["limit"] == 2
    assert captured["settings"]["timeout_seconds"] == 900
    assert captured["settings"]["dtype"] == "bfloat16"
    assert captured["settings"]["topology"] == {"tensor_parallel_size": 1}
    assert result["metrics"] == {"modelopt_vlm_benchmark_realworldqa.accuracy": 0.5}
    assert result["profile_path"] == str(output / "profile.json")


def test_post_mip_prefix100_adapter_averages_repeated_bounded_tasks(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"
    captured = {"invocations": 0}

    def fake_evaluate(args, *, settings_overrides, preflight_callback):
        captured["invocations"] += 1
        captured.update(args=args, settings_overrides=settings_overrides)
        preflight_callback({"profile": suites.EVALUATION_PROFILE, "sample_limit": None})
        runs = []
        score_offset = (captured["invocations"] - 1) * 0.2
        for index, realworldqa_score in enumerate(
            (0.4 + score_offset, 0.6 + score_offset), start=1
        ):
            result_path = tmp_path / f"run-{index}.json"
            result_path.write_text(
                json.dumps(
                    {
                        "sample_counts": {"realworldqa": 100, "mmmu_val": 100},
                        "mmmu_parser_audit": {
                            "sample_count": 100,
                            "status_counts": {"parsed": 90, "fallback_random": 10},
                        },
                    }
                )
            )
            runs.append(
                {
                    "metrics": {
                        "modelopt_vlm_benchmark_realworldqa.exact_match_none": (realworldqa_score),
                        "modelopt_vlm_benchmark_mmmu_val.mmmu_acc_none": 0.3,
                    },
                    "result_path": str(result_path),
                }
            )
        return {"runs": runs}

    monkeypatch.setattr(post_mip, "evaluate", fake_evaluate)
    result = post_mip.evaluate_realworldqa_mmmu_prefix100_checkpoint(
        model,
        output_root=output,
        settings={
            "batch_size": 1,
            "timeout_seconds": 14400,
            "dtype": "bfloat16",
            "topology": {"tensor_parallel_size": 1},
        },
    )

    assert captured["args"].suite == suites.TASK_PREFIX100_REPEAT2_SUITE
    assert captured["args"].batch_size == 1
    assert captured["args"].seed == 42
    assert captured["settings_overrides"] == {
        "dtype": "bfloat16",
        "topology": {"tensor_parallel_size": 1},
    }
    assert result["metrics"] == {
        "modelopt_vlm_benchmark_mmmu_val.mmmu_acc_none": 0.3,
        "modelopt_vlm_benchmark_realworldqa.exact_match_none": 0.5,
    }
    assert result["profile"] == post_mip.TASK_PREFIX100_REPEAT2_PROFILE
    summary = json.loads(Path(result["result_path"]).read_text())
    assert summary["suite"] == suites.TASK_PREFIX100_REPEAT2_SUITE
    assert summary["profile"] == post_mip.TASK_PREFIX100_REPEAT2_PROFILE
    assert summary["metrics"] == result["metrics"]
    assert summary["result_paths"] == result["run_result_paths"]
    assert summary["sample_counts"] == {"mmmu_val": 200, "realworldqa": 200}
    assert summary["mmmu_parser_audit"] == {
        "sample_count": 200,
        "status_counts": {"fallback_random": 20, "parsed": 180},
    }

    refreshed = post_mip.evaluate_realworldqa_mmmu_prefix100_checkpoint(
        model,
        output_root=output,
        settings={
            "batch_size": 1,
            "timeout_seconds": 14400,
            "dtype": "bfloat16",
            "topology": {"tensor_parallel_size": 1},
        },
    )
    assert refreshed["metrics"][
        "modelopt_vlm_benchmark_realworldqa.exact_match_none"
    ] == pytest.approx(0.7)
    assert json.loads(Path(refreshed["result_path"]).read_text())["metrics"] == refreshed["metrics"]


def test_post_mip_prefix100_rejects_different_repetition_metrics(
    monkeypatch,
    tmp_path,
):
    def fake_evaluate(args, *, settings_overrides, preflight_callback):
        return {
            "runs": [
                {"metrics": {"realworldqa.accuracy": 0.5}, "result_path": "first.json"},
                {"metrics": {"mmmu.accuracy": 0.5}, "result_path": "second.json"},
            ]
        }

    monkeypatch.setattr(post_mip, "evaluate", fake_evaluate)

    with pytest.raises(RuntimeError, match="produced different metrics"):
        post_mip.evaluate_realworldqa_mmmu_prefix100_checkpoint(
            tmp_path / "model",
            output_root=tmp_path / "output",
            settings={},
        )


def test_deprecated_post_mip_profile_alias_forwards_to_canonical(monkeypatch, tmp_path):
    expected = {"metrics": {"accuracy": 0.5}}
    monkeypatch.setattr(
        post_mip,
        "evaluate_realworldqa_mmmu_prefix100_checkpoint",
        lambda *_args, **_kwargs: expected,
    )

    with pytest.warns(FutureWarning, match="is deprecated"):
        result = post_mip.evaluate_e2e_full_eval_checkpoint(
            tmp_path / "model",
            output_root=tmp_path / "output",
            settings={},
        )

    assert result is expected
