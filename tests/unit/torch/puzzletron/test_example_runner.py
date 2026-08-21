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

"""Tests for the public Puzzletron stage runner."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from examples.puzzletron import main as puzzletron_main
from examples.puzzletron.main import (
    _complete_composite_stage,
    _validate_worker_result,
    build_worker_command,
)
from modelopt.torch.puzzletron.manifest import (
    stage_manifest_from_config,
    validate_stage_execution_record,
    write_stage_manifest,
)
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.stage_runner import StageResult, run_stage


def test_worker_command_propagates_gpu_count_to_composite_followups():
    command = build_worker_command(
        config_path="experiment.yaml",
        stage="build_library",
        overrides=(),
        gpus_per_node=1,
    )

    assert command[command.index("--gpus-per-node") + 1] == "1"


def test_build_library_worker_does_not_forward_mutable_base_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "puzzle_dir": str(tmp_path),
        "embedding_pruning": {"enabled": True, "widths": [256]},
        "build_library": {"enabled": True},
        "execution": {"gpus_per_node": 1},
    }
    initial = StageResult(
        stage="build_library",
        status="success",
        manifest_path=tmp_path / "manifests" / "build_library.json",
        message="initial root build",
    )
    captured_outputs = {}

    monkeypatch.setattr(
        puzzletron_main.mtpz.pipeline_config,
        "pipeline_config_from_path",
        lambda *_args, **_kwargs: deepcopy(config),
    )
    monkeypatch.setattr(
        puzzletron_main.mtpz.stage_runner,
        "run_stage",
        lambda *_args, **_kwargs: initial,
    )
    monkeypatch.setattr(
        puzzletron_main,
        "_run_embedding_stage",
        lambda **_kwargs: {
            "stage": "build_library",
            "widths": [256],
            "scenarios_root": str(tmp_path / "scenarios"),
        },
    )

    def complete_composite(_config, _stage, outputs):
        captured_outputs.update(outputs)
        return initial

    monkeypatch.setattr(puzzletron_main, "_complete_composite_stage", complete_composite)
    monkeypatch.setattr(puzzletron_main, "_validate_worker_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(puzzletron_main, "refresh_campaign_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(puzzletron_main.mtpz.tools, "mprint", lambda *_args, **_kwargs: None)

    puzzletron_main._run_worker(
        SimpleNamespace(
            config=tmp_path / "experiment.yaml",
            override=[],
            worker_stage="build_library",
            scenario_child=False,
            gpus_per_node=1,
        )
    )

    assert captured_outputs == {
        "stage": "build_library",
        "widths": [256],
        "scenarios_root": str(tmp_path / "scenarios"),
    }
    assert "base_manifest" not in captured_outputs


def test_build_library_composite_preserves_authored_and_effective_config(tmp_path: Path) -> None:
    authored_config = {
        "puzzle_dir": str(tmp_path),
        "experiment": {"dir": str(tmp_path)},
        "model": {"source": "example/model"},
        "build_library": {"enabled": True},
        "embedding_pruning": {"enabled": False},
        "vllm_stats": {"subblock_stats_filename": "subblock_stats.json"},
    }
    worker_config = deepcopy(authored_config)
    worker_config["build_library"]["include_noops"] = False
    worker_config["_runtime"] = {
        "config_path": str(tmp_path / "experiment.yaml"),
        "authored_config": deepcopy(authored_config),
    }

    outputs = {}
    for name in ("replacement_library.json", "candidate_library.json", "subblock_stats.json"):
        path = tmp_path / name
        path.write_text("{}\n")
        outputs[name.removesuffix(".json")] = str(path)

    manifest_path = tmp_path / "manifests" / "build_library.json"
    initial = stage_manifest_from_config("build_library", worker_config)
    initial.complete(outputs=outputs)
    write_stage_manifest(manifest_path, initial)
    initial_pointer = json.loads(manifest_path.read_text())

    result = _complete_composite_stage(
        worker_config,
        "build_library",
        {"stage": "build_library", "widths": [], "scenarios_root": str(tmp_path / "scenarios")},
    )

    pointer = json.loads(manifest_path.read_text())
    resolved = json.loads(
        (tmp_path / pointer["execution_record"]["resolved_config_path"]).read_text()
    )
    assert pointer["config"] == authored_config
    assert pointer["semantic_config"] == initial_pointer["semantic_config"]
    assert resolved["resolved_stage_config"]["build_library"]["include_noops"] is False
    validate_stage_execution_record(manifest_path, expected_stage="build_library")
    _validate_worker_result(worker_config, result, expected_stage="build_library")
    assert stage_is_complete(authored_config, "build_library")
    assert stage_is_complete(worker_config, "build_library")


def test_loaded_stage_run_publishes_distinct_authored_and_effective_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            (
                f"puzzle_dir: {tmp_path}",
                "experiment:",
                f"  dir: {tmp_path}",
                "sort_sanity:",
                "  enabled: true",
                "width_sanity:",
                "  enabled: true",
                "slicing_sanity:",
                "  enabled: true",
            )
        )
        + "\n"
    )
    override = "+slicing_sanity.tolerance=0.25"
    config = pipeline_config_from_path(config_path, overrides=[override])
    manifest_path = tmp_path / "manifests" / "slicing_sanity.json"

    def capture_handler(effective_config, manifest):
        manifest.complete(outputs={})
        write_stage_manifest(manifest_path, manifest)
        return StageResult(
            stage="slicing_sanity",
            status="success",
            manifest_path=manifest_path,
            message="captured",
        )

    run_stage(config, "slicing_sanity", handlers={"slicing_sanity": capture_handler})

    pointer = json.loads(manifest_path.read_text())
    resolved = json.loads(
        (tmp_path / pointer["execution_record"]["resolved_config_path"]).read_text()
    )
    assert pointer["config"]["slicing_sanity"] == {
        "enabled": True,
        "tolerance": 0.25,
    }
    assert pointer["config"]["sort_sanity"] == {"enabled": True}
    assert "search_space" not in pointer["config"]
    assert resolved["resolved_stage_config"]["search_space"] == {"axes": {}}
    assert resolved["provenance"]["overrides"] == [override]
