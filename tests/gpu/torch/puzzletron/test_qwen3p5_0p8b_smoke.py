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

"""Opt-in real-checkpoint smoke for the default Qwen 3.5 0.8B MIP route.

This recipe validates the conservative FFN-only search. The broader advanced
search remains a separate unvalidated follow-up.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import yaml
from datasets import Dataset, DatasetDict

if TYPE_CHECKING:
    from pathlib import Path


def _save_messages_dataset(path: Path) -> None:
    response = (
        "Compression removes redundant parameters while keeping useful behavior. " * 64
    ).strip()
    messages = [
        {"role": "user", "content": "What is model compression?"},
        {"role": "assistant", "content": response},
    ]
    rows = [{"messages": messages}] * 16
    DatasetDict(
        {
            "train": Dataset.from_list(rows),
            "validation": Dataset.from_list(rows),
        }
    ).save_to_disk(str(path))


@pytest.mark.integration
@pytest.mark.manual(reason="downloads and prunes the real Qwen 3.5 0.8B checkpoint")
@pytest.mark.timeout(2400)
def test_qwen3p5_0p8b_default_orchestrated_smoke_completes_at_mip(
    project_root_path: Path,
    tmp_path: Path,
) -> None:
    """Run the enabled DAG through MIP; requires one H100 80GB GPU."""

    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    cache = tmp_path / "cache"
    runner = tmp_path / "runner.yaml"
    _save_messages_dataset(dataset)
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "local-smoke", "max_nodes": 1},
                    "execution_contract": {
                        "repository": str(project_root_path),
                        "venv": sys.prefix,
                        "container": None,
                        "container_mounts": None,
                        "prerun_commands": [],
                        "postrun_commands": [],
                    },
                }
            },
            sort_keys=False,
        )
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PUZZLETRON_RUN_ROOT": str(results),
            "PUZZLETRON_DATASET_PATH": str(dataset),
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root_path / "examples/puzzletron/orchestrate.py"),
            "--experiment",
            str(
                project_root_path / "examples/puzzletron/configs/families/qwen3_5/"
                "qwen3p5_0p8b/runs/default.yaml"
            ),
            "--runner",
            str(runner),
            "--execution",
            str(
                project_root_path / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b/"
                "execution.smoke.yaml"
            ),
            "--stage",
            "full",
            "--local",
            "--color",
            "never",
        ],
        cwd=project_root_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=2300,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "Qwen 3.5 0.8B orchestrated smoke failed.\n"
            f"stdout tail:\n{completed.stdout[-12000:]}\n"
            f"stderr tail:\n{completed.stderr[-12000:]}"
        )

    sanity_summaries = {
        stage: json.loads((results / f"artifacts/{stage}/summary.json").read_text(encoding="utf-8"))
        for stage in ("width_sanity", "slicing_sanity")
    }
    for stage, summary in sanity_summaries.items():
        assert summary["schema_version"] == 1
        assert summary["stage"] == stage
        assert set(summary["axes"]) == {"ffn_intermediate"}

    width_summary = sanity_summaries["width_sanity"]
    assert width_summary["verdict"] in {"passed", "warning"}
    if width_summary["verdict"] == "warning":
        assert width_summary["passed"] is False
        assert width_summary["findings"]
        assert {finding["severity"] for finding in width_summary["findings"]} == {"warning"}

    slicing_summary = sanity_summaries["slicing_sanity"]
    assert slicing_summary["passed"] is True
    assert slicing_summary["verdict"] == "passed"
    assert slicing_summary["provenance"]["backend"] == "distributed_parent_sweep"
    diagnostic_summary = json.loads(
        (results / "artifacts/activation_diagnostic/activation_diagnostic_summary.json").read_text(
            encoding="utf-8"
        )
    )
    primary_metric = diagnostic_summary["primary_metric"]
    assert primary_metric in slicing_summary["metric_specs"]

    run_config = yaml.safe_load(
        (
            project_root_path
            / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/default.yaml"
        ).read_text(encoding="utf-8")
    )
    width_sanity_config = run_config["width_sanity"]
    expected_physical_cases = width_sanity_config["layer_count"]
    assert expected_physical_cases == 2
    assert width_sanity_config["target_count_per_axis"] == 1

    ffn_rows_by_case = {}
    for row in slicing_summary["rows"]:
        if row["axis"] != "ffn_intermediate":
            continue
        case = (row["layer_idx"], row["target_value"])
        ffn_rows_by_case.setdefault(case, {})[row["method"]] = row
    physical_ffn_rows = [
        row
        for row in slicing_summary["rows"]
        if row["axis"] == "ffn_intermediate" and row["method"] == "physical"
    ]
    physical_cases = {(row["layer_idx"], row["target_value"]) for row in physical_ffn_rows}
    assert len(physical_ffn_rows) == len(physical_cases) == expected_physical_cases
    assert len({layer_idx for layer_idx, _ in physical_cases}) == expected_physical_cases
    for row in physical_ffn_rows:
        assert row["parent_role"].startswith("realized_")
        assert row["teacher_value"] == 3584
        assert row["target_value"] in {3072, 2048}
        assert row["num_changed_layers"] == 1
        assert json.loads(row["changed_layers"]) == [row["layer_idx"]]
        matching_rows = ffn_rows_by_case[(row["layer_idx"], row["target_value"])]
        assert "sorted" in matching_rows
        for metric_row in (matching_rows["sorted"], row):
            value = metric_row.get(primary_metric)
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value)

    mip_manifest = json.loads((results / "manifests/mip.json").read_text(encoding="utf-8"))
    assert mip_manifest["status"] == "success"
    active_profiles = json.loads((results / "mip/active_profiles.json").read_text(encoding="utf-8"))
    assert active_profiles["status"] == "success"
    assert active_profiles["profile_ids"] == ["params-90"]
    assert [profile["id"] for profile in active_profiles["profiles"]] == ["params-90"]
    assert active_profiles["profiles"][0]["feasible_count"] >= 1
