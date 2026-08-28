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

"""Assertions shared by real-checkpoint post-materialization evaluation tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.post_mip.records import CandidateLedger

__all__ = ["assert_pruned_checkpoints_completed_benchmark"]


def _intermediate_sizes(value: Any) -> list[int]:
    if isinstance(value, dict):
        sizes = []
        for key, child in value.items():
            if (
                key == "intermediate_size"
                and isinstance(child, int)
                and not isinstance(child, bool)
            ):
                sizes.append(child)
            else:
                sizes.extend(_intermediate_sizes(child))
        return sizes
    if isinstance(value, list):
        return [size for child in value for size in _intermediate_sizes(child)]
    return []


def assert_pruned_checkpoints_completed_benchmark(
    run_root: Path,
    *,
    checkpoint_node: str,
    evaluation_node: str,
    task: str,
    limit: int,
) -> None:
    """Verify every checkpoint revision from one node completed the benchmark."""

    ledger = CandidateLedger(run_root / "artifacts/post_mip")
    candidate_set = ledger.load_candidate_set(checkpoint_node)
    checkpoints = {
        Path(ledger.revisions[revision_id].artifact["checkpoint"]).resolve()
        for revision_id in candidate_set.revision_ids
    }
    assert checkpoints
    for checkpoint in checkpoints:
        config_path = checkpoint / "config.json"
        assert config_path.is_file()
        assert any(checkpoint.glob("*.safetensors"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        language_config = config.get("text_config", config)
        teacher_size = language_config["intermediate_size"]
        block_configs = language_config.get("block_configs", config.get("block_configs"))
        assert isinstance(block_configs, list) and block_configs
        assert any(size < teacher_size for size in _intermediate_sizes(block_configs))

    summaries = sorted(
        run_root.glob(
            f"artifacts/post_mip/nodes/{evaluation_node}/executions/*/raw/*/"
            "lmms_eval/attempt_*/summary.json"
        )
    )
    assert summaries
    evaluated = set()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary["checkpoint"]).resolve()
        evaluated.add(checkpoint)
        assert checkpoint in checkpoints
        task_counts = {
            name: count
            for name, count in summary["sample_counts"].items()
            if name == task or name.startswith(f"{task}_")
        }
        assert task_counts
        assert all(count == limit for count in task_counts.values())
        task_metrics = {
            name: value
            for name, value in summary["metrics"].items()
            if name == task or name.startswith(f"{task}.")
        }
        assert task_metrics
        assert all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in task_metrics.values()
        )

        command = json.loads((summary_path.parent / "command.json").read_text(encoding="utf-8"))
        argv = command["argv"]
        model_args = argv[argv.index("--model_args") + 1]
        assert f"model={checkpoint}" in model_args.split(",")

    assert evaluated == checkpoints
