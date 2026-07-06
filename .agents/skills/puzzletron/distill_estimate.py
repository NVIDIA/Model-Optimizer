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

"""This AI-generated script is for experimental use only.

Estimate a distillation and downstream-evaluation workflow from a completed run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import yaml

_ITERATION_RE = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] iteration\s+"
    r"(?P<iteration>\d+)/\s*\d+.*?elapsed time per iteration \(ms\): (?P<ms>[\d.]+)"
)
_EVALUATE_RE = re.compile(r"evaluate\s+.*?: \([\d.]+, ([\d.]+)\)")
_MMLU_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}):(\d{2}:\d{2}:\d{2})")
_RESULT_TIME_RE = re.compile(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)\.json$")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unavailable"
    minutes = round(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def _format_size(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "unavailable"
    return f"{num_bytes / 1024**3:.1f} GiB"


def _latest_run_config(run_dir: Path) -> tuple[Path, dict]:
    configs = sorted((run_dir / "checkpoints").glob("iter_*/run_config.yaml"))
    if not configs:
        raise FileNotFoundError(f"no checkpoint run_config.yaml found under {run_dir}")
    path = configs[-1]
    with path.open(encoding="utf-8") as stream:
        return path, yaml.safe_load(stream)


def _iteration_records(log_text: str) -> list[tuple[int, datetime, float]]:
    return [
        (
            int(match["iteration"]),
            datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S"),
            float(match["ms"]) / 1000,
        )
        for match in _ITERATION_RE.finditer(log_text)
    ]


def _training_calibration(log_text: str, source_eval_interval: int) -> tuple[float, float, float]:
    records = _iteration_records(log_text)
    if len(records) < 2:
        raise ValueError("the run log does not contain enough timestamped iterations")

    step_seconds = statistics.median(record[2] for record in records)
    evaluation_times = [float(value) / 1000 for value in _EVALUATE_RE.findall(log_text)]
    validation_seconds = statistics.median(evaluation_times) if evaluation_times else 0.0

    boundary_overheads = []
    for previous, current in pairwise(records):
        if previous[0] % source_eval_interval == 0:
            wall_gap = (current[1] - previous[1]).total_seconds()
            boundary_overheads.append(max(0.0, wall_gap - current[2] - validation_seconds))
    export_seconds = statistics.median(boundary_overheads) if boundary_overheads else 0.0
    return step_seconds, validation_seconds, export_seconds


def _directory_size(path: Path) -> int:
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file() and "eval_results" not in file.relative_to(path).parts
    )


def _checkpoint_size(run_dir: Path) -> float | None:
    exports = sorted((run_dir / "hf_validation").glob("iter_*"))
    sizes = [_directory_size(path) for path in exports if path.is_dir()]
    return statistics.median(sizes) if sizes else None


def _result_timestamp(path: Path) -> datetime | None:
    match = _RESULT_TIME_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S.%f")


def _mmlu_duration(checkpoint: Path, limit: int) -> float | None:
    log = checkpoint / "eval_results" / f"mmlu_limit_{limit}.log"
    if not log.is_file():
        return None
    match = _MMLU_TIME_RE.search(log.read_text(encoding="utf-8", errors="replace"))
    result_times = [
        timestamp
        for path in (checkpoint / "eval_results" / "mmlu").rglob("results_*.json")
        if (timestamp := _result_timestamp(path)) is not None
    ]
    if not match or not result_times:
        return None
    started = datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H:%M:%S")
    return (max(result_times) - started).total_seconds()


def _mmlu_pro_duration(checkpoint: Path, limit: int) -> float | None:
    result_dir = checkpoint / "eval_results" / f"mmlu_pro_limit_{limit}"
    manifest = result_dir / "manifest.json"
    if not manifest.is_file():
        return None
    with manifest.open(encoding="utf-8") as stream:
        if json.load(stream).get("limit_per_subject") != limit:
            return None
    result_times = [
        timestamp
        for path in result_dir.rglob("results_*.json")
        if (timestamp := _result_timestamp(path)) is not None
    ]
    if not result_times:
        return None
    started = datetime.fromtimestamp(manifest.stat().st_mtime)
    return (max(result_times) - started).total_seconds()


def _evaluation_calibration(run_dir: Path, limit: int, mmlu_pro: bool) -> float | None:
    durations = []
    for checkpoint in sorted((run_dir / "hf_validation").glob("iter_*")):
        duration = (
            _mmlu_pro_duration(checkpoint, limit) if mmlu_pro else _mmlu_duration(checkpoint, limit)
        )
        if duration is not None and duration > 0:
            durations.append(duration)
    return statistics.median(durations) if durations else None


def _future_milestones(current: int, target: int, interval: int) -> list[int]:
    first = (current // interval + 1) * interval
    milestones = list(range(first, target + 1, interval))
    if target > current and (not milestones or milestones[-1] != target):
        milestones.append(target)
    return milestones


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing distillation run used for calibration")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--hf-export-interval", type=int)
    parser.add_argument("--mmlu-interval", type=int)
    parser.add_argument("--mmlu-pro-interval", type=int)
    parser.add_argument("--mmlu-limit", type=int, default=25)
    parser.add_argument("--mmlu-pro-limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Estimate time and storage for continuing a distillation run."""
    args = _parse_args()
    config_path, config = _latest_run_config(args.run_dir)
    train = config["train"]
    dataset = config["dataset"]
    current_iteration = int(config_path.parent.name.removeprefix("iter_"))
    tokens_per_iteration = train["global_batch_size"] * dataset["sequence_length"]
    target_iteration = math.ceil(args.target_tokens / tokens_per_iteration)

    source_eval_interval = train["eval_interval"]
    eval_interval = args.eval_interval or source_eval_interval
    export_interval = args.hf_export_interval or eval_interval
    mmlu_interval = args.mmlu_interval or export_interval
    mmlu_pro_interval = args.mmlu_pro_interval or export_interval
    for name, value in (
        ("target_tokens", args.target_tokens),
        ("eval_interval", eval_interval),
        ("hf_export_interval", export_interval),
        ("mmlu_interval", mmlu_interval),
        ("mmlu_pro_interval", mmlu_pro_interval),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if export_interval % eval_interval:
        raise ValueError("hf_export_interval must be a multiple of eval_interval")
    if mmlu_interval % export_interval or mmlu_pro_interval % export_interval:
        raise ValueError("evaluation intervals must be multiples of hf_export_interval")

    log_path = args.run_dir / "log.txt"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    step_seconds, validation_seconds, export_seconds = _training_calibration(
        log_text, source_eval_interval
    )
    checkpoint_size = _checkpoint_size(args.run_dir)
    mmlu_seconds = _evaluation_calibration(args.run_dir, args.mmlu_limit, mmlu_pro=False)
    mmlu_pro_seconds = _evaluation_calibration(args.run_dir, args.mmlu_pro_limit, mmlu_pro=True)

    additional_iterations = max(0, target_iteration - current_iteration)
    validation_iterations = _future_milestones(current_iteration, target_iteration, eval_interval)
    export_iterations = _future_milestones(current_iteration, target_iteration, export_interval)
    mmlu_iterations = _future_milestones(current_iteration, target_iteration, mmlu_interval)
    mmlu_pro_iterations = _future_milestones(current_iteration, target_iteration, mmlu_pro_interval)

    distill_seconds = (
        additional_iterations * step_seconds
        + len(validation_iterations) * validation_seconds
        + len(export_iterations) * export_seconds
    )
    mmlu_total = None if mmlu_seconds is None else len(mmlu_iterations) * mmlu_seconds
    mmlu_pro_total = (
        None if mmlu_pro_seconds is None else len(mmlu_pro_iterations) * mmlu_pro_seconds
    )
    known_total = distill_seconds + (mmlu_total or 0) + (mmlu_pro_total or 0)

    print("\nPuzzletron experiment estimate")
    print("=" * 72)
    print(f"Calibration run:       {args.run_dir}")
    print(f"Current iteration:     {current_iteration:,}")
    print(f"Target iteration:      {target_iteration:,}")
    print(f"Additional iterations: {additional_iterations:,}")
    print(f"Tokens per iteration:  {tokens_per_iteration:,}")
    print(f"Target tokens:         {args.target_tokens:,}")
    print()
    print(
        f"Validation:            {len(validation_iterations):>4} runs,"
        f" every {eval_interval} iterations"
    )
    print(
        f"HF exports:            {len(export_iterations):>4} checkpoints,"
        f" every {export_interval} iterations"
    )
    print(
        f"MMLU:                  {len(mmlu_iterations):>4} runs, every {mmlu_interval} iterations"
    )
    print(
        f"MMLU-Pro:              {len(mmlu_pro_iterations):>4} runs,"
        f" every {mmlu_pro_interval} iterations"
    )
    print()
    print(f"Distillation:          {_format_duration(distill_seconds)}")
    print(
        f"MMLU (limit={args.mmlu_limit}):       {_format_duration(mmlu_total)}"
        f"  ({_format_duration(mmlu_seconds)} each)"
    )
    print(
        f"MMLU-Pro (limit={args.mmlu_pro_limit}):   {_format_duration(mmlu_pro_total)}"
        f"  ({_format_duration(mmlu_pro_seconds)} each)"
    )
    print(f"Known serial total:    {_format_duration(known_total)}")
    storage = None if checkpoint_size is None else len(export_iterations) * checkpoint_size
    print(f"New HF storage:        {_format_size(storage)}  ({_format_size(checkpoint_size)} each)")
    print()
    print("Calibration:")
    print(f"  training step:       {step_seconds:.2f}s")
    print(f"  validation stage:    {validation_seconds:.2f}s")
    print(f"  HF export overhead:  {export_seconds:.2f}s")
    print("  Estimates exclude startup, queue time, and parallel execution savings.")


if __name__ == "__main__":
    main()
