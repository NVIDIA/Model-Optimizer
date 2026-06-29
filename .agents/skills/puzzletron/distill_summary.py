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

# Generated with Claude Code

"""Summarize and compare Puzzletron distillation runs.

Usage:
  python distill_summary.py [puzzle_dir]
"""

import ast
import contextlib
import glob
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

_RECIPE_FIELDS = (
    "student_hf_path",
    "teacher_hf_path",
    "train_iters",
    "lr",
    "min_lr",
    "lr_warmup_iters",
    "mbs",
    "gbs",
    "seq_length",
    "tp_size",
    "pp_size",
    "cp_size",
    "ep_size",
    "eval_interval",
    "eval_iters",
    "log_interval",
    "seed",
    "kd_loss_scale",
    "no_skip_lm_loss",
)


@dataclass(frozen=True)
class MmluResult:
    """Saved aggregate MMLU accuracy and its sample limit."""

    accuracy: float
    limit: int | None
    mtime: float


@dataclass(frozen=True)
class RunSummary:
    """Evidence-backed summary of one distillation output directory."""

    name: str
    output_dir: Path
    arguments: dict[str, str]
    dataset: str
    checkpoints: tuple[int, ...]
    status: str
    mmlu: MmluResult | None
    allocated_bytes: int


def find_puzzle_dir_candidates() -> list[Path]:
    """Return deduplicated existing Puzzletron output directories."""
    candidates = sorted(
        glob.glob("puzzle_dir_*")
        + glob.glob("../puzzle_dir_*")
        + glob.glob("/workspace/puzzle_dir_*")
    )
    seen = set()
    result = []
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path.is_dir() and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def parse_log_arguments(log_path: Path) -> dict[str, str]:
    """Read the argument dump emitted by ``examples/megatron_bridge/distill.py``."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}

    match = re.search(
        r"^=+ Arguments =+\s*$\n(?P<body>.*?)^=+\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return {}

    arguments = {}
    for line in match.group("body").splitlines():
        field = re.match(r"^(\w+)\s{2,}(.+?)\s*$", line)
        if field:
            arguments[field.group(1)] = field.group(2)
    return arguments


def parse_data_paths(value: str | None) -> list[str]:
    """Extract dataset paths from a logged Megatron weighted-blend list."""
    if not value:
        return []
    try:
        items = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    paths = []
    for index, item in enumerate(items):
        if index % 2 == 1 and isinstance(item, str):
            paths.append(item)
    return paths


def dataset_label(paths: list[str]) -> str:
    """Return a concise dataset label from logged indexed-dataset paths."""
    if not paths:
        return "unknown"
    if all("wikitext" in path.lower() for path in paths):
        return "WikiText-103"
    if all("Nemotron-Post-Training-Dataset-v2" in path for path in paths):
        splits = []
        for path in paths:
            match = re.search(r"_default_(.+?)_messages(?:_max\d+)?$", path)
            splits.append(match.group(1) if match else Path(path).name)
        return f"Nemotron v2: {'+'.join(splits)}"

    labels = []
    for path in paths:
        label = Path(path).name
        label = re.sub(r"(?:_text_document|_messages_max\d+)$", "", label)
        labels.append(label)
    return "+".join(labels)


def checkpoint_iterations(output_dir: Path) -> tuple[int, ...]:
    """Return sorted checkpoint iterations found under a run."""
    iterations = []
    for path in (output_dir / "checkpoints").glob("iter_*"):
        with contextlib.suppress(ValueError):
            iterations.append(int(path.name.removeprefix("iter_")))
    return tuple(sorted(iterations))


def has_hf_weights(hf_dir: Path) -> bool:
    """Return whether a directory contains top-level Hugging Face model weights."""
    if not hf_dir.is_dir():
        return False
    return any(hf_dir.glob("*.safetensors")) or any(hf_dir.glob("pytorch_model*.bin"))


def read_mmlu_result(hf_dir: Path) -> MmluResult | None:
    """Return the newest full MMLU result, falling back to the newest limited result."""
    results = []
    for path in (hf_dir / "eval_results" / "mmlu").glob("**/results_*.json"):
        try:
            data = json.loads(path.read_text())
            metrics = data.get("results", {}).get("mmlu", {})
            accuracy = metrics.get("acc,none", metrics.get("acc"))
            if accuracy is None:
                continue
            limit = data.get("config", {}).get("limit")
            results.append(MmluResult(float(accuracy), limit, path.stat().st_mtime))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    full = [result for result in results if result.limit is None]
    return max(full or results, key=lambda result: result.mtime, default=None)


def allocated_size(path: Path) -> int:
    """Return allocated filesystem bytes without invoking a shell command."""
    total = 0
    seen = set()
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root, filename)
            try:
                stat = file_path.stat()
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_blocks * 512
    return total


def summarize_run(output_dir: Path) -> RunSummary:
    """Build a summary from one distillation output directory."""
    arguments = parse_log_arguments(output_dir / "log.txt")
    checkpoints = checkpoint_iterations(output_dir)
    hf_dir = output_dir / "hf"
    if has_hf_weights(hf_dir):
        status = "DONE"
    elif checkpoints:
        status = "CHECKPOINT"
    else:
        status = "STARTED"
    return RunSummary(
        name=output_dir.name,
        output_dir=output_dir,
        arguments=arguments,
        dataset=dataset_label(parse_data_paths(arguments.get("data_paths"))),
        checkpoints=checkpoints,
        status=status,
        mmlu=read_mmlu_result(hf_dir),
        allocated_bytes=allocated_size(output_dir),
    )


def format_bytes(size: int) -> str:
    """Format allocated bytes using binary units."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            decimals = 1 if value < 10 and unit != "B" else 0
            return f"{value:.{decimals}f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def format_checkpoints(iterations: tuple[int, ...]) -> str:
    """Format saved iterations compactly."""
    if not iterations:
        return "—"
    if len(iterations) <= 3:
        return ",".join(map(str, iterations))
    differences = {right - left for left, right in pairwise(iterations)}
    if len(differences) == 1:
        step = differences.pop()
        return f"{iterations[0]}-{iterations[-1]}/{step}"
    return f"{iterations[0]},…, {iterations[-1]}"


def format_mmlu(result: MmluResult | None) -> str:
    """Format MMLU accuracy while making limited evaluations explicit."""
    if result is None:
        return "—"
    suffix = f" (limit={result.limit})" if result.limit is not None else ""
    return f"{result.accuracy:.4f}{suffix}"


def argument(arguments: dict[str, str], name: str, default: str = "—") -> str:
    """Return one logged argument or a display default."""
    return arguments.get(name, default)


def truncate(value: str, width: int) -> str:
    """Truncate a table value with an ellipsis."""
    if len(value) <= width:
        return value
    return value[: width - 1] + "…"


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]):
    """Print a fixed-width terminal table."""
    line = "  ".join("─" * width for width in widths)
    print("  ".join(f"{header:<{width}}" for header, width in zip(headers, widths)))
    print(line)
    for row in rows:
        values = [truncate(value, width) for value, width in zip(row, widths)]
        print("  ".join(f"{value:<{width}}" for value, width in zip(values, widths)))


def recipe_signature(run: RunSummary) -> tuple[str, ...]:
    """Return the fields that must match for a controlled dataset comparison."""
    return tuple(argument(run.arguments, field) for field in _RECIPE_FIELDS)


def print_matched_groups(runs: list[RunSummary]):
    """Print groups that share model initialization and the complete training recipe."""
    groups: dict[tuple[str, ...], list[RunSummary]] = {}
    for run in runs:
        groups.setdefault(recipe_signature(run), []).append(run)
    matched = [group for group in groups.values() if len(group) > 1]
    print("\nMatched-recipe groups (safe for dataset comparisons):")
    if not matched:
        print("  none")
        return
    for index, group in enumerate(matched, start=1):
        names = ", ".join(run.name for run in group)
        datasets = "; ".join(run.dataset for run in group)
        print(f"  {index}. {names}")
        print(f"     datasets: {datasets}")


def print_model_paths(runs: list[RunSummary]):
    """Print common model paths or per-run paths when they differ."""
    students = {argument(run.arguments, "student_hf_path") for run in runs}
    teachers = {argument(run.arguments, "teacher_hf_path") for run in runs}
    if len(students) == 1 and len(teachers) == 1:
        print(f"\nCommon student: {next(iter(students))}")
        print(f"Common teacher: {next(iter(teachers))}")
        return
    print("\nModel paths:")
    for run in runs:
        print(f"  {run.name}")
        print(f"    student: {argument(run.arguments, 'student_hf_path')}")
        print(f"    teacher: {argument(run.arguments, 'teacher_hf_path')}")


def print_summary(puzzle_dir: Path):
    """Print outcome and recipe tables for all distillation runs."""
    distillation_dir = puzzle_dir / "distillation"
    if not distillation_dir.is_dir():
        raise FileNotFoundError(f"No distillation directory under {puzzle_dir}")
    runs = [summarize_run(path) for path in sorted(distillation_dir.iterdir()) if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No distillation runs under {distillation_dir}")

    print(f"\nDistillation summary — {puzzle_dir}")
    print("\nOutcomes and artifacts")
    outcome_rows = [
        (
            run.name,
            run.dataset,
            run.status,
            argument(run.arguments, "train_iters"),
            format_checkpoints(run.checkpoints),
            format_mmlu(run.mmlu),
            format_bytes(run.allocated_bytes),
        )
        for run in runs
    ]
    print_table(
        ("Run", "Dataset", "Status", "Steps", "Saved", "MMLU", "Disk"),
        outcome_rows,
        (34, 28, 10, 6, 12, 17, 9),
    )

    print("\nRecipe details")
    recipe_rows = [
        (
            run.name,
            f"{argument(run.arguments, 'lr')}→{argument(run.arguments, 'min_lr')}",
            argument(run.arguments, "lr_warmup_iters"),
            f"{argument(run.arguments, 'gbs')}x{argument(run.arguments, 'seq_length')}",
            f"{argument(run.arguments, 'eval_interval')}x{argument(run.arguments, 'eval_iters')}",
            argument(run.arguments, "log_interval"),
            argument(run.arguments, "tp_size"),
        )
        for run in runs
    ]
    print_table(
        ("Run", "LR->min LR", "Warm", "GBSxsequence", "Val everyxbatches", "Log", "TP"),
        recipe_rows,
        (34, 19, 6, 14, 18, 5, 3),
    )

    print_matched_groups(runs)
    print_model_paths(runs)
    print("\nEvidence: log argument dumps, checkpoint directories, HF MMLU JSON, allocated files.")
    print("Full MMLU results are preferred; limited results are labeled explicitly.")


def resolve_puzzle_dir(argv: list[str]) -> Path:
    """Resolve an explicit puzzle directory or require unambiguous auto-discovery."""
    if len(argv) > 1:
        raise ValueError("Usage: distill_summary.py [puzzle_dir]")
    if argv:
        path = Path(argv[0]).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Puzzle directory does not exist: {path}")
        return path
    candidates = find_puzzle_dir_candidates()
    if len(candidates) != 1:
        paths = "\n".join(f"  {path}" for path in candidates) or "  none"
        raise ValueError(f"Specify puzzle_dir; discovered candidates:\n{paths}")
    return candidates[0]


def main(argv: list[str]) -> int:
    """Run the command-line summary."""
    try:
        print_summary(resolve_puzzle_dir(argv))
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
