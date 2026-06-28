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
"""Progress report for /puzzletron eval mmlu (all checkpoints).

Usage:
  python eval_progress.py [puzzle_dir]

Reads checkpoint list from eval_list.py and checks eval_results/mmlu/ presence
plus JSON results to determine status and accuracy for each checkpoint.
"""

import contextlib
import glob
import json
import os
import re
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from itertools import pairwise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class EvalProgress:
    """Progress within the current lm-eval phase."""

    phase: str
    current: int | None = None
    total: int | None = None
    pct: int | None = None
    phase_elapsed_s: int | None = None
    phase_remaining_s: int | None = None


@dataclass(frozen=True)
class MmluResult:
    """A saved MMLU result and its sample limit, if any."""

    accuracy: float
    limit: int | None
    mtime: float


_MAJOR_TQDM_RE = re.compile(
    r"(?P<label>Loading weights|Tokenizing inputs|Running loglikelihood requests):\s*"
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<current>\d+)/(?P<total>\d+)"
    r"(?:\s*\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:]+),)?"
)
_ANY_TQDM_RE = re.compile(
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<current>\d+)/(?P<total>\d+)"
    r"(?:\s*\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:]+),)?"
)


def parse_tqdm_time(value):
    """Parse tqdm's MM:SS or H:MM:SS duration format."""
    if not value:
        return None
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def progress_from_match(match):
    """Convert a named tqdm match to structured progress."""
    return EvalProgress(
        phase=match.group("label"),
        current=int(match.group("current")),
        total=int(match.group("total")),
        pct=int(match.group("pct")),
        phase_elapsed_s=parse_tqdm_time(match.group("elapsed")),
        phase_remaining_s=parse_tqdm_time(match.group("remaining")),
    )


def parse_eval_progress(text):
    """Parse the current phase without confusing nested/reset tqdm bars."""
    major_matches = list(_MAJOR_TQDM_RE.finditer(text))
    major_match = major_matches[-1] if major_matches else None

    if "Saving results aggregated" in text:
        return EvalProgress(phase="Saving results")

    context_matches = list(re.finditer(r"Building contexts for (mmlu_\S+) on rank", text))
    last_context_pos = context_matches[-1].start() if context_matches else -1
    major_pos = major_match.start() if major_match else -1

    # Context construction has one reset tqdm bar per MMLU subtask. Combine
    # them into one task-level phase instead of exposing the latest reset bar.
    if context_matches and last_context_pos > major_pos:
        selected_tasks = set(re.findall(r"Task: (mmlu_[^\s(]+) \(", text))
        started_tasks = {match.group(1) for match in context_matches}
        total = max(len(selected_tasks), len(started_tasks))
        segment = text[last_context_pos:]
        task_matches = list(_ANY_TQDM_RE.finditer(segment))
        task_match = task_matches[-1] if task_matches else None
        task_fraction = int(task_match.group("pct")) / 100 if task_match else 0.0
        completed_equivalent = max(0, len(started_tasks) - 1) + task_fraction
        pct = round(100 * completed_equivalent / total) if total else None

        remaining_s = None
        timestamps = [
            int(hour) * 3600 + int(minute) * 60 + int(second)
            for hour, minute, second in re.findall(
                r"(\d{2}):(\d{2}):(\d{2}).*?Building contexts for mmlu_\S+ on rank", text
            )
        ]
        if len(timestamps) >= 2:
            intervals = [(later - earlier) % (24 * 3600) for earlier, later in pairwise(timestamps)]
            average_task_s = sum(intervals) / len(intervals)
            current_task_remaining_s = (
                parse_tqdm_time(task_match.group("remaining")) if task_match else 0
            )
            remaining_s = round(
                average_task_s * max(0, total - len(started_tasks))
                + (current_task_remaining_s or 0)
            )

        return EvalProgress(
            phase="Building contexts",
            current=len(started_tasks),
            total=total,
            pct=pct,
            phase_remaining_s=remaining_s,
        )

    if major_match:
        progress = progress_from_match(major_match)
        if progress.phase == "Loading weights" and progress.pct == 100:
            return EvalProgress(phase="Preparing tasks")
        return progress
    return EvalProgress(phase="Initializing")


def parse_memory_mib(dir_name):
    """Parse memory in MiB from a target_memory_<val>MiB directory name."""
    m = re.search(r"target_memory_([\d_]+)MiB", dir_name)
    if not m:
        return None
    parts = m.group(1).split("_", 1)
    return float(parts[0]) if len(parts) == 1 else float(f"{parts[0]}.{parts[1]}")


def find_teacher(puzzle_dir):
    """Find the original HF model path from the YAML config matching puzzle_dir."""
    puzzle_dir_abs = os.path.abspath(puzzle_dir)
    puzzle_dir_name = os.path.basename(puzzle_dir_abs)
    for yaml_path in glob.glob("examples/puzzletron/configs/**/*.yaml", recursive=True):
        try:
            content = open(yaml_path).read()
        except OSError:
            continue
        if puzzle_dir_abs in content or puzzle_dir_name in content:
            m = re.search(r"^input_hf_model_path\s*:\s*(\S+)", content, re.MULTILINE)
            if m:
                return m.group(1)
    return None


def get_distill_checkpoints(puzzle_dir):
    """Return list of (label, path) for distillation HF exports."""
    entries = []
    distill_base = os.path.join(puzzle_dir, "distillation")
    if not os.path.isdir(distill_base):
        return entries
    for run in sorted(os.listdir(distill_base)):
        hf_dir = os.path.join(distill_base, run, "hf")
        if os.path.isdir(hf_dir) and any(
            f.endswith((".safetensors", ".bin", "config.json")) for f in os.listdir(hf_dir)
        ):
            entries.append((f"distill:{run}", hf_dir))
    return entries


def get_checkpoints(puzzle_dir):
    """Return list of (label, path) for teacher + sweep solutions + distillation checkpoints."""
    teacher_path = find_teacher(puzzle_dir)
    ckpt_dirs = sorted(
        glob.glob(f"{puzzle_dir}/mip/puzzle_solutions/*/solutions--checkpoints/solution_0")
    )
    entries = []
    if teacher_path:
        entries.append(("teacher", teacher_path))
    for ckpt in ckpt_dirs:
        dir_name = ckpt.split("/mip/puzzle_solutions/")[1].split("/")[0]
        mem = parse_memory_mib(dir_name)
        label = f"{mem:,.0f} MiB" if mem is not None else dir_name
        entries.append((label, ckpt))
    entries.extend(get_distill_checkpoints(puzzle_dir))
    return entries


def get_running_evals():
    """Return process and current phase details for running lm-eval jobs."""
    results = {}
    try:
        ps_out = subprocess.run(  # nosec B603 B607
            ["ps", "-ww", "aux"], capture_output=True, text=True
        ).stdout
        for line in ps_out.splitlines():
            if "lm_eval_hf.py" not in line or "pretrained=" not in line:
                continue
            # Skip shell/wrapper processes — only match actual python processes
            if re.search(r"\s+(bash|sh|tee)\s", line) or "/bin/bash" in line or "/bin/sh" in line:
                continue
            path_m = re.search(r"pretrained=([^,\s]+)", line)
            pid_m = re.match(r"\S+\s+(\d+)", line)
            if not path_m or not pid_m:
                continue
            path = os.path.abspath(path_m.group(1))
            pid = pid_m.group(1)
            elapsed = None
            progress = EvalProgress(phase="Initializing")
            try:
                r = subprocess.run(  # nosec B603 B607
                    ["ps", "-o", "etimes=", "-p", pid], capture_output=True, text=True
                )
                elapsed = int(r.stdout.strip())
            except Exception:
                pass
            # Try to read tqdm progress. First check the eval log file (stdout is a
            # pipe to tee, so process fds won't contain the log file directly).
            log_candidates = []
            # For distillation checkpoints: <run_dir>/eval_mmlu.log
            parent = os.path.dirname(path)
            log_candidates.append(os.path.join(parent, "eval_mmlu.log"))
            # Sandboxed progress commands may not be allowed to inspect another
            # process's file descriptors. Search recent Puzzletron eval logs and
            # validate them against the model path before parsing.
            log_candidates.extend(
                glob.glob("/workspace/puzzle_dir_*/**/*mmlu*.log", recursive=True)
            )
            # Check files open by the python process itself
            try:
                for fd in os.listdir(f"/proc/{pid}/fd"):
                    try:
                        target = os.readlink(f"/proc/{pid}/fd/{fd}")
                    except OSError:
                        continue
                    if os.path.isfile(target):
                        log_candidates.append(target)
            except OSError:
                pass
            # The log is written by tee (sibling of python in the parent shell).
            # Find siblings by looking at all children of the parent process.
            try:
                status = open(f"/proc/{pid}/status").read()
                ppid_m = re.search(r"PPid:\s+(\d+)", status)
                if ppid_m:
                    siblings = subprocess.run(  # nosec B603 B607
                        ["ps", "--ppid", ppid_m.group(1), "-o", "pid="],
                        capture_output=True,
                        text=True,
                    ).stdout.split()
                    for spid in siblings:
                        spid = spid.strip()
                        if spid == pid:
                            continue
                        try:
                            for fd in os.listdir(f"/proc/{spid}/fd"):
                                try:
                                    target = os.readlink(f"/proc/{spid}/fd/{fd}")
                                except OSError:
                                    continue
                                if os.path.isfile(target) and target.endswith(".log"):
                                    log_candidates.append(target)
                        except OSError:
                            pass
            except Exception:
                pass
            existing_logs = {candidate for candidate in log_candidates if os.path.isfile(candidate)}
            for log_path in sorted(existing_logs, key=os.path.getmtime, reverse=True):
                try:
                    with open(log_path, "rb") as f:
                        text = f.read().decode("utf-8", errors="replace")
                    if path not in text:
                        continue
                    progress = parse_eval_progress(text)
                    if progress.phase != "Initializing":
                        break
                except OSError:
                    continue
            results[path] = {"pid": pid, "elapsed_s": elapsed, "progress": progress}
    except Exception:
        pass
    return results


def fmt_time(seconds):
    """Format seconds as a human-readable duration string."""
    if seconds is None:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def get_mmlu_accuracy(path):
    """Return the preferred saved MMLU result, favoring full evaluations."""
    results_dir = os.path.join(path, "eval_results", "mmlu")
    if not os.path.isdir(results_dir):
        return None
    # lm_eval saves results under a subdirectory named after the model path
    # (slashes replaced by __), then results_<timestamp>.json inside it.
    # Use os.walk instead of glob — glob skips dirs starting with '.' (e.g.
    # when the model path was relative, lm_eval names the subdir '..__..').
    candidates = []
    for dirpath, _dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.startswith("results_") or not fname.endswith(".json"):
                continue
            data = None
            with contextlib.suppress(OSError, json.JSONDecodeError, KeyError):
                data = json.load(open(os.path.join(dirpath, fname)))
            if data is None:
                continue
            results = data.get("results", {})
            if "mmlu" not in results:
                continue
            accuracy = results["mmlu"].get("acc,none")
            if accuracy is None:
                accuracy = results["mmlu"].get("acc")
            if accuracy is None:
                continue
            limit = data.get("config", {}).get("limit")
            result_path = os.path.join(dirpath, fname)
            candidates.append(
                MmluResult(
                    accuracy=float(accuracy),
                    limit=int(limit) if limit is not None else None,
                    mtime=os.path.getmtime(result_path),
                )
            )
    if candidates:
        return max(candidates, key=lambda result: (result.limit is None, result.mtime))
    # results dir exists but no readable JSON yet — still running
    return "running"


def fmt_acc(acc):
    """Format accuracy value for display."""
    if acc is None:
        return ""
    if acc == "running":
        return "running"
    if isinstance(acc, MmluResult):
        suffix = f" (limit={acc.limit})" if acc.limit is not None else ""
        return f"{acc.accuracy:.4f}{suffix}"
    return f"{acc:.4f}"


def fmt_running_timing(eval_info):
    """Format phase-aware timing without presenting a phase ETA as overall ETA."""
    elapsed_s = eval_info.get("elapsed_s")
    progress = eval_info.get("progress")
    if not isinstance(progress, EvalProgress):
        progress = EvalProgress(phase="Initializing")

    phase_detail = progress.phase
    if progress.pct is not None:
        phase_detail += f" {progress.pct}%"
    if progress.current is not None and progress.total is not None:
        phase_detail += f" ({progress.current}/{progress.total})"

    timing = f"phase: {phase_detail}  total elapsed {fmt_time(elapsed_s)}"
    if progress.phase_elapsed_s is not None:
        timing += f"  phase elapsed {fmt_time(progress.phase_elapsed_s)}"
    if progress.phase_remaining_s is not None:
        timing += f"  phase remaining ~{fmt_time(progress.phase_remaining_s)}"
    return timing


# --- main ---

if len(sys.argv) == 3 and sys.argv[1] == "--log-file":
    if sys.argv[2] == "-":
        log_text = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    else:
        with open(sys.argv[2], "rb") as log_file:
            log_text = log_file.read().decode("utf-8", errors="replace")
    log_progress = parse_eval_progress(log_text)
    print(
        f"phase={log_progress.phase} pct={log_progress.pct} "
        f"current={log_progress.current} total={log_progress.total} "
        f"phase_elapsed={fmt_time(log_progress.phase_elapsed_s)} "
        f"phase_remaining={fmt_time(log_progress.phase_remaining_s)}"
    )
    sys.exit(0)

puzzle_dir = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None

if puzzle_dir is None:
    candidates = sorted(glob.glob("puzzle_dir_*") + glob.glob("../puzzle_dir_*"))
    candidates += sorted(glob.glob("/workspace/puzzle_dir_*"))
    seen: set = set()
    deduped = []
    for c in candidates:
        abs_c = os.path.abspath(c)
        if abs_c not in seen:
            seen.add(abs_c)
            deduped.append(c)
    candidates = [c for c in deduped if os.path.isdir(c)]
    if len(candidates) == 1:
        puzzle_dir = candidates[0]
    elif len(candidates) == 0:
        print("No puzzle_dir_* found. Specify: /puzzletron eval progress <puzzle_dir>")
        sys.exit(1)
    else:
        print("Multiple puzzle directories found. Please specify one:")
        for i, c in enumerate(candidates):
            print(f"  {i}  {c}")
        print("\nUsage: /puzzletron eval progress <puzzle_dir>")
        sys.exit(1)

entries = get_checkpoints(puzzle_dir)
if not entries:
    print(f"No checkpoints found under {puzzle_dir}.")
    sys.exit(1)

running_evals = get_running_evals()

done = []
running = []
pending = []
for label, path in entries:
    acc = get_mmlu_accuracy(path)
    abs_path = os.path.abspath(path)
    has_process = any(os.path.abspath(p) == abs_path for p in running_evals)
    if has_process:
        running.append((label, path))
    elif acc not in (None, "running"):
        done.append((label, path))
    else:
        pending.append((label, path))

DIV = "─" * 80
print(f"\nMMlu eval progress  ({len(done)}/{len(entries)} done)")
print(DIV)
print(f"  {'Status':<10}  {'Checkpoint':<14}  {'MMLU acc':>20}  Path")
print(DIV)
for label, path in entries:
    acc = get_mmlu_accuracy(path)
    abs_path = os.path.abspath(path)
    eval_info = next((v for p, v in running_evals.items() if os.path.abspath(p) == abs_path), None)
    if eval_info is not None:
        status = "[RUNNING]"
        acc_str = "..."
        timing_str = f"  {fmt_running_timing(eval_info)}"
    elif acc not in (None, "running"):
        status = "[DONE]"
        acc_str = fmt_acc(acc)
        timing_str = ""
    elif acc == "running":
        status = "[ ]"
        acc_str = "pending"
        timing_str = ""
    else:
        status = "[ ]"
        acc_str = "pending"
        timing_str = ""
    print(f"  {status:<10}  {label:<14}  {acc_str:>20}  {path}")
    if timing_str:
        print(f"  {'':10}  {'':14}  {timing_str}")
print(DIV)
print(f"  Done:    {len(done)}/{len(entries)}")
if running:
    print(f"  Running: {', '.join(lbl for lbl, _ in running)}")
if pending:
    print(f"  Pending: {', '.join(lbl for lbl, _ in pending)}")
