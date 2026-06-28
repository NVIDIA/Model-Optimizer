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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    """Return dict of checkpoint_path -> {pid, elapsed_s, pct_done} for running lm_eval processes."""
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
            pct = None
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
            for log_path in log_candidates:
                try:
                    with open(log_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        tail = f.read().decode("utf-8", errors="replace")
                    for tqdm_m in re.finditer(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", tail):
                        pct = int(tqdm_m.group(1))
                    if pct is not None:
                        break
                except OSError:
                    continue
            results[path] = {"pid": pid, "elapsed_s": elapsed, "pct": pct}
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


# Rough baseline for full MMLU eval duration in seconds (measured: Qwen3-8B/8GPU/bs4).
# Used to estimate remaining time during the model-loading phase before tqdm starts.
# Update with measured values as more runs complete.
_MMLU_EVAL_BASELINE_S = 2700  # ~45 min, measured Qwen3-8B/8GPU/batch_size=4


def get_mmlu_accuracy(path):
    """Return overall MMLU accuracy from saved JSON results, or None if not done."""
    results_dir = os.path.join(path, "eval_results", "mmlu")
    if not os.path.isdir(results_dir):
        return None
    # lm_eval saves results under a subdirectory named after the model path
    # (slashes replaced by __), then results_<timestamp>.json inside it.
    # Use os.walk instead of glob — glob skips dirs starting with '.' (e.g.
    # when the model path was relative, lm_eval names the subdir '..__..').
    for dirpath, _dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.startswith("results_") or not fname.endswith(".json"):
                continue
            data = None
            with contextlib.suppress(OSError, json.JSONDecodeError, KeyError):
                data = json.load(open(os.path.join(dirpath, fname)))
            if data is not None:
                results = data.get("results", {})
                if "mmlu" in results:
                    return results["mmlu"].get("acc,none") or results["mmlu"].get("acc")
    # results dir exists but no readable JSON yet — still running
    return "running"


def fmt_acc(acc):
    """Format accuracy value for display."""
    if acc is None:
        return ""
    if acc == "running":
        return "running"
    return f"{acc:.4f}"


# --- main ---

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
queued = []
pending = []
for label, path in entries:
    acc = get_mmlu_accuracy(path)
    abs_path = os.path.abspath(path)
    has_process = any(os.path.abspath(p) == abs_path for p in running_evals)
    if has_process:
        running.append((label, path))
    elif acc not in (None, "running"):
        done.append((label, path))
    elif acc == "running":
        # eval_results dir exists but no active process — queued/waiting
        queued.append((label, path))
    else:
        pending.append((label, path))

DIV = "─" * 66
print(f"\nMMlu eval progress  ({len(done)}/{len(entries)} done)")
print(DIV)
print(f"  {'Status':<10}  {'Checkpoint':<14}  {'MMLU acc':>9}  Path")
print(DIV)
for label, path in entries:
    acc = get_mmlu_accuracy(path)
    abs_path = os.path.abspath(path)
    eval_info = next((v for p, v in running_evals.items() if os.path.abspath(p) == abs_path), None)
    if eval_info is not None:
        status = "[RUNNING]"
        acc_str = "..."
        elapsed_s = eval_info.get("elapsed_s")
        pct = eval_info.get("pct")
        elapsed_str = fmt_time(elapsed_s)
        if pct and pct > 0 and elapsed_s:
            remaining_s = elapsed_s * (100 - pct) / pct
            timing_str = f"  {pct}% done  elapsed {elapsed_str}  remaining ~{fmt_time(remaining_s)}"
        elif elapsed_s:
            # Loading phase: use baseline estimate for remaining
            remaining_s = max(0, _MMLU_EVAL_BASELINE_S - elapsed_s)
            timing_str = (
                f"  loading...  elapsed {elapsed_str}  remaining ~{fmt_time(remaining_s)} est."
            )
        else:
            timing_str = "  loading..."
    elif acc not in (None, "running"):
        status = "[DONE]"
        acc_str = f"{acc:.4f}"
        timing_str = ""
    elif acc == "running":
        status = "[QUEUED]"
        acc_str = "..."
        timing_str = "  waiting for previous eval to finish"
    else:
        status = "[ ]"
        acc_str = "pending"
        timing_str = ""
    print(f"  {status:<10}  {label:<14}  {acc_str:>9}  {path}")
    if timing_str:
        print(f"  {'':10}  {'':14}  {timing_str}")
print(DIV)
print(f"  Done:    {len(done)}/{len(entries)}")
if running:
    print(f"  Running: {', '.join(lbl for lbl, _ in running)}")
if queued:
    print(f"  Queued:  {', '.join(lbl for lbl, _ in queued)}")
if pending:
    print(f"  Pending: {', '.join(lbl for lbl, _ in pending)}")
