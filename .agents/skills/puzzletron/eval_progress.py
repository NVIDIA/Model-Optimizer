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


def get_running_checkpoint():
    """Return the checkpoint path currently being evaluated by lm_eval, or None."""
    try:
        result = subprocess.run(  # nosec B603 B607
            ["ps", "-ww", "aux"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "lm_eval_hf.py" in line and "pretrained=" in line:
                m = re.search(r"pretrained=(/[^,\s]+)", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None


def get_mmlu_accuracy(path):
    """Return overall MMLU accuracy from saved JSON results, or None if not done."""
    results_dir = os.path.join(path, "eval_results", "mmlu")
    if not os.path.isdir(results_dir):
        return None
    # lm_eval saves results under a subdirectory named after the model path
    # (slashes replaced by __), then results_<timestamp>.json inside it
    for fname in glob.glob(f"{results_dir}/**/results_*.json", recursive=True):
        data = None
        with contextlib.suppress(OSError, json.JSONDecodeError, KeyError):
            data = json.load(open(fname))
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

running_ckpt = get_running_checkpoint()

done = []
running = []
pending = []
for label, path in entries:
    acc = get_mmlu_accuracy(path)
    if acc not in (None, "running"):
        done.append((label, path))
    elif acc == "running" or (
        running_ckpt and os.path.abspath(path) == os.path.abspath(running_ckpt)
    ):
        running.append((label, path))
    else:
        pending.append((label, path))

DIV = "─" * 66
print(f"\nMMlu eval progress  ({len(done)}/{len(entries)} done)")
print(DIV)
print(f"  {'Status':<10}  {'Checkpoint':<14}  {'MMLU acc':>9}  Path")
print(DIV)
for label, path in entries:
    acc = get_mmlu_accuracy(path)
    is_running = (acc == "running") or (
        running_ckpt and os.path.abspath(path) == os.path.abspath(running_ckpt)
    )
    if acc not in (None, "running"):
        status = "[DONE]"
        acc_str = f"{acc:.4f}"
    elif is_running:
        status = "[RUNNING]"
        acc_str = "..."
    else:
        status = "[ ]"
        acc_str = "pending"
    print(f"  {status:<10}  {label:<14}  {acc_str:>9}  {path}")
print(DIV)
print(f"  Done:    {len(done)}/{len(entries)}")
if running:
    print(f"  Running: {', '.join(lbl for lbl, _ in running)}")
if pending:
    print(f"  Pending: {', '.join(lbl for lbl, _ in pending)}")
