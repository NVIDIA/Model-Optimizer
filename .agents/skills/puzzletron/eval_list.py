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
"""List available checkpoints for MMLU eval.

Usage:
  python eval_list.py [puzzle_dir]

If puzzle_dir is omitted, scans for puzzle_dir_* directories under the repo
root. If exactly one is found it is used automatically; if multiple are found
they are listed and the script exits asking the user to specify one.

Output columns (tab-separated):
  #    Label        MMLU    Path

MMLU column shows "done" if eval_results/mmlu/ already exists.
"""

import glob
import os
import re
import sys


def parse_memory_mib(dir_name):
    """Parse memory in MiB from a target_memory_<val>MiB directory name.

    Directory names encode decimal values with underscores replacing the decimal
    point, e.g. target_memory_10194_364013671875MiB -> 10194.364013671875 MiB.
    Round numbers like target_memory_10000MiB have no underscore.
    """
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


def list_checkpoints(puzzle_dir):
    """Print available checkpoints (teacher + sweep solutions + distillation) with MMLU eval status."""
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

    if not entries:
        print(f"No checkpoints found under {puzzle_dir}.")
        sys.exit(1)

    def has_mmlu(path):
        return os.path.isdir(os.path.join(path, "eval_results", "mmlu"))

    max_label = max(len(e[0]) for e in entries)
    print(f"\n{'#':<4}  {'Label':<{max_label}}  {'MMLU':^6}  Path")
    print("-" * (4 + 2 + max_label + 2 + 6 + 2 + 60))
    for i, (label, path) in enumerate(entries):
        eval_mark = "done" if has_mmlu(path) else ""
        print(f"{i:<4}  {label:<{max_label}}  {eval_mark:^6}  {path}")
    print()
    print("Usage: /puzzletron eval mmlu <index>")
    print("       /puzzletron eval mmlu <index> --limit 10   (smoke test)")


# --- main ---

if len(sys.argv) > 1:
    puzzle_dir = sys.argv[1].rstrip("/")
    if not os.path.isdir(puzzle_dir):
        print(f"Directory not found: {puzzle_dir}")
        sys.exit(1)
    list_checkpoints(puzzle_dir)
else:
    # Auto-discover puzzle_dir_* under the repo root
    candidates = sorted(glob.glob("puzzle_dir_*") + glob.glob("../puzzle_dir_*"))
    # Also check /workspace if we're inside it
    candidates += sorted(glob.glob("/workspace/puzzle_dir_*"))
    # Deduplicate while preserving order
    seen: set = set()
    deduped = []
    for c in candidates:
        abs_c = os.path.abspath(c)
        if abs_c not in seen:
            seen.add(abs_c)
            deduped.append(c)
    candidates = deduped
    candidates = [c for c in candidates if os.path.isdir(c)]

    if len(candidates) == 1:
        list_checkpoints(candidates[0])
    elif len(candidates) == 0:
        print("No puzzle_dir_* directories found. Please specify the path explicitly:")
        print("  /puzzletron eval list <puzzle_dir>")
        sys.exit(1)
    else:
        print("Multiple puzzle directories found. Please specify one:")
        for i, c in enumerate(candidates):
            print(f"  {i}  {c}")
        print("\nUsage: /puzzletron eval list <puzzle_dir>")
        sys.exit(1)
