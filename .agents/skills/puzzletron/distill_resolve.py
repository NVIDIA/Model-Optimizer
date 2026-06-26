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
"""Resolve distillation paths for a given puzzle_dir and compression ratio.

Usage:
  python distill_resolve.py [puzzle_dir] [--ratio <r>]

Outputs shell-eval-able KEY=value lines:
  STUDENT_PATH=...
  TEACHER_PATH=...
  OUTPUT_DIR=...
  HF_EXPORT_PATH=...
  RATIO_LABEL=...

Exits non-zero with a human-readable error if resolution fails.
"""

import csv
import glob
import os
import re
import sys


def find_puzzle_dir():
    """Return deduplicated list of existing puzzle_dir_* candidates."""
    candidates = sorted(glob.glob("puzzle_dir_*") + glob.glob("../puzzle_dir_*"))
    candidates += sorted(glob.glob("/workspace/puzzle_dir_*"))
    seen: set = set()
    deduped = []
    for c in candidates:
        abs_c = os.path.abspath(c)
        if abs_c not in seen:
            seen.add(abs_c)
            deduped.append(c)
    return [c for c in deduped if os.path.isdir(c)]


def find_teacher(puzzle_dir):
    """Return the teacher HF model path from YAML configs, or None."""
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


def find_student_path(puzzle_dir, target_memory_mib):
    """Find the solution_0 checkpoint dir matching target_memory_mib."""
    # Convert float to the underscore-encoded directory name fragment
    # e.g. 18349.855224609375 -> "18349_855224609375"
    mib_str = f"{target_memory_mib:.12g}"
    mib_encoded = mib_str.replace(".", "_")

    pattern = f"{puzzle_dir}/mip/puzzle_solutions/target_memory_{mib_encoded}MiB*/solutions--checkpoints/solution_0"
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Fallback: match by integer part
    int_part = str(int(target_memory_mib))
    pattern2 = f"{puzzle_dir}/mip/puzzle_solutions/target_memory_{int_part}*MiB*/solutions--checkpoints/solution_0"
    matches2 = glob.glob(pattern2)
    if matches2:
        return matches2[0]

    return None


def parse_args(argv):
    """Parse [puzzle_dir] [--ratio <r>] from argv."""
    puzzle_dir = None
    ratio = None
    i = 0
    while i < len(argv):
        if argv[i] == "--ratio" and i + 1 < len(argv):
            ratio = argv[i + 1]
            i += 2
        elif not argv[i].startswith("--"):
            puzzle_dir = argv[i]
            i += 1
        else:
            i += 1
    return puzzle_dir, ratio


def error(msg):
    """Print msg to stderr and exit with code 1."""
    print(msg, file=sys.stderr)
    sys.exit(1)


# --- main ---

puzzle_dir, ratio_str = parse_args(sys.argv[1:])

# Resolve puzzle_dir
if puzzle_dir is None:
    candidates = find_puzzle_dir()
    if len(candidates) == 0:
        error("No puzzle_dir_* found. Specify: distill_resolve.py <puzzle_dir>")
    elif len(candidates) > 1:
        error(
            "Multiple puzzle directories found. Please specify one:\n"
            + "\n".join(f"  {c}" for c in candidates)
        )
    puzzle_dir = candidates[0]

if not os.path.isdir(puzzle_dir):
    error(f"puzzle_dir not found: {puzzle_dir}")

# Read sweep CSV
sweep_csv = os.path.join(puzzle_dir, "mip_sweep_results.csv")
if not os.path.isfile(sweep_csv):
    error(f"No sweep results found at {sweep_csv}. Run /puzzletron mip sweep first.")

with open(sweep_csv) as f:
    rows = [r for r in csv.DictReader(f) if r.get("compression_rate")]

if not rows:
    error("Sweep CSV is empty.")

available = {float(r["compression_rate"]): r for r in rows}

if ratio_str is None:
    if len(available) == 1:
        ratio = next(iter(available.keys()))
    else:
        rates = sorted(available.keys())
        error(
            "Multiple compression ratios available. Specify one with --ratio:\n"
            + "\n".join(f"  {r:.2f}" for r in rates)
        )
else:
    try:
        ratio = float(ratio_str)
    except ValueError:
        error(f"Invalid --ratio value: {ratio_str!r}  (expected a float like 0.9)")
    if ratio not in available:
        rates = sorted(available.keys())
        error(
            f"Ratio {ratio} not found in sweep results. Available:\n"
            + "\n".join(f"  {r:.2f}" for r in rates)
        )

row = available[ratio]
target_memory_mib = float(row["target_memory_mib"])

# Find student checkpoint path
student_path = find_student_path(puzzle_dir, target_memory_mib)
if student_path is None:
    error(
        f"Could not find student checkpoint for ratio {ratio} "
        f"(target_memory={target_memory_mib} MiB) under {puzzle_dir}/mip/puzzle_solutions/"
    )

# Find teacher path
teacher_path = find_teacher(puzzle_dir)
if teacher_path is None:
    error(f"Could not find teacher HF model path in YAML configs for puzzle_dir {puzzle_dir}.")

# Build output dirs
ratio_label = f"{ratio:.1f}x".replace("1.0x", "1.0x")
output_dir = os.path.join(puzzle_dir, "distillation", ratio_label)
hf_export_path = os.path.join(output_dir, "hf")

# Output shell-eval-able lines
print(f"STUDENT_PATH={student_path}")
print(f"TEACHER_PATH={teacher_path}")
print(f"OUTPUT_DIR={output_dir}")
print(f"HF_EXPORT_PATH={hf_export_path}")
print(f"RATIO_LABEL={ratio_label}")
