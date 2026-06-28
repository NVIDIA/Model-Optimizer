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
"""Teacher vs. compressed model accuracy for the single constrained MIP solution.

For sweep results across multiple compression rates use mip_sweep.py instead.
"""

import csv
import glob
import json
import os
import re
import sys

KEYS = ["lm_loss", "token_accuracy_top_1", "token_accuracy_top_5", "token_accuracy_top_10"]

puzzle_dir = sys.argv[1] if len(sys.argv) > 1 else None
if not puzzle_dir:
    print("Usage: mip_losses.py <puzzle_dir>")
    sys.exit(1)
LOG = f"{puzzle_dir}/log.txt"
try:
    text = open(LOG).read()
except FileNotFoundError:
    print(f"No log.txt found at {LOG}. Run /puzzletron all first.")
    sys.exit(1)

# Extract the configured target_memory from the log (logged in the args dict)
target_mem_match = re.search(r"'target_memory':\s*([\d.]+)", text)
configured_target = float(target_mem_match.group(1)) if target_mem_match else None

# Find all validation directories
solutions_dirs = sorted(glob.glob(f"{puzzle_dir}/mip/puzzle_solutions/*/solutions--validation"))
if not solutions_dirs:
    print(f"No MIP validation results found under {puzzle_dir}. Has the pipeline completed?")
    sys.exit(1)

# Prefer the directory whose name matches the configured target_memory (the constrained run),
# not the sweep directories which have the teacher memory as target.
chosen_dir = None
if configured_target is not None:
    target_str = str(int(configured_target))
    for d in solutions_dirs:
        if f"target_memory_{target_str}" in d and "num_params" in d:
            chosen_dir = d
            break

if chosen_dir is None:
    # Fall back to the first (smallest target = most compressed)
    chosen_dir = solutions_dirs[0]

solutions_dir = chosen_dir

# Get teacher memory from sweep CSV if available, else from dir name
teacher_memory_mib = None
sweep_csv = os.path.join(puzzle_dir, "mip_sweep_results.csv")
if os.path.exists(sweep_csv):
    with open(sweep_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            teacher_memory_mib = float(row["teacher_memory_mib"])
            break

# Get target_memory from the chosen dir name
dir_name = os.path.basename(os.path.dirname(solutions_dir))
mem_match = re.search(r"target_memory_([\d_]+)MiB", dir_name)
target_memory_mib = float(mem_match.group(1).replace("_", "")) if mem_match else configured_target

results = {}
for name in ["teacher", "solution_0"]:
    path = os.path.join(solutions_dir, f"{name}.json")
    if not os.path.exists(path):
        print(f"{name}.json not found at {path}")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    results[name] = {k: round(data[k]["avg"], 4) for k in KEYS if k in data}

col_w = 30
print(f"\n{'Metric':<{col_w}}  {'Teacher':>10}  {'Compressed (solution_0)':>24}")
print("-" * (col_w + 38))
mem_teacher = f"{teacher_memory_mib:,.0f} MiB" if teacher_memory_mib else "n/a"
mem_solution = f"{target_memory_mib:,.0f} MiB" if target_memory_mib else "n/a"
print(f"{'target_memory':<{col_w}}  {mem_teacher:>10}  {mem_solution:>24}")
print("-" * (col_w + 38))
for k in KEYS:
    teacher_val = results["teacher"].get(k, "n/a")
    student_val = results["solution_0"].get(k, "n/a")
    print(f"{k:<{col_w}}  {teacher_val!s:>10}  {student_val!s:>24}")
print()
print(f"Results from: {solutions_dir}")
if os.path.exists(sweep_csv):
    print("Sweep results: use /puzzletron mip sweep losses")
