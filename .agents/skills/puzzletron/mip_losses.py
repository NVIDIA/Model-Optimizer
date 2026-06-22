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
"""Teacher vs. compressed model accuracy for the MIP solution."""

import glob
import json
import os
import re
import sys

KEYS = ["lm_loss", "token_accuracy_top_1", "token_accuracy_top_5", "token_accuracy_top_10"]

LOG = "./log.txt"
try:
    text = open(LOG).read()
except FileNotFoundError:
    print("No log.txt found. Run /puzzletron all first.")
    sys.exit(1)

# Extract puzzle_dir from any path that contains /mip/puzzle_solutions/ in the log
match = re.search(r"(\S+)/mip/puzzle_solutions/", text)
if not match:
    # Fall back: look for the ckpts/teacher path logged early in the run
    match = re.search(r"(\S+)/ckpts/teacher", text)
if not match:
    print("Could not find puzzle_dir in log.txt. Has the pipeline run?")
    sys.exit(1)

puzzle_dir = match.group(1)

solutions_dirs = sorted(glob.glob(f"{puzzle_dir}/mip/puzzle_solutions/*/solutions--validation"))
if not solutions_dirs:
    print(f"No MIP validation results found under {puzzle_dir}. Has the pipeline completed?")
    sys.exit(1)

solutions_dir = solutions_dirs[-1]

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
for k in KEYS:
    teacher_val = results["teacher"].get(k, "n/a")
    student_val = results["solution_0"].get(k, "n/a")
    print(f"{k:<{col_w}}  {teacher_val!s:>10}  {student_val!s:>24}")
print()
print(f"Results from: {solutions_dir}")
