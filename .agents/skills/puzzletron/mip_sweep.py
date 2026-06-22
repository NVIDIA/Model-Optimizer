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
"""Display MIP sweep results across all compression rates from the sweep CSV."""

import contextlib
import csv
import os
import re
import sys

LOG = "./log.txt"
try:
    text = open(LOG).read()
except FileNotFoundError:
    print("No log.txt found. Run /puzzletron all first.")
    sys.exit(1)

match = re.search(r"(\S+)/mip/puzzle_solutions/", text)
if not match:
    match = re.search(r"(\S+)/ckpts/teacher", text)
if not match:
    print("Could not find puzzle_dir in log.txt.")
    sys.exit(1)

puzzle_dir = match.group(1)
sweep_csv = os.path.join(puzzle_dir, "mip_sweep_results.csv")

if not os.path.exists(sweep_csv):
    print(f"No sweep results found at {sweep_csv}.")
    print("Enable sweep in the config YAML and re-run /puzzletron mip <nproc>.")
    sys.exit(1)

with open(sweep_csv) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("Sweep CSV is empty.")
    sys.exit(1)

COLS = [
    ("compression_rate", "rate", 6),
    ("target_memory_mib", "target_mem", 12),
    ("actual_memory_mib", "actual_mem", 12),
    ("num_params", "num_params", 12),
    ("lm_loss", "lm_loss", 8),
    ("token_accuracy_top_1", "top_1", 7),
    ("token_accuracy_top_5", "top_5", 7),
    ("token_accuracy_top_10", "top_10", 8),
]

header = "  ".join(f"{label:>{w}}" for _, label, w in COLS)
divider = "-" * len(header)
print(f"\n{header}")
print(divider)
for row in rows:
    line_parts = []
    for key, _, w in COLS:
        val = row.get(key, "n/a")
        with contextlib.suppress(ValueError, TypeError):
            val = f"{float(val):.4f}" if "." in val else f"{int(val):,}"
        line_parts.append(f"{val!s:>{w}}")
    print("  ".join(line_parts))
print()
print(f"Results from: {sweep_csv}")
