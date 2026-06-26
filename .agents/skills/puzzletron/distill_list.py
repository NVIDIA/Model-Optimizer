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
"""List distillation runs under puzzle_dir/distillation/.

Usage:
  python distill_list.py [puzzle_dir]

Shows each ratio subdirectory with:
  - Status: not started / in-progress (iter N / total) / done (HF exported)
  - Output dir path
"""

import contextlib
import glob
import os
import sys


def find_puzzle_dir_candidates():
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


def latest_iter(output_dir):
    """Return the latest checkpoint iteration number, or None."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    iters = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("iter_"):
            with contextlib.suppress(IndexError, ValueError):
                iters.append(int(name.split("_")[1]))
    return max(iters) if iters else None


def get_status(output_dir):
    """Return (status_str, detail_str) for a distillation run dir."""
    hf_path = os.path.join(output_dir, "hf")
    if os.path.isdir(hf_path) and any(
        f.endswith((".safetensors", ".bin", "config.json")) for f in os.listdir(hf_path)
    ):
        return "DONE", f"HF exported → {hf_path}"

    itr = latest_iter(output_dir)
    if itr is not None:
        return "IN-PROGRESS", f"latest iter: {itr}"

    if os.path.isdir(output_dir):
        return "STARTED", "no checkpoint yet"

    return "NOT STARTED", ""


def list_distillation_runs(puzzle_dir):
    """Print a table of distillation runs found under puzzle_dir/distillation/."""
    distill_base = os.path.join(puzzle_dir, "distillation")

    if not os.path.isdir(distill_base):
        print(f"No distillation runs found under {puzzle_dir}/distillation/")
        print("Run: /puzzletron distill run [--puzzle_dir <dir>] [--ratio <r>]")
        return

    runs = sorted(
        d for d in os.listdir(distill_base) if os.path.isdir(os.path.join(distill_base, d))
    )

    if not runs:
        print(f"No distillation runs found under {distill_base}/")
        print("Run: /puzzletron distill run [--puzzle_dir <dir>] [--ratio <r>]")
        return

    div = "─" * 72
    print(f"\nDistillation runs under {distill_base}/")
    print(div)
    print(f"  {'Ratio':<10}  {'Status':<14}  Detail")
    print(div)
    for run in runs:
        output_dir = os.path.join(distill_base, run)
        status, detail = get_status(output_dir)
        print(f"  {run:<10}  {status:<14}  {detail}")
    print(div)
    print()
    print(
        "To check training progress: /puzzletron distill progress [--puzzle_dir <dir>] [--ratio <r>]"
    )


# --- main ---

puzzle_dir = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None

if puzzle_dir is None:
    candidates = find_puzzle_dir_candidates()
    if len(candidates) == 1:
        puzzle_dir = candidates[0]
    elif len(candidates) == 0:
        print("No puzzle_dir_* found. Specify: /puzzletron distill list <puzzle_dir>")
        sys.exit(1)
    else:
        print("Multiple puzzle directories found. Please specify one:")
        for c in candidates:
            print(f"  {c}")
        print("\nUsage: /puzzletron distill list <puzzle_dir>")
        sys.exit(1)

list_distillation_runs(puzzle_dir)
