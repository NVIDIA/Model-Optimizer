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
"""Progress report for the full Puzzletron pipeline (all 8 steps)."""

import glob
import re
import sys
from datetime import datetime

puzzle_dir = sys.argv[1] if len(sys.argv) > 1 else None
if not puzzle_dir:
    print("Usage: all_progress.py <puzzle_dir>")
    sys.exit(1)
LOG = f"{puzzle_dir}/log.txt"
try:
    lines = open(LOG).readlines()
    text = "".join(lines)
except FileNotFoundError:
    print(f"No log.txt found at {LOG}. Run /puzzletron all first.")
    sys.exit(0)


def fmt(s):
    """Format seconds as 'Xm Ys', or '—' if None."""
    return f"{int(s) // 60}m {int(s) % 60}s" if s is not None else "—"


def get_ts(line):
    """Extract a datetime from a log line timestamp, or None."""
    m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


now = datetime.now().replace(microsecond=0)
DIV = "─" * 68

step_events = []
for line in lines:
    m = re.search(r"Puzzletron Progress (\d+)/(\d+): (.+)", line)
    if m:
        step_num = int(m.group(1))
        total_steps = int(m.group(2))
        desc = m.group(3).strip()
        ts = get_ts(line)
        step_events.append((step_num, total_steps, desc, ts))

total_steps = step_events[-1][1] if step_events else 8
seen_steps = {e[0]: (e[2], e[3]) for e in step_events}
last_step_num = max(seen_steps.keys()) if seen_steps else 0

pipeline_complete_ts = None
if last_step_num == total_steps and total_steps in seen_steps:
    pipeline_complete_ts = seen_steps[total_steps][1]

cur_detail = ""
step_remaining = None
batch_matches = re.findall(r"calculate_losses_pipeline[^:]*:\s*(\d+)%.*?(\d+)/(\d+)", text)
cbc_matches = re.findall(r"After (\d+) nodes.*?\(([\d.]+) seconds\)", text)

sol_dir_match = re.search(
    r"'output_dir': '([^']+single_sequence_replacement_solutions--validation[^']*)'", text
)
sol_done, sol_total = None, None
if sol_dir_match:
    sol_dir = sol_dir_match.group(1)
    sol_done = len(glob.glob(f"{sol_dir}/solution*.json"))
    sol_list_match = re.search(r"'solutions_to_validate': \[([\d, ]+)\]", text)
    if sol_list_match:
        sol_total = len(sol_list_match.group(1).split(","))
pct, cur_b, total_b = batch_matches[-1] if batch_matches else (None, None, None)
if sol_done is not None and sol_total:
    cur_detail = f" ({sol_done}/{sol_total} solutions)"
elif batch_matches:
    cur_detail = f" ({cur_b}/{total_b} batches)"
elif cbc_matches:
    nodes, secs = cbc_matches[-1]
    cur_detail = f" (MIP solver: {int(nodes):,} nodes, {float(secs):.1f}s)"

pipeline_start = step_events[0][3] if step_events else None
end_ts = pipeline_complete_ts or now
total_elapsed = int((end_ts - pipeline_start).total_seconds()) if pipeline_start else 0

step_ts_list = sorted(seen_steps.items())
cur_step_start_ts = seen_steps[last_step_num][1] if last_step_num in seen_steps else None
if not pipeline_complete_ts and cur_step_start_ts:
    cur_step_elapsed = int((now - cur_step_start_ts).total_seconds())
    if sol_done and sol_total and sol_done > 0:
        rate_per_sol = cur_step_elapsed / sol_done
        step_remaining = rate_per_sol * (sol_total - sol_done)
    elif cur_b is not None and total_b is not None and int(cur_b) > 0 and int(cur_b) < int(total_b):
        rate_per_batch = cur_step_elapsed / int(cur_b)
        step_remaining = rate_per_batch * (int(total_b) - int(cur_b))

print(f"\nOverall: Puzzletron full pipeline (steps 1–{total_steps})")  # noqa: RUF001
print(DIV)
print(f"  {'Status':<10}  {'Step':<4}  {'Description':<34}  {'Elapsed':>8}")
print(DIV)

for i, (snum, (sdesc, sts)) in enumerate(step_ts_list):
    next_ts = (
        step_ts_list[i + 1][1][1] if i + 1 < len(step_ts_list) else (pipeline_complete_ts or now)
    )
    elapsed = int((next_ts - sts).total_seconds()) if sts and next_ts else None
    is_last = snum == last_step_num
    is_done = not is_last or pipeline_complete_ts is not None
    detail = ""
    if is_last and not is_done:
        detail = cur_detail
    label = f"{snum}/{total_steps}: {sdesc}{detail}"
    status = "[DONE]" if is_done else "[RUNNING]"
    print(
        f"  {status:<10}  {'':<4}  {label:<34}  {fmt(elapsed) if elapsed is not None else '—':>8}"
    )

_STEP_NAMES = {
    1: "starting puzzletron pipeline",
    2: "converting model to Puzzletron heterogeneous format (single-gpu)",
    3: "scoring pruning activations (multi-gpu)",
    4: "pruning the model and saving pruned checkpoints (single-gpu)",
    5: "building replacement library and subblock statistics (single-gpu)",
    6: "calculating one block scores (multi-gpu)",
    7: "running MIP and realizing models (multi-gpu)",
    8: "puzzletron pipeline completed (multi-gpu)",
}

for snum in range(last_step_num + 1, total_steps + 1):
    desc = _STEP_NAMES.get(snum, "pending")
    print(f"  {'[ ]':<10}  {'':<4}  {f'{snum}/{total_steps}: {desc}':<34}  {'':>8}")

print(DIV)
done_steps = len([s for s in seen_steps if s != last_step_num or pipeline_complete_ts])

# Per-step baseline durations (seconds) measured on Qwen3-8B with 8 GPUs.
# Step 1 is excluded from scaling because it's always trivially fast (~1s).
# Update these values as real runs complete to keep estimates accurate.
_BASELINE_S = {
    1: 1,  # starting pipeline
    2: 29,  # model conversion (single-gpu)   — measured: Qwen3-8B/8GPU
    3: 128,  # activation scoring (multi-gpu)  — measured: Qwen3-8B/8GPU
    4: 58,  # pruning & saving (single-gpu)   — measured: Qwen3-8B/8GPU
    5: 15,  # replacement library (single-gpu) — measured: Qwen3-8B/8GPU
    6: 3187,  # one block scores (multi-gpu)    — measured: Qwen3-8B/8GPU (53m7s)
    7: 315,  # MIP solve                       — measured: Qwen3-8B/8GPU (5m15s)
    8: 1,  # completion / realisation        — measured: Qwen3-8B/8GPU (~0s)
}

# Scale factor: ratio of actual vs. baseline for completed non-trivial steps.
# Starts at 1.0 (trust baselines); updates as steps finish.
_scale_samples = []
for i, (snum, (sdesc, sts)) in enumerate(step_ts_list):
    next_ts = (
        step_ts_list[i + 1][1][1] if i + 1 < len(step_ts_list) else (pipeline_complete_ts or None)
    )
    if next_ts and sts and snum > 1 and snum in _BASELINE_S:
        actual = int((next_ts - sts).total_seconds())
        baseline = _BASELINE_S[snum]
        if baseline > 0:
            _scale_samples.append(actual / baseline)
_scale = sum(_scale_samples) / len(_scale_samples) if _scale_samples else 1.0


def step_est(snum):
    """Estimate duration (seconds) for a pipeline step using scaled baselines."""
    return int(_BASELINE_S.get(snum, 600) * _scale)


# If no sub-step progress is available for the current step, estimate remaining
# time from the baseline minus elapsed.
if not pipeline_complete_ts and cur_step_start_ts and step_remaining is None:
    cur_step_elapsed = int((now - cur_step_start_ts).total_seconds())
    step_remaining = max(0, step_est(last_step_num) - cur_step_elapsed)

if pipeline_complete_ts:
    est_rem = "done"
elif step_remaining is not None:
    future_s = sum(step_est(s) for s in range(last_step_num + 1, total_steps + 1))
    est_rem = fmt(step_remaining + future_s)
else:
    future_s = sum(step_est(s) for s in range(last_step_num, total_steps + 1))
    est_rem = fmt(future_s) if future_s else "calculating..."

finished_str = (
    pipeline_complete_ts.strftime("%H:%M:%S")
    if pipeline_complete_ts
    else now.strftime("%H:%M:%S") + " (in progress)"
)
print(f"  Started:   {pipeline_start.strftime('%H:%M:%S') if pipeline_start else '—'}")
print(f"  Finished:  {finished_str}")
print(f"  Elapsed:   {fmt(total_elapsed)}")
print(f"  Completed: {done_steps}/{total_steps} steps")
print(f"  Remaining: {est_rem} estimated")
results_match = re.search(r"Results written to: (\S+)", text)
if not results_match:
    results_match = re.search(r"\[run_puzzle\.py:335\]\s+(\S+)", text)
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
