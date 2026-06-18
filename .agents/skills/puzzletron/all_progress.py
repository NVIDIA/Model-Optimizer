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

LOG = "./log.txt"
try:
    lines = open(LOG).readlines()
    text = "".join(lines)
except FileNotFoundError:
    print("No log.txt found. Run /puzzletron all first.")
    sys.exit(0)


def norm(r):
    """Normalize a compression rate to a canonical float string."""
    return str(float(r))


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
for line in lines:
    ts = get_ts(line)
    if ts and "sweep.py:292" in line:
        pipeline_complete_ts = ts
        break

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
if sol_done is not None and sol_total:
    cur_detail = f" ({sol_done}/{sol_total} solutions)"
elif batch_matches:
    pct, cur_b, total_b = batch_matches[-1]
    cur_detail = f" ({cur_b}/{total_b} batches)"
elif cbc_matches:
    nodes, secs = cbc_matches[-1]
    cur_detail = f" (MIP solver: {int(nodes):,} nodes, {float(secs):.1f}s)"

rates_match = re.search(r"Compression rates: \[(.*?)\]", text)
all_rates = [norm(r.strip()) for r in rates_match.group(1).split(",")] if rates_match else []
rate_start = {}
for line in lines:
    if "sweep.py:258" in line:
        m = re.search(r"compression_rate=([\d.]+)", line)
        if m:
            r = norm(m.group(1))
            if r in all_rates and r not in rate_start:
                rate_start[r] = get_ts(line)
rate_done = set()
for i, r in enumerate(all_rates[:-1]):
    if all_rates[i + 1] in rate_start:
        rate_done.add(r)
if pipeline_complete_ts and all_rates and all_rates[-1] in rate_start:
    rate_done.add(all_rates[-1])

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
    elif batch_matches and int(cur_b) > 0 and int(cur_b) < int(total_b):
        rate_per_batch = cur_step_elapsed / int(cur_b)
        step_remaining = rate_per_batch * (int(total_b) - int(cur_b))
    elif all_rates and last_step_num == 7:
        done_count_r = len(rate_done)
        remaining_count_r = len(all_rates) - done_count_r
        rate_elapsed = {}
        for i, r in enumerate(all_rates):
            if r not in rate_start:
                continue
            if i + 1 < len(all_rates) and all_rates[i + 1] in rate_start:
                rate_elapsed[r] = int(
                    (rate_start[all_rates[i + 1]] - rate_start[r]).total_seconds()
                )
        avg_r = sum(rate_elapsed.values()) / len(rate_elapsed) if rate_elapsed else None
        if avg_r and remaining_count_r:
            step_remaining = avg_r * remaining_count_r

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
        if snum == 7 and all_rates:
            detail = f" ({len(rate_done)}/{len(all_rates)} rates done)"
    label = f"{snum}/{total_steps}: {sdesc}{detail}"
    status = "[DONE]" if is_done else "[RUNNING]"
    print(
        f"  {status:<10}  {'':<4}  {label:<34}  {fmt(elapsed) if elapsed is not None else '—':>8}"
    )

for snum in range(last_step_num + 1, total_steps + 1):
    print(f"  {'[ ]':<10}  {'':<4}  {f'{snum}/{total_steps}: pending':<34}  {'':>8}")

print(DIV)
if all_rates and last_step_num >= 7:
    print(f"  MIP rates: {len(rate_done)}/{len(all_rates)} done", end="")
    running_rate = next((r for r in all_rates if r in rate_start and r not in rate_done), None)
    if running_rate:
        print(f"  (running: {running_rate})", end="")
    print()

done_steps = len([s for s in seen_steps if s != last_step_num or pipeline_complete_ts])
step_durations = []
for i, (snum, (sdesc, sts)) in enumerate(step_ts_list):
    next_ts = (
        step_ts_list[i + 1][1][1] if i + 1 < len(step_ts_list) else (pipeline_complete_ts or None)
    )
    if next_ts and sts:
        step_durations.append(int((next_ts - sts).total_seconds()))
avg_step_s = sum(step_durations) / len(step_durations) if step_durations else None

CONFIG_PATH = (
    "examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml"
)
sweep_enabled = True
sweep_n_rates = 6
try:
    cfg_text = open(CONFIG_PATH).read()
    _en_m = re.search(r"sweep:\s*\n\s+enabled:\s*(true|false)", cfg_text)
    if _en_m:
        sweep_enabled = _en_m.group(1) == "true"
    _rates_m = re.search(r"memory_compression_rates:\s*\[([^\]]+)\]", cfg_text)
    if _rates_m:
        sweep_n_rates = len(_rates_m.group(1).split(","))
except Exception:
    pass
effective_n_rates = len(all_rates) if all_rates else sweep_n_rates
RATE_S = 250  # ~4m 10s per compression rate (historical)


def step_est(snum):
    """Estimate duration in seconds for a pending pipeline step."""
    if snum == 7:
        return (RATE_S * effective_n_rates) if sweep_enabled else 120
    elif snum == 8:
        return 60
    return avg_step_s or 0


if pipeline_complete_ts:
    est_rem = "done"
elif step_remaining is not None:
    future_s = sum(step_est(s) for s in range(last_step_num + 1, total_steps + 1))
    est_rem = fmt(step_remaining + future_s)
else:
    cur_s = step_est(last_step_num)
    future_s = cur_s + sum(step_est(s) for s in range(last_step_num + 1, total_steps + 1))
    est_rem = fmt(future_s) if (cur_s or future_s) else "calculating..."

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
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
