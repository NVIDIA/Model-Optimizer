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
"""Progress report for the Puzzletron MIP step."""

import re
import sys
from datetime import datetime

LOG = "./log.txt"
try:
    lines = open(LOG).readlines()
    text = "".join(lines)
except FileNotFoundError:
    print("No log.txt found. Run /puzzletron mip first.")
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

now = datetime.now().replace(microsecond=0)
sweep_start = rate_start.get(all_rates[0]) if all_rates else None

rate_done = set()
for i, r in enumerate(all_rates[:-1]):
    if all_rates[i + 1] in rate_start:
        rate_done.add(r)
last = all_rates[-1] if all_rates else None
sweep_complete_ts = None
for line in lines:
    ts = get_ts(line)
    if ts and "sweep.py:292" in line:
        sweep_complete_ts = ts
        break
if sweep_complete_ts and last and last in rate_start:
    rate_done.add(last)

rate_elapsed = {}
for i, r in enumerate(all_rates):
    if r not in rate_start:
        continue
    if i + 1 < len(all_rates):
        end = rate_start[all_rates[i + 1]]
    else:
        end = sweep_complete_ts or now
    rate_elapsed[r] = int((end - rate_start[r]).total_seconds())

running_rate = next((r for r in all_rates if r in rate_start and r not in rate_done), None)

cur_detail = ""
if running_rate:
    batch_matches = re.findall(r"calculate_losses_pipeline[^:]*:\s*(\d+)%.*?(\d+)/(\d+)", text)
    cbc_matches = re.findall(r"After (\d+) nodes.*?\(([\d.]+) seconds\)", text)
    if batch_matches:
        pct, cur, total = batch_matches[-1]
        cur_detail = f" — validating ({cur}/{total} batches)"
    elif cbc_matches:
        nodes, secs = cbc_matches[-1]
        cur_detail = f" — MIP solver ({int(nodes):,} nodes, {float(secs):.1f}s)"

end_ts = sweep_complete_ts or now
total_elapsed = int((end_ts - sweep_start).total_seconds()) if sweep_start else 0

done_count = len(rate_done)
remaining_count = len(all_rates) - done_count
avg_s = sum(rate_elapsed[r] for r in rate_done) / done_count if done_count else None
est_rem = (
    fmt(avg_s * remaining_count)
    if avg_s and remaining_count
    else ("done" if not remaining_count else "calculating...")
)

DIV = "─" * 62

print(f"\nOverall: Puzzletron step 7/8 — MIP sweep ({len(all_rates)} compression rates)")
print(DIV)
print(f"  {'Status':<10}  {'Phase':<32}  {'Elapsed':>8}")
print(DIV)
print(f"  [DONE]      {'Prep (teacher memory + rate list)':<32}  {'<1s':>8}")
for r in all_rates:
    if r not in rate_start:
        print(f"  [ ]         {f'compression_rate={r}':<32}  {'pending':>8}")
    elif r == running_rate:
        print(
            f"  [RUNNING]   {f'compression_rate={r}{cur_detail}':<32}  {fmt(rate_elapsed.get(r)):>8}"
        )
    else:
        print(f"  [DONE]      {f'compression_rate={r}':<32}  {fmt(rate_elapsed.get(r)):>8}")
print(DIV)
finished_str = (
    sweep_complete_ts.strftime("%H:%M:%S")
    if sweep_complete_ts
    else now.strftime("%H:%M:%S") + " (in progress)"
)
print(f"  Started:   {sweep_start.strftime('%H:%M:%S') if sweep_start else '—'}")
print(f"  Finished:  {finished_str}")
print(f"  Elapsed:   {fmt(total_elapsed)}")
print(f"  Completed: {done_count}/{len(all_rates)} compression rates")
print(f"  Remaining: {est_rem} estimated")
results_match = re.search(r"Results written to: (\S+)", text)
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
