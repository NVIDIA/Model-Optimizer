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


now = datetime.now().replace(microsecond=0)

rates_match = re.search(r"Compression rates: \[(.*?)\]", text)
all_rates = [norm(r.strip()) for r in rates_match.group(1).split(",")] if rates_match else []

# Detect completion via step 8 marker or sweep.py:292
complete_ts = None
for line in lines:
    ts = get_ts(line)
    if ts and ("Results written to:" in line or "Puzzletron Progress 8/8" in line):
        complete_ts = ts
        break

cbc_matches = re.findall(r"After (\d+) nodes.*?\(([\d.]+) seconds\)", text)

# ── Sweep disabled: single MIP solve ─────────────────────────────────────────
if not all_rates:
    step7_ts = None
    for line in lines:
        ts = get_ts(line)
        if ts and "Puzzletron Progress 7/8" in line:
            step7_ts = ts
            break

    end_ts = complete_ts or now
    total_elapsed = int((end_ts - step7_ts).total_seconds()) if step7_ts else 0

    cbc_detail = ""
    if cbc_matches:
        nodes, secs = cbc_matches[-1]
        cbc_detail = f" ({int(nodes):,} nodes, {float(secs):.1f}s)"

    DIV = "─" * 62
    print("\nOverall: Puzzletron step 7/8 — MIP solve (sweep disabled)")
    print(DIV)
    print(f"  {'Status':<10}  {'Phase':<32}  {'Elapsed':>8}")
    print(DIV)
    print(f"  {'[DONE]':<10}  {'Prep (loading model + scores)':<32}  {'<1s':>8}")
    status = "[DONE]" if complete_ts else "[RUNNING]"
    label = f"MIP solve{cbc_detail}"
    print(f"  {status:<10}  {label:<32}  {fmt(total_elapsed):>8}")
    print(DIV)
    finished_str = (
        complete_ts.strftime("%H:%M:%S")
        if complete_ts
        else now.strftime("%H:%M:%S") + " (in progress)"
    )
    print(f"  Started:   {step7_ts.strftime('%H:%M:%S') if step7_ts else '—'}")
    print(f"  Finished:  {finished_str}")
    print(f"  Elapsed:   {fmt(total_elapsed)}")
    print(f"  Remaining: {'done' if complete_ts else 'calculating...'}")
    results_match = re.search(r"Results written to: (\S+)", text)
    if not results_match:
        results_match = re.search(r"\[run_puzzle\.py:335\]\s+(\S+)", text)
    if results_match:
        print(f"\n  Results:   {results_match.group(1)}")
    sys.exit(0)

# ── Sweep enabled: per-rate progress ─────────────────────────────────────────
rate_start = {}
for line in lines:
    m = re.search(r"compression_rate=([\d.]+)", line)
    if m:
        r = norm(m.group(1))
        if r in all_rates and r not in rate_start:
            rate_start[r] = get_ts(line)

sweep_start = rate_start.get(all_rates[0]) if all_rates else None

rate_done = set()
for i, r in enumerate(all_rates[:-1]):
    if all_rates[i + 1] in rate_start:
        rate_done.add(r)
last = all_rates[-1] if all_rates else None
if complete_ts and last and last in rate_start:
    rate_done.add(last)

rate_elapsed = {}
for i, r in enumerate(all_rates):
    if r not in rate_start:
        continue
    end = rate_start[all_rates[i + 1]] if i + 1 < len(all_rates) else (complete_ts or now)
    rate_elapsed[r] = int((end - rate_start[r]).total_seconds())

running_rate = next((r for r in all_rates if r in rate_start and r not in rate_done), None)

cur_detail = ""
if running_rate:
    batch_matches = re.findall(r"calculate_losses_pipeline[^:]*:\s*(\d+)%.*?(\d+)/(\d+)", text)
    if batch_matches:
        pct, cur, total = batch_matches[-1]
        cur_detail = f" — validating ({cur}/{total} batches)"
    elif cbc_matches:
        nodes, secs = cbc_matches[-1]
        cur_detail = f" — MIP solver ({int(nodes):,} nodes, {float(secs):.1f}s)"

end_ts = complete_ts or now
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
print(f"  {'[DONE]':<10}  {'Prep (teacher memory + rate list)':<32}  {'<1s':>8}")
for r in all_rates:
    if r not in rate_start:
        print(f"  {'[ ]':<10}  {f'compression_rate={r}':<32}  {'pending':>8}")
    elif r == running_rate:
        print(
            f"  {'[RUNNING]':<10}  {f'compression_rate={r}{cur_detail}':<32}  {fmt(rate_elapsed.get(r)):>8}"
        )
    else:
        print(f"  {'[DONE]':<10}  {f'compression_rate={r}':<32}  {fmt(rate_elapsed.get(r)):>8}")
print(DIV)
finished_str = (
    complete_ts.strftime("%H:%M:%S") if complete_ts else now.strftime("%H:%M:%S") + " (in progress)"
)
print(f"  Started:   {sweep_start.strftime('%H:%M:%S') if sweep_start else '—'}")
print(f"  Finished:  {finished_str}")
print(f"  Elapsed:   {fmt(total_elapsed)}")
print(f"  Completed: {done_count}/{len(all_rates)} compression rates")
print(f"  Remaining: {est_rem} estimated")
results_match = re.search(r"Results written to: (\S+)", text)
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
