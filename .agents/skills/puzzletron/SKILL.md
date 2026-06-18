---
name: puzzletron
description: "End-to-end workflow for model pruning and MIP-based optimization. Use `mip_sweep` to run the MIP sweep. Usage: /puzzletron <command>"
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1 — Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If the first word of args does **not exactly match** `mip_sweep` or `all`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron** — end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `mip_sweep <nproc_per_node>` — Run the MIP sweep (nproc_per_node: number of GPUs per node)
- `mip_sweep progress` — Show live MIP sweep progress with timing summary
- `all <nproc_per_node>` — Run the full Puzzletron pipeline (nproc_per_node: number of GPUs per node)
- `all progress` — Show live full pipeline progress with timing summary

Usage: `/puzzletron <command> [args]`

---

**STEP 2 — Only if the first word of args exactly matches a command name, execute it. Never reach this step if args were empty.**

## Command: all

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `all 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `all --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **all progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- Otherwise use the parsed value and run the full pipeline.

### all \<nproc_per_node\>

Run the following Bash command, substituting `<nproc_per_node>` with the parsed value:

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### all progress

Run the following Python script verbatim. Do not modify it. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 - << 'PYEOF'
import re, sys
from datetime import datetime

LOG = './log.txt'
try:
    lines = open(LOG).readlines()
    text = ''.join(lines)
except FileNotFoundError:
    print("No log.txt found. Run /puzzletron all first.")
    sys.exit(0)

def norm(r): return str(float(r))
def fmt(s): return f"{int(s)//60}m {int(s)%60}s" if s is not None else "—"
def get_ts(line):
    m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None

now = datetime.now().replace(microsecond=0)
DIV = '─' * 68

# Parse pipeline steps from "Puzzletron Progress X/8: <desc>" lines
step_events = []
for line in lines:
    m = re.search(r'Puzzletron Progress (\d+)/(\d+): (.+)', line)
    if m:
        step_num = int(m.group(1))
        total_steps = int(m.group(2))
        desc = m.group(3).strip()
        ts = get_ts(line)
        step_events.append((step_num, total_steps, desc, ts))

total_steps = step_events[-1][1] if step_events else 8
seen_steps = {e[0]: (e[2], e[3]) for e in step_events}
last_step_num = max(seen_steps.keys()) if seen_steps else 0

# Determine if pipeline is fully complete
pipeline_complete_ts = None
for line in lines:
    ts = get_ts(line)
    if ts and 'sweep.py:292' in line:
        pipeline_complete_ts = ts
        break

# Sub-step detail and remaining estimate for current step
cur_detail = ""
step_remaining = None
batch_matches = re.findall(r'calculate_losses_pipeline[^:]*:\s*(\d+)%.*?(\d+)/(\d+)', text)
cbc_matches = re.findall(r'After (\d+) nodes.*?\(([\d.]+) seconds\)', text)
# Step 6: count completed solution files for real progress
import glob as _glob, os as _os
sol_dir_match = re.search(r"'output_dir': '([^']+single_sequence_replacement_solutions--validation[^']*)'", text)
sol_done, sol_total = None, None
if sol_dir_match:
    sol_dir = sol_dir_match.group(1)
    sol_files = _glob.glob(f"{sol_dir}/solution*.json")
    sol_done = len(sol_files)
    sol_list_match = re.search(r"'solutions_to_validate': \[([\d, ]+)\]", text)
    if sol_list_match:
        sol_total = len(sol_list_match.group(1).split(','))
if sol_done is not None and sol_total:
    cur_detail = f" ({sol_done}/{sol_total} solutions)"
elif batch_matches:
    pct, cur_b, total_b = batch_matches[-1]
    cur_detail = f" ({cur_b}/{total_b} batches)"
elif cbc_matches:
    nodes, secs = cbc_matches[-1]
    cur_detail = f" (MIP solver: {int(nodes):,} nodes, {float(secs):.1f}s)"

# MIP sweep compression rate detail (step 7)
rates_match = re.search(r'Compression rates: \[(.*?)\]', text)
all_rates = [norm(r.strip()) for r in rates_match.group(1).split(',')] if rates_match else []
rate_start = {}
for line in lines:
    if 'sweep.py:258' in line:
        m = re.search(r'compression_rate=([\d.]+)', line)
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

# Overall timing
pipeline_start = step_events[0][3] if step_events else None
end_ts = pipeline_complete_ts if pipeline_complete_ts else now
total_elapsed = int((end_ts - pipeline_start).total_seconds()) if pipeline_start else 0

# Estimate remaining time for current running step
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
            if i + 1 < len(all_rates) and all_rates[i+1] in rate_start:
                rate_elapsed[r] = int((rate_start[all_rates[i+1]] - rate_start[r]).total_seconds())
        avg_r = sum(rate_elapsed.values()) / len(rate_elapsed) if rate_elapsed else None
        if avg_r and remaining_count_r:
            step_remaining = avg_r * remaining_count_r

print(f"\nOverall: Puzzletron full pipeline (steps 1–{total_steps})")
print(DIV)
print(f"  {'Status':<10}  {'Step':<4}  {'Description':<34}  {'Elapsed':>8}")
print(DIV)

for i, (snum, (sdesc, sts)) in enumerate(step_ts_list):
    next_ts = step_ts_list[i+1][1][1] if i+1 < len(step_ts_list) else (pipeline_complete_ts if pipeline_complete_ts else now)
    elapsed = int((next_ts - sts).total_seconds()) if sts and next_ts else None
    is_last = (snum == last_step_num)
    is_done = not is_last or pipeline_complete_ts is not None
    detail = ""
    if is_last and not is_done:
        detail = cur_detail
        if snum == 7 and all_rates:
            detail = f" ({len(rate_done)}/{len(all_rates)} rates done)"
    label = f"{snum}/{total_steps}: {sdesc}{detail}"
    status = "[DONE]" if is_done else "[RUNNING]"
    print(f"  {status:<10}  {'':<4}  {label:<34}  {fmt(elapsed) if elapsed is not None else '—':>8}")

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
avg_step_s = None
step_durations = []
for i, (snum, (sdesc, sts)) in enumerate(step_ts_list):
    next_ts = step_ts_list[i+1][1][1] if i+1 < len(step_ts_list) else (pipeline_complete_ts if pipeline_complete_ts else None)
    if next_ts and sts:
        step_durations.append(int((next_ts - sts).total_seconds()))
if step_durations:
    avg_step_s = sum(step_durations) / len(step_durations)

# Parse config for sweep settings to estimate step 7 duration
CONFIG_PATH = 'examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml'
sweep_enabled = True
sweep_n_rates = 6
try:
    cfg_text = open(CONFIG_PATH).read()
    _en_m = re.search(r'sweep:\s*\n\s+enabled:\s*(true|false)', cfg_text)
    if _en_m:
        sweep_enabled = _en_m.group(1) == 'true'
    _rates_m = re.search(r'memory_compression_rates:\s*\[([^\]]+)\]', cfg_text)
    if _rates_m:
        sweep_n_rates = len(_rates_m.group(1).split(','))
except Exception:
    pass
# Prefer actual rate count from log if step 7 has already started
effective_n_rates = len(all_rates) if all_rates else sweep_n_rates
RATE_S = 250  # ~4m 10s per compression rate (historical)

def step_est(snum):
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

print(f"  Started:   {pipeline_start.strftime('%H:%M:%S') if pipeline_start else '—'}")
print(f"  Finished:  {pipeline_complete_ts.strftime('%H:%M:%S') if pipeline_complete_ts else now.strftime('%H:%M:%S') + ' (in progress)'}")
print(f"  Elapsed:   {fmt(total_elapsed)}")
print(f"  Completed: {done_steps}/{total_steps} steps")
print(f"  Remaining: {est_rem} estimated")
results_match = re.search(r'Results written to: (\S+)', text)
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
PYEOF
```

## Command: mip_sweep

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `mip_sweep 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `mip_sweep --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **mip_sweep progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- Otherwise use the parsed value and run the sweep.

### mip_sweep \<nproc_per_node\>

Run the following Bash command, substituting `<nproc_per_node>` with the parsed value:

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  --mip-only 2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### mip_sweep progress

Run the following Python script verbatim. Do not modify it. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 - << 'PYEOF'
import re, sys
from datetime import datetime

LOG = './log.txt'
try:
    lines = open(LOG).readlines()
    text = ''.join(lines)
except FileNotFoundError:
    print("No log.txt found. Run /puzzletron mip_sweep first.")
    sys.exit(0)

def norm(r): return str(float(r))
def fmt(s): return f"{int(s)//60}m {int(s)%60}s" if s is not None else "—"
def get_ts(line):
    m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None

rates_match = re.search(r'Compression rates: \[(.*?)\]', text)
all_rates = [norm(r.strip()) for r in rates_match.group(1).split(',')] if rates_match else []

# Collect start timestamp per rate
rate_start = {}
for line in lines:
    if 'sweep.py:258' in line:
        m = re.search(r'compression_rate=([\d.]+)', line)
        if m:
            r = norm(m.group(1))
            if r in all_rates and r not in rate_start:
                rate_start[r] = get_ts(line)

now = datetime.now().replace(microsecond=0)
sweep_start = rate_start.get(all_rates[0]) if all_rates else None

# Rate is done when the next rate has started; last rate done when sweep.py:287 appears
rate_done = set()
for i, r in enumerate(all_rates[:-1]):
    if all_rates[i + 1] in rate_start:
        rate_done.add(r)
last = all_rates[-1]
sweep_complete_ts = None
for line in lines:
    ts = get_ts(line)
    if ts and 'sweep.py:292' in line:
        sweep_complete_ts = ts
        break
if sweep_complete_ts and last in rate_start:
    rate_done.add(last)

# Per-rate elapsed = next rate start - this rate start (or completion ts or now for last)
rate_elapsed = {}
for i, r in enumerate(all_rates):
    if r not in rate_start:
        continue
    if i + 1 < len(all_rates):
        end = rate_start[all_rates[i + 1]]
    else:
        end = sweep_complete_ts if sweep_complete_ts else now
    rate_elapsed[r] = int((end - rate_start[r]).total_seconds())

# Currently running rate
running_rate = next((r for r in all_rates if r in rate_start and r not in rate_done), None)

# Sub-step detail for running rate
cur_detail = ""
if running_rate:
    batch_matches = re.findall(r'calculate_losses_pipeline[^:]*:\s*(\d+)%.*?(\d+)/(\d+)', text)
    cbc_matches = re.findall(r'After (\d+) nodes.*?\(([\d.]+) seconds\)', text)
    if batch_matches:
        pct, cur, total = batch_matches[-1]
        cur_detail = f" — validating ({cur}/{total} batches)"
    elif cbc_matches:
        nodes, secs = cbc_matches[-1]
        cur_detail = f" — MIP solver ({int(nodes):,} nodes, {float(secs):.1f}s)"

end_ts = sweep_complete_ts if sweep_complete_ts else now
total_elapsed = int((end_ts - sweep_start).total_seconds()) if sweep_start else 0

done_count = len(rate_done)
remaining_count = len(all_rates) - done_count
avg_s = sum(rate_elapsed[r] for r in rate_done) / done_count if done_count else None
est_rem = fmt(avg_s * remaining_count) if avg_s and remaining_count else ("done" if not remaining_count else "calculating...")

DIV = '─' * 62

print(f"\nOverall: Puzzletron step 7/8 — MIP sweep ({len(all_rates)} compression rates)")
print(DIV)
print(f"  {'Status':<10}  {'Phase':<32}  {'Elapsed':>8}")
print(DIV)
print(f"  [DONE]      {'Prep (teacher memory + rate list)':<32}  {'<1s':>8}")
for r in all_rates:
    if r not in rate_start:
        print(f"  [ ]         {f'compression_rate={r}':<32}  {'pending':>8}")
    elif r == running_rate:
        detail = cur_detail
        print(f"  [RUNNING]   {f'compression_rate={r}{detail}':<32}  {fmt(rate_elapsed.get(r)):>8}")
    else:
        print(f"  [DONE]      {f'compression_rate={r}':<32}  {fmt(rate_elapsed.get(r)):>8}")
print(DIV)
print(f"  Started:   {sweep_start.strftime('%H:%M:%S') if sweep_start else '—'}")
print(f"  Finished:  {sweep_complete_ts.strftime('%H:%M:%S') if sweep_complete_ts else now.strftime('%H:%M:%S') + ' (in progress)'}")
print(f"  Elapsed:   {fmt(total_elapsed)}")
print(f"  Completed: {done_count}/{len(all_rates)} compression rates")
print(f"  Remaining: {est_rem} estimated")
results_match = re.search(r'Results written to: (\S+)', text)
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
PYEOF
```
