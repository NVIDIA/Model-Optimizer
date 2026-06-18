---
name: puzzletron
description: "End-to-end workflow for model pruning and MIP-based optimization. Use `mip_sweep` to run the MIP sweep. Usage: /puzzletron <command>"
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1 — Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If the first word of args does **not exactly match** `mip_sweep`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron** — end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `mip_sweep <nproc_per_node>` — Run the MIP sweep (nproc_per_node: number of GPUs per node)
- `mip_sweep progress` — Show live MIP sweep progress with timing summary

Usage: `/puzzletron <command> [args]`

---

**STEP 2 — Only if the first word of args exactly matches a command name, execute it. Never reach this step if args were empty.**

## Command: mip_sweep

Parse the second word of args.

- If no second word is provided, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the second word is exactly `progress`, execute the **mip_sweep progress** sub-command below.
- Otherwise treat the second word as `nproc_per_node` and run the sweep.

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
print(f"  Completed: {done_count}/{len(all_rates)} compression rates", end="")
print(f"  (avg {fmt(avg_s)}/rate)" if avg_s else "")
print(f"  Remaining: {est_rem} estimated")
results_match = re.search(r'Results written to: (\S+)', text)
if results_match:
    print(f"\n  Results:   {results_match.group(1)}")
PYEOF
```
