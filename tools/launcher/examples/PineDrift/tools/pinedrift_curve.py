#!/usr/bin/env python3
"""Pull the DFlash2 training curve out of a run and sanity-check its shape.

Usage: pinedrift_curve.py [cicd_id]   (default: the id in ~/.pinedrift_dflash2_last)

Prefers the newest checkpoint's trainer_state.json: it carries the complete
log_history with real global_step values, survives a preemption restart (which
writes a NEW log-<jobid>.out and would otherwise hide the pre-restart curve), and
does not depend on the log formatter. Falls back to parsing the logs when no
checkpoint has been written yet.
"""
import ast
import glob
import json
import os
import re
import sys

JOBDIR = os.path.expanduser("~/lustre/pinedrift-experiments/cicd")
STATE = os.path.expanduser("~/.pinedrift_dflash2_last")
cicd = sys.argv[1] if len(sys.argv) > 1 else open(STATE).read().split()[0]


def norm(d):
    """train_acc is logged per parallel head ('train_acc/parallel_0_step_0')."""
    out = {}
    for k, v in d.items():
        kk = "train_acc" if k.startswith("train_acc") else k
        try:
            out[kk] = float(v)
        except (TypeError, ValueError):
            out[kk] = v
    return out


# trainer_state.json has authoritative step numbers and survives restarts, but the
# custom accuracy metric is NOT in its log_history (only loss/grad_norm/lr/epoch).
# The logs do carry train_acc, and every log dict carries an ABSOLUTE `epoch`, so
# step = epoch / num_train_epochs * max_steps recovers the true step even across a
# preemption restart. Parse the logs, take the scale from trainer_state.
total, n_ep = None, None
ckpts = sorted(
    glob.glob(f"{JOBDIR}/{cicd}/dflash2/checkpoint-*"),
    key=lambda p: int(p.rsplit("-", 1)[1]),
)
if ckpts:
    st = json.load(open(f"{ckpts[-1]}/trainer_state.json"))
    total, n_ep = st.get("max_steps"), st.get("num_train_epochs")

logs = sorted(glob.glob(f"{JOBDIR}/{cicd}/*/log-*.out"), key=os.path.getmtime)
by_step = {}
for log in logs:
    for line in open(log, errors="replace"):
        if "'loss'" not in line:
            continue
        m = re.search(r"\{[^{}]*'loss'[^{}]*\}", line)
        if not m:
            continue
        try:
            d = norm(ast.literal_eval(m.group(0)))
        except Exception:
            continue
        if "loss" not in d:
            continue
        if total and n_ep and "epoch" in d:
            d["step"] = int(round(d["epoch"] / n_ep * total))
        else:
            d["step"] = (len(by_step) + 1) * 20
        by_step[d["step"]] = d          # a restart replays a few steps; last wins
pts = [by_step[k] for k in sorted(by_step)]
source = f"{len(logs)} log(s)" + (f" + {os.path.basename(ckpts[-1])}/trainer_state.json" if ckpts else "")

if not pts:
    raise SystemExit(f"{cicd}: no logged steps yet")

steps = [p["step"] for p in pts]
keys = [k for k in ("loss", "train_acc", "grad_norm", "learning_rate", "epoch") if k in pts[0]]
print(f"source  : {source}")
print(f"points  : {len(pts)}   steps {steps[0]}..{steps[-1]}"
      + (f" of {total} ({steps[-1] / total * 100:.1f}% of 5 epochs)" if total else ""))
print()
hdr = "  step  " + "".join(f"{k:>13}" for k in keys)
print(hdr); print("  " + "-" * (len(hdr) - 2))
for i, d in enumerate(pts):
    if i < 3 or i >= len(pts) - 3 or i % max(1, len(pts) // 18) == 0:
        row = f"  {steps[i]:>6}"
        for k in keys:
            v = d.get(k)
            row += f"{v:>13.5g}" if isinstance(v, float) else f"{str(v):>13}"
        print(row)


def plot(name, xs, vals, height=14, width=74):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1e-9
    print(f"\n  {name}   [{lo:.4g} .. {hi:.4g}]")
    grid = [[" "] * width for _ in range(height)]
    n = len(vals)
    for x in range(width):
        j = int(x * (n - 1) / max(1, width - 1))
        y = int((vals[j] - lo) / (hi - lo) * (height - 1))
        grid[height - 1 - y][x] = "*"
    for r, row in enumerate(grid):
        print(f"  {hi - (hi - lo) * r / (height - 1):9.4g} |{''.join(row)}")
    print(f"  {'':9} +{'-' * width}")
    print(f"  {'':9}  step {xs[0]:<{width - 14}}{xs[-1]}")


for k in ("loss", "train_acc"):
    if k in pts[0]:
        plot(k, steps, [p[k] for p in pts])

print("\n  ---- checks ----")
loss = [p["loss"] for p in pts]
n = len(loss)
w = max(1, n // 10)
hm, tm = sum(loss[:w]) / w, sum(loss[-w:]) / w
print(f"  loss  first10% {hm:.4f} -> last10% {tm:.4f}   ({(tm - hm) / hm * 100:+.1f}%)")
if any(v != v for v in loss):
    print("  FAIL: NaN in loss")
elif tm >= hm:
    print("  WARN: loss not decreasing")
else:
    print("  ok: loss decreasing")
if "train_acc" in pts[0]:
    acc = [p["train_acc"] for p in pts]
    ah, at = sum(acc[:w]) / w, sum(acc[-w:]) / w
    print(f"  acc   first10% {ah:.4f} -> last10% {at:.4f}   ({at - ah:+.4f})")
    if max(acc) > 0.999:
        print("  FAIL: train_acc ~1.0 -- empty loss mask, not convergence (README trap 1)")
    elif at <= ah:
        print("  WARN: accuracy not improving")
    else:
        print("  ok: accuracy improving")
if "grad_norm" in pts[0]:
    # Skip warmup: the pre-warmup grad norm is legitimately ~30x the settled median.
    skip = min(10, n // 4)
    g = [p["grad_norm"] for p in pts[skip:]] or [p["grad_norm"] for p in pts]
    med = sorted(g)[len(g) // 2]
    big = [(steps[i + skip], v) for i, v in enumerate(g) if v > 25 * med]
    print(f"  grad  median {med:.3g}, max {max(g):.3g}, "
          + (f"{len(big)} spikes >25x median (first at step {big[0][0]})" if big
             else "no >25x spikes after warmup"))
