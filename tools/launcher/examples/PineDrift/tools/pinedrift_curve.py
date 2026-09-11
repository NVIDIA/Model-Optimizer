#!/usr/bin/env python3
"""Pull the DFlash2 training curve out of a run's log and sanity-check its shape.

Usage: pinedrift_curve.py [cicd_id]   (default: the id in ~/.pinedrift_dflash2_last)

Prints the raw points, an ASCII plot, and a verdict on the failure modes that
matter for this harness -- see the checks at the bottom for what each one means.
"""
import ast
import glob
import os
import re
import sys

JOBDIR = os.path.expanduser("~/lustre/pinedrift-experiments/cicd")
STATE = os.path.expanduser("~/.pinedrift_dflash2_last")

cicd = sys.argv[1] if len(sys.argv) > 1 else open(STATE).read().split()[0]
logs = sorted(glob.glob(f"{JOBDIR}/{cicd}/*/log-*.out"), key=os.path.getmtime)
if not logs:
    raise SystemExit(f"no log under {JOBDIR}/{cicd}")
log = logs[-1]

# HF Trainer logs a plain dict per logging_steps; DFlash adds train_acc/plosses.
pts = []
for line in open(log, errors="replace"):
    if "'loss'" not in line and '"loss"' not in line:
        continue
    m = re.search(r"\{[^{}]*'loss'[^{}]*\}", line)
    if not m:
        continue
    try:
        d = ast.literal_eval(m.group(0))
    except Exception:
        continue
    if isinstance(d, dict) and "loss" in d:
        # values arrive as strings in this trainer's formatter; and the accuracy key
        # is per-parallel-head ("train_acc/parallel_0_step_0"), not a bare "train_acc".
        out = {}
        for k, v in d.items():
            kk = "train_acc" if k.startswith("train_acc") else k
            try:
                out[kk] = float(v)
            except (TypeError, ValueError):
                out[kk] = v
        pts.append(out)

if not pts:
    print(f"log: {log}")
    print("no logged steps yet")
    raise SystemExit(0)

keys = [k for k in ("loss", "train_acc", "grad_norm", "learning_rate", "epoch") if k in pts[0]]
extra = [k for k in pts[0] if k not in keys and k != "step"]
print(f"log     : {log}")
print(f"points  : {len(pts)}   fields: {sorted(pts[0])}")
print()

hdr = "  step " + "".join(f"{k:>13}" for k in keys)
print(hdr); print("  " + "-" * (len(hdr) - 2))
step = 0
LOGGING_STEPS = 20
for i, d in enumerate(pts):
    step = d.get("step", (i + 1) * LOGGING_STEPS)
    if i < 5 or i >= len(pts) - 5 or i % max(1, len(pts) // 20) == 0:
        row = f"  {step:>5}"
        for k in keys:
            v = d[k]
            row += f"{v:>13.5g}" if isinstance(v, float) else f"{v:>13}"
        print(row)


def plot(name, vals, height=14, width=76):
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
        tick = hi - (hi - lo) * r / (height - 1)
        print(f"  {tick:9.4g} |{''.join(row)}")
    print(f"  {'':9} +{'-' * width}")
    print(f"  {'':9}  step {0:<{width - 12}}{(len(vals)) * LOGGING_STEPS}")


for k in ("loss", "train_acc"):
    if k in pts[0]:
        plot(k, [p[k] for p in pts])

print("\n  ---- checks ----")
loss = [p["loss"] for p in pts]
acc = [p.get("train_acc") for p in pts] if "train_acc" in pts[0] else None
n = len(loss)
head, tail = loss[: max(1, n // 10)], loss[-max(1, n // 10) :]
hm, tm = sum(head) / len(head), sum(tail) / len(tail)
print(f"  loss  first10% {hm:.4f} -> last10% {tm:.4f}   ({(tm - hm) / hm * 100:+.1f}%)")
if any(v != v for v in loss):
    print("  FAIL: NaN in loss")
elif tm >= hm:
    print("  WARN: loss not decreasing")
else:
    print("  ok: loss decreasing")
if acc and all(a is not None for a in acc):
    am_h = sum(acc[: max(1, n // 10)]) / max(1, n // 10)
    am_t = sum(acc[-max(1, n // 10) :]) / max(1, n // 10)
    print(f"  acc   first10% {am_h:.4f} -> last10% {am_t:.4f}   ({am_t - am_h:+.4f})")
    if max(acc) > 0.999:
        print("  FAIL: train_acc hits ~1.0 -- empty loss mask, not convergence "
              "(see README trap 1)")
    elif am_t <= am_h:
        print("  WARN: accuracy not improving")
    else:
        print("  ok: accuracy improving")
if "grad_norm" in pts[0]:
    g = [p["grad_norm"] for p in pts]
    med = sorted(g)[len(g) // 2]
    big = [(i * LOGGING_STEPS, v) for i, v in enumerate(g) if v > 25 * med]
    print(f"  grad_norm median {med:.3g}, max {max(g):.3g}"
          + (f", {len(big)} spikes >25x median (first at step {big[0][0]})" if big else ", no >25x spikes"))
