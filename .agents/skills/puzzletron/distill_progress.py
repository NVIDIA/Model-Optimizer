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
"""Show distillation training progress for a puzzle_dir.

Usage:
  python distill_progress.py [puzzle_dir] [--ratio <r>]

Shows for each matching run:
  - Current iteration / total iterations
  - Elapsed time and estimated remaining time
  - Whether a distillation process is currently running
  - HF export status
  - Last loss value
"""

import contextlib
import glob
import os
import re
import subprocess  # nosec B404
import sys
from datetime import datetime


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


def parse_args(argv):
    """Parse [puzzle_dir] [--ratio <r>] from argv."""
    puzzle_dir = None
    ratio = None
    i = 0
    while i < len(argv):
        if argv[i] == "--ratio" and i + 1 < len(argv):
            ratio = argv[i + 1]
            i += 2
        elif not argv[i].startswith("--"):
            puzzle_dir = argv[i]
            i += 1
        else:
            i += 1
    return puzzle_dir, ratio


def fmt_duration(seconds):
    """Format seconds as 'Xh Ym Zs' or 'Xm Ys'."""
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    return f"{m}m {sec}s"


def get_running_output_dirs():
    """Return set of output_dir paths where distill.py is currently running."""
    running = set()
    try:
        result = subprocess.run(  # nosec B603 B607
            ["ps", "-ww", "aux"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "distill.py" in line and "output_dir" in line:
                m = re.search(r"--output_dir\s+(\S+)", line)
                if m:
                    running.add(os.path.abspath(m.group(1)))
    except Exception:
        pass
    return running


def latest_iter(output_dir):
    """Return (latest_iter, all_iters_sorted) from checkpoints dir."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None, []
    iters = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("iter_"):
            with contextlib.suppress(IndexError, ValueError):
                iters.append(int(name.split("_")[1]))
    iters_sorted = sorted(iters)
    return (max(iters) if iters else None), iters_sorted


def parse_log_timing(output_dir):
    """Parse ./log.txt for iteration timing if it belongs to this output_dir.

    Returns dict with keys: cur_iter, total_iters, avg_iter_ms, started_ts, last_loss.
    """
    log_path = "./log.txt"
    if not os.path.isfile(log_path):
        return {}
    try:
        text = open(log_path).read()
    except OSError:
        return {}

    # Confirm this log belongs to this output_dir
    abs_out = os.path.abspath(output_dir)
    if abs_out not in text:
        return {}

    result = {}

    # Parse: [datetime] iteration N/ M | ... elapsed time per iteration (ms): XXX | ...
    iter_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] iteration\s+(\d+)/\s*(\d+)"
        r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    )
    matches = iter_pattern.findall(text)
    if matches:
        ts_str, cur, total, _ms = matches[-1]
        result["cur_iter"] = int(cur)
        result["total_iters"] = int(total)
        # Average over last 5 iterations for a stable estimate
        recent = matches[-5:]
        result["avg_iter_ms"] = sum(float(m[3]) for m in recent) / len(recent)
        result["last_iter_ts"] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

    # Parse training start time
    m = re.search(
        r"\[after dataloaders are built\] datetime: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text
    )
    if m:
        result["started_ts"] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

    # Parse last loss
    loss_pattern = re.compile(r"\[.*?\] iteration\s+\d+/\s*\d+.*?total loss:\s*([\d.Ee+\-]+)")
    loss_matches = loss_pattern.findall(text)
    if loss_matches:
        result["last_loss"] = float(loss_matches[-1])

    return result


def read_last_loss_tb(output_dir):
    """Try to read the last logged loss from tensorboard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        tb_dir = os.path.join(output_dir, "tensorboard")
        if not os.path.isdir(tb_dir):
            return None
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        loss_tag = next((t for t in tags if "loss" in t.lower() and "kd" in t.lower()), None)
        if loss_tag is None:
            loss_tag = next((t for t in tags if "loss" in t.lower()), None)
        if loss_tag:
            events = ea.Scalars(loss_tag)
            if events:
                last = events[-1]
                return last.value, last.step
    except Exception:
        pass
    return None


def show_progress(puzzle_dir, ratio_filter):
    """Print training progress for distillation runs under puzzle_dir."""
    distill_base = os.path.join(puzzle_dir, "distillation")
    if not os.path.isdir(distill_base):
        print(f"No distillation directory found at {distill_base}")
        return

    runs = sorted(
        d for d in os.listdir(distill_base) if os.path.isdir(os.path.join(distill_base, d))
    )
    if ratio_filter:
        runs = [r for r in runs if r == ratio_filter or r == f"{float(ratio_filter):.1f}x"]

    if not runs:
        msg = "No distillation runs found"
        if ratio_filter:
            msg += f" for ratio {ratio_filter}"
        print(msg + f" under {distill_base}/")
        return

    running_dirs = get_running_output_dirs()
    now = datetime.now().replace(microsecond=0)

    div = "─" * 76
    print(f"\nDistillation progress — {puzzle_dir}")
    print(div)

    for run in runs:
        output_dir = os.path.join(distill_base, run)
        abs_out = os.path.abspath(output_dir)
        is_running = abs_out in running_dirs

        latest, all_iters = latest_iter(output_dir)
        hf_dir = os.path.join(output_dir, "hf")
        hf_done = (
            os.path.isdir(hf_dir)
            and any(f.endswith((".safetensors", ".bin", "config.json")) for f in os.listdir(hf_dir))
            if os.path.isdir(hf_dir)
            else False
        )

        timing = parse_log_timing(output_dir) if is_running else {}

        print(f"\n  Ratio:      {run}")
        print(f"  Output dir: {output_dir}")

        # Status line
        if is_running:
            if timing.get("cur_iter") and timing.get("total_iters"):
                cur = timing["cur_iter"]
                total = timing["total_iters"]
                print(f"  Status:     RUNNING  (iter {cur}/{total})")
            else:
                print("  Status:     RUNNING")
        elif hf_done:
            print("  Status:     DONE (HF exported)")
        elif latest is not None:
            print(f"  Status:     STOPPED (latest checkpoint: iter {latest})")
        else:
            print("  Status:     NOT STARTED")

        # Timing block (running only)
        if is_running and timing:
            started = timing.get("started_ts")
            elapsed_s = int((now - started).total_seconds()) if started else None
            avg_ms = timing.get("avg_iter_ms")
            cur_iter = timing.get("cur_iter", 0)
            total_iters = timing.get("total_iters")
            remaining_iters = (total_iters - cur_iter) if total_iters else None
            remaining_s = (
                (avg_ms / 1000.0 * remaining_iters)
                if (avg_ms and remaining_iters is not None)
                else None
            )

            if started:
                print(f"  Started:    {started.strftime('%H:%M:%S')}")
            if elapsed_s is not None:
                print(f"  Elapsed:    {fmt_duration(elapsed_s)}")
            if avg_ms:
                print(f"  Iter time:  {avg_ms / 1000:.1f}s/iter (avg last 5)")
            if remaining_s is not None:
                print(f"  Remaining:  ~{fmt_duration(remaining_s)} ({remaining_iters} iters left)")

        # Checkpoint iters (stopped/done)
        if latest is not None and not is_running:
            print(f"  Checkpoints: iters {', '.join(str(i) for i in all_iters)}")

        # HF export
        if hf_done:
            print(f"  HF export:  {hf_dir}")
        elif os.path.isdir(hf_dir):
            print("  HF export:  in progress or empty")
        else:
            print("  HF export:  not yet")

        # Loss — prefer log parse (fast), fall back to tensorboard
        loss_val = timing.get("last_loss")
        if loss_val is not None:
            print(f"  Last loss:  {loss_val:.4f} (iter {timing.get('cur_iter', '?')})")
        else:
            tb = read_last_loss_tb(output_dir)
            if tb:
                print(f"  Last loss:  {tb[0]:.4f} (step {tb[1]})")

        # Log file
        log_path = os.path.abspath("./log.txt")
        if os.path.isfile(log_path) and is_running:
            print(f"  Log file:   {log_path}")

    print(f"\n{div}")


# --- main ---

puzzle_dir, ratio = parse_args(sys.argv[1:])

if puzzle_dir is None:
    candidates = find_puzzle_dir_candidates()
    if len(candidates) == 1:
        puzzle_dir = candidates[0]
    elif len(candidates) == 0:
        print("No puzzle_dir_* found. Specify: /puzzletron distill progress <puzzle_dir>")
        sys.exit(1)
    else:
        print("Multiple puzzle directories found. Please specify one:")
        for c in candidates:
            print(f"  {c}")
        print("\nUsage: /puzzletron distill progress <puzzle_dir> [--ratio <r>]")
        sys.exit(1)

show_progress(puzzle_dir, ratio)
