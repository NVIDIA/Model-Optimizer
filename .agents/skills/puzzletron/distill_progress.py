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
  - Training/validation objective loss and validation student-CE sparklines
  - Convergence verdict based on validation objective loss
"""

import contextlib
import glob
import os
import re
import subprocess  # nosec B404
import sys
from datetime import datetime

SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"


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


def read_run_config(output_dir):
    """Return dict with dataset names, gbs, seq_len, train_iters from latest checkpoint config."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return {}
    # Find the latest iter_* checkpoint that has a run_config.yaml
    configs = sorted(glob.glob(os.path.join(ckpt_dir, "iter_*/run_config.yaml")), reverse=True)
    if not configs:
        return {}
    try:
        text = open(configs[0]).read()
    except OSError:
        return {}

    result = {}

    # Extract blend paths -> simplified dataset names
    blend_paths = re.findall(r"- /workspace/\S+", text)
    names = []
    for bp in blend_paths:
        name = bp.strip().lstrip("- ")
        basename = os.path.basename(name)
        # Shorten: remove long prefixes/suffixes for readability
        short = re.sub(r"nvidia--", "nvidia/", basename)
        short = re.sub(r"Salesforce--", "Salesforce/", short)
        short = re.sub(r"_text_document$", "", short)
        short = re.sub(r"_messages_max\d+$", "", short)
        names.append(short)
    if names:
        result["datasets"] = names

    m = re.search(r"global_batch_size:\s*(\d+)", text)
    if m:
        result["gbs"] = int(m.group(1))

    m = re.search(r"seq_length:\s*(\d+)", text)
    if m:
        result["seq_len"] = int(m.group(1))

    m = re.search(r"train_iters:\s*(\d+)", text)
    if m:
        result["train_iters"] = int(m.group(1))

    return result


def fmt_tokens(n):
    """Format token count as e.g. '122M', '1.2B', or '328K'."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}K"


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


def parse_log_text(log_path):
    """Read log file, return text or empty string on failure."""
    if not os.path.isfile(log_path):
        return ""
    try:
        return open(log_path).read()
    except OSError:
        return ""


def parse_training_loss_history(text, output_dir):
    """Return list of (iter, loss) from training log lines for this output_dir."""
    if not text:
        return []
    pattern = re.compile(r"\] iteration\s+(\d+)/\s*\d+.*?total loss:\s*([\d.Ee+\-]+)")
    return [(int(m[0]), float(m[1])) for m in pattern.findall(text)]


def parse_validation_loss_history(text, output_dir):
    """Return list of (iter, loss) from validation log lines for this output_dir."""
    if not text:
        return []
    pattern = re.compile(
        r"validation loss at iteration\s+(\d+).*?total loss value:\s*([\d.Ee+\-]+)"
    )
    return list({int(iteration): float(loss) for iteration, loss in pattern.findall(text)}.items())


def parse_validation_ce_history(text, output_dir):
    """Return validation student cross-entropy as (iter, loss) pairs from log lines."""
    if not text:
        return []
    pattern = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([\d.Ee+\-]+)")
    return list({int(iteration): float(loss) for iteration, loss in pattern.findall(text)}.items())


def read_loss_history_tb(output_dir):
    """Return (train_history, val_history) as (iter, loss) lists from tensorboard."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # Megatron saves to tb_logs/, not tensorboard/
        tb_dir = next(
            (
                os.path.join(output_dir, d)
                for d in ("tb_logs", "tensorboard")
                if os.path.isdir(os.path.join(output_dir, d))
            ),
            None,
        )
        if tb_dir is None:
            return [], []
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])

        # Pick the KD / distillation loss tag for training
        train_tag = next(
            (t for t in tags if "loss" in t.lower() and "valid" not in t.lower()),
            None,
        )
        val_tag = next(
            (t for t in tags if "valid" in t.lower() and "loss" in t.lower()),
            None,
        )
        train_hist = [(e.step, e.value) for e in ea.Scalars(train_tag)] if train_tag else []
        val_hist = [(e.step, e.value) for e in ea.Scalars(val_tag)] if val_tag else []
        return train_hist, val_hist
    except Exception:
        return [], []


def parse_log_timing(text, output_dir):
    """Extract timing/progress info from log text for this output_dir.

    Returns dict with keys: cur_iter, total_iters, avg_iter_ms, started_ts, last_loss.
    """
    if not text:
        return {}

    result = {}

    iter_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] iteration\s+(\d+)/\s*(\d+)"
        r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    )
    matches = iter_pattern.findall(text)
    if matches:
        ts_str, cur, total, _ms = matches[-1]
        result["cur_iter"] = int(cur)
        result["total_iters"] = int(total)
        recent = matches[-5:]
        result["avg_iter_ms"] = sum(float(m[3]) for m in recent) / len(recent)
        result["last_iter_ts"] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

    m = re.search(
        r"\[after dataloaders are built\] datetime: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text
    )
    if m:
        result["started_ts"] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

    loss_pattern = re.compile(r"\[.*?\] iteration\s+\d+/\s*\d+.*?total loss:\s*([\d.Ee+\-]+)")
    loss_matches = loss_pattern.findall(text)
    if loss_matches:
        result["last_loss"] = float(loss_matches[-1])

    return result


def bucket_losses(history, bucket_size):
    """Average (iter, loss) pairs into fixed-size iteration buckets."""
    if not history:
        return []
    buckets: dict = {}
    for itr, loss in history:
        b = (itr // bucket_size) * bucket_size + bucket_size
        buckets.setdefault(b, []).append(loss)
    return [(b, sum(v) / len(v)) for b, v in sorted(buckets.items())]


def make_sparkline(history):
    """Build a sparkline string from a list of (iter, loss) pairs.

    Higher loss maps to a taller bar so the curve visually descends as training improves.
    """
    if not history:
        return "—"
    losses = [loss for _, loss in history]
    if len(losses) == 1:
        return SPARKLINE_BLOCKS[4]
    min_l, max_l = min(losses), max(losses)
    if max_l == min_l:
        return SPARKLINE_BLOCKS[4] * len(losses)
    return "".join(SPARKLINE_BLOCKS[int((loss - min_l) / (max_l - min_l) * 7)] for loss in losses)


def convergence_verdict(val_history):
    """Return (status, detail) string pair based on validation loss trend."""
    if len(val_history) < 2:
        return "NOT ENOUGH DATA", f"need ≥2 val checkpoints (have {len(val_history)})"

    losses = [loss for _, loss in val_history]
    k = min(3, len(losses))

    # Divergence: last point higher than k checkpoints ago
    if losses[-1] > losses[-k]:
        pct = (losses[-1] - losses[-k]) / losses[-k] * 100
        return "DIVERGING", f"+{pct:.1f}% over last {k} checkpoints — consider stopping"

    improvement = (losses[-k] - losses[-1]) / losses[-k] * 100

    if improvement > 2.0:
        return "CONVERGING", f"-{improvement:.1f}% over last {k} checkpoints"
    elif improvement > 0.5:
        return "DIMINISHING RETURNS", f"-{improvement:.1f}% over last {k} checkpoints"
    else:
        return "PLATEAU", f"<0.5% change over last {k} checkpoints — safe to stop"


def show_progress(puzzle_dir, ratio_filter):
    """Print training progress for distillation runs under puzzle_dir."""
    distill_base = os.path.join(puzzle_dir, "distillation")
    if not os.path.isdir(distill_base):
        print(f"No distillation directory found at {distill_base}")
        return

    running_dirs = get_running_output_dirs()

    runs = sorted(
        d for d in os.listdir(distill_base) if os.path.isdir(os.path.join(distill_base, d))
    )
    # Also include runs that are currently active but haven't saved a checkpoint yet
    abs_distill_base = os.path.abspath(distill_base)
    existing_abs = {os.path.abspath(os.path.join(distill_base, r)) for r in runs}
    for abs_running in running_dirs:
        if abs_running.startswith(abs_distill_base + os.sep) and abs_running not in existing_abs:
            runs.append(os.path.relpath(abs_running, abs_distill_base))
    runs = sorted(runs)

    if ratio_filter:
        try:
            numeric_form = f"{float(ratio_filter):.1f}x"
        except (ValueError, TypeError):
            numeric_form = None
        runs = [r for r in runs if r == ratio_filter or (numeric_form and r == numeric_form)]

    if not runs:
        msg = "No distillation runs found"
        if ratio_filter:
            msg += f" for ratio {ratio_filter}"
        print(msg + f" under {distill_base}/")
        return
    now = datetime.now().replace(microsecond=0)

    div = "─" * 76
    print(f"\nDistillation progress — {puzzle_dir}")
    print(div)

    for run in runs:
        output_dir = os.path.join(distill_base, run)
        abs_out = os.path.abspath(output_dir)
        is_running = abs_out in running_dirs
        run_log_path = os.path.join(output_dir, "log.txt")
        log_text = parse_log_text(run_log_path)

        latest, all_iters = latest_iter(output_dir)
        hf_dir = os.path.join(output_dir, "hf")
        hf_done = (
            os.path.isdir(hf_dir)
            and any(f.endswith((".safetensors", ".bin", "config.json")) for f in os.listdir(hf_dir))
            if os.path.isdir(hf_dir)
            else False
        )

        timing = parse_log_timing(log_text, output_dir) if is_running else {}

        run_cfg = read_run_config(output_dir)

        print(f"\n  Ratio:      {run}")
        print(f"  Output dir: {output_dir}")

        # Dataset + token info
        if run_cfg.get("datasets"):
            print(f"  Dataset:    {', '.join(run_cfg['datasets'])}")
        if run_cfg.get("gbs") and run_cfg.get("seq_len"):
            gbs, seq_len = run_cfg["gbs"], run_cfg["seq_len"]
            tokens_per_iter = gbs * seq_len
            # Use live iter from timing if running, else latest checkpoint
            effective_iter = timing.get("cur_iter") if is_running and timing else latest
            if effective_iter is not None:
                tokens_done = effective_iter * tokens_per_iter
                total_iters_cfg = run_cfg.get("train_iters")
                tokens_total = (total_iters_cfg * tokens_per_iter) if total_iters_cfg else None
                token_str = fmt_tokens(tokens_done)
                if tokens_total:
                    token_str += f" / {fmt_tokens(tokens_total)}"
                print(f"  Tokens:     {token_str}  (GBS={gbs}, seq={seq_len})")

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

        # Timing block
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
        elif not is_running and all_iters:
            # For completed/stopped runs, estimate elapsed from first→last checkpoint mtime
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            try:
                first_mtime = os.path.getmtime(os.path.join(ckpt_dir, f"iter_{all_iters[0]:07d}"))
                last_mtime = os.path.getmtime(os.path.join(ckpt_dir, f"iter_{all_iters[-1]:07d}"))
                elapsed_s = int(last_mtime - first_mtime)
                if elapsed_s > 0:
                    print(f"  Elapsed:    {fmt_duration(elapsed_s)}  (first→last checkpoint)")
            except OSError:
                pass

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

        # Log file (running)
        if is_running:
            log_path = os.path.abspath(run_log_path)
            if os.path.isfile(log_path):
                print(f"  Log file:   {log_path}")

        # --- Loss curves and convergence ---
        train_hist = parse_training_loss_history(log_text, output_dir)
        val_hist = parse_validation_loss_history(log_text, output_dir)
        val_ce_hist = parse_validation_ce_history(log_text, output_dir)

        # Fall back to tensorboard for stopped/older runs
        if not train_hist and not val_hist:
            train_hist, val_hist = read_loss_history_tb(output_dir)

        if train_hist or val_hist or val_ce_hist:
            # Infer bucket size: aim for ~10-20 bars in the sparkline.
            # Prefer val checkpoint spacing; fall back to train data range.
            bucket_size = None
            if len(val_hist) >= 2:
                diff = val_hist[1][0] - val_hist[0][0]
                if diff > 0:
                    bucket_size = diff
            if bucket_size is None and train_hist:
                max_iter = max(i for i, _ in train_hist)
                bucket_size = max(1, max_iter // 15)
            if bucket_size is None:
                bucket_size = 500

            bucketed_train = bucket_losses(train_hist, bucket_size) if train_hist else []

            print()
            if bucketed_train:
                first_l, last_l = bucketed_train[0][1], bucketed_train[-1][1]
                spark = make_sparkline(bucketed_train)
                print(f"  Train loss: {spark}  ({first_l:.3f} → {last_l:.3f})")
            if val_hist:
                first_l, last_l = val_hist[0][1], val_hist[-1][1]
                spark = make_sparkline(val_hist)
                print(f"  Val loss:   {spark}  ({first_l:.3f} → {last_l:.3f})")
                verdict, detail = convergence_verdict(val_hist)
                print(f"  Convergence: {verdict}  ({detail})")
            if val_ce_hist:
                first_l, last_l = val_ce_hist[0][1], val_ce_hist[-1][1]
                spark = make_sparkline(val_ce_hist)
                print(f"  Student CE: {spark}  ({first_l:.3f} → {last_l:.3f})")
        elif timing.get("last_loss") is not None:
            # Fallback: single loss value from timing parse
            print(f"  Last loss:  {timing['last_loss']:.4f} (iter {timing.get('cur_iter', '?')})")

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
