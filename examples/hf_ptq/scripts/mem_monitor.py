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

"""Sidecar memory/utilization monitor for HF PTQ runs.

Samples GPU (device-level) and CPU memory + utilization at a fixed interval while
a *separate* workload process (e.g. ``hf_ptq.py``) runs, appends a CSV timeseries,
and prints a peak/mean summary on exit. This keeps profiling out of the workload
itself, so it can verify per-run budgets (e.g. the single-GPU layerwise target of
<=80 GB GPU / <=80 GB CPU) without perturbing calibration.

GPU memory is read at the *device* level (via NVML, falling back to ``nvidia-smi``)
so the monitor observes the workload's usage against the physical device budget
even though it runs as its own process. If no GPU driver is reachable, GPU columns
are left blank and only CPU is reported. Indices are physical NVML/PCI indices; if a
node's CUDA device order differs from NVML order, set ``CUDA_DEVICE_ORDER=PCI_BUS_ID``
so ``--gpus`` matches the GPUs the workload actually uses.

Two ways to bind the monitor to a workload:

* **Wrap mode** (preferred) — pass the workload after ``--``; the monitor launches
  it, tracks its process tree (for RSS + CPU%), and exits with its return code::

      python scripts/mem_monitor.py --gpus 2,3 --out mem.csv --summary peak.txt -- \\
          python hf_ptq.py ...

* **Standalone** — run in the background and stop it with a signal, ``--duration``,
  or ``--pid`` (tracks that PID's tree and exits when it does)::

      python scripts/mem_monitor.py --gpus 2,3 --out mem.csv & MON=$!
      python hf_ptq.py ...
      kill "$MON"
"""

import argparse
import contextlib
import csv
import shutil
import signal
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

import psutil

MB = 1024**2

CpuStat = namedtuple("CpuStat", ["sys_used", "rss", "sys_util", "proc_util"])


def _resolve_gpu_indices(spec: str) -> list[int] | None:
    """Parse ``--gpus``: ``none``/empty -> [], ``all`` -> None (all devices), else CSV of indices.

    Non-integer tokens (e.g. the GPU-UUID / MIG ids that ``CUDA_VISIBLE_DEVICES`` may
    hold) cannot be mapped to NVML indices, so GPU monitoring is disabled with a
    warning rather than crashing the wrapped workload.
    """
    spec = (spec or "").strip()
    if spec.lower() in ("", "none"):
        return []
    if spec.lower() == "all":
        return None
    try:
        return [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        print(
            f"mem_monitor: --gpus={spec!r} is not integer indices "
            "(UUID/MIG ids unsupported); disabling GPU monitoring.",
            file=sys.stderr,
        )
        return []


class GpuSampler:
    """Device-level GPU memory + utilization sampler backed by NVML, falling back to ``nvidia-smi``.

    ``indices`` is a list of physical device indices, ``None`` for all devices, or
    an empty list to disable GPU sampling. ``self.indices`` holds the indices that
    were actually resolved (empty if no driver is reachable). ``sample()`` returns
    ``{index: (used_bytes, util_pct_or_None)}``.
    """

    def __init__(self, indices: list[int] | None):
        self.indices: list[int] = []
        self._backend = None
        self._handles: dict[int, object] = {}
        self._wanted: set[int] = set()
        if indices == []:
            return
        self._init_nvml(indices) or self._init_smi(indices)

    def _init_nvml(self, indices: list[int] | None) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            wanted = range(count) if indices is None else [i for i in indices if i < count]
            self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in wanted}
            self.indices = sorted(self._handles)
            self._pynvml = pynvml
            self._backend = "nvml"
            if indices is not None and (missing := [i for i in indices if i >= count]):
                print(
                    f"mem_monitor: requested GPU indices {missing} exceed device "
                    f"count {count}; not monitored.",
                    file=sys.stderr,
                )
            return True
        except Exception:
            return False

    def _init_smi(self, indices: list[int] | None) -> bool:
        if shutil.which("nvidia-smi") is None:
            return False
        try:
            available = {i for i, _, _ in self._query_smi()}
            self.indices = sorted(available if indices is None else available.intersection(indices))
            self._wanted = set(self.indices)
            self._backend = "smi"
            return True
        except Exception:
            return False

    @staticmethod
    def _query_smi() -> list[tuple[int, int, int | None]]:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            index, used_mb, util = (part.strip() for part in line.split(","))
            try:
                util_pct: int | None = int(util)
            except ValueError:
                util_pct = None  # e.g. "[N/A]" on MIG / unsupported devices
            rows.append((int(index), int(used_mb) * MB, util_pct))
        return rows

    def sample(self) -> dict[int, tuple[int | None, int | None]]:
        """Return ``{device_index: (used_bytes, util_pct_or_None)}`` for the resolved indices."""
        if self._backend == "nvml":
            result = {}
            for i, handle in self._handles.items():
                try:
                    used = self._pynvml.nvmlDeviceGetMemoryInfo(handle).used
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except self._pynvml.NVMLError:
                    used = util = None
                result[i] = (used, util)
            return result
        if self._backend == "smi":
            return {i: (used, util) for i, used, util in self._query_smi() if i in self._wanted}
        return {}


def _sample_cpu(proc: psutil.Process | None) -> CpuStat:
    """System used memory + %, and (if ``proc`` given) its process-tree RSS + the process %."""
    sys_used = psutil.virtual_memory().used
    sys_util = psutil.cpu_percent(None)
    if proc is None:
        return CpuStat(sys_used, None, sys_util, None)
    try:
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                rss += child.memory_info().rss
        return CpuStat(sys_used, rss, sys_util, proc.cpu_percent(None))
    except psutil.Error:
        return CpuStat(sys_used, None, sys_util, None)


class _Accumulator:
    """Tracks running peak (for memory) and mean (for utilization) of a metric."""

    def __init__(self):
        self.peak = 0
        self._sum = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        self.peak = max(self.peak, value)
        self._sum += value
        self._count += 1

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def seen(self) -> bool:
        return self._count > 0


class _Metrics:
    """Time-aggregated peak/mean accumulators for every sampled metric."""

    def __init__(self, gpu_indices: list[int]):
        self.gpu = {i: (_Accumulator(), _Accumulator()) for i in gpu_indices}  # (memory, util)
        self.sys_cpu_mem = _Accumulator()
        self.sys_cpu_util = _Accumulator()
        self.rss = _Accumulator()
        self.proc_util = _Accumulator()


def _cell(acc: _Accumulator, value, scale: int = 1, ndigits: int | None = None):
    """Record ``value`` into ``acc`` and return its CSV cell ("" when the value is missing)."""
    if value is None:
        return ""
    acc.add(value)
    return round(value / scale, ndigits) if ndigits is not None else value


def _split_command(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Split ``argv`` on the first ``--`` into (monitor_args, wrapped_command_or_None)."""
    if "--" not in argv:
        return argv, None
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sidecar GPU/CPU memory + utilization monitor.",
        epilog="Append '-- <command>' to launch and monitor a workload, exiting with its return code.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument(
        "--gpus",
        default="all",
        help="Physical GPU indices to monitor: 'all', 'none', or a CSV like '2,3'.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Track this PID's process tree (ignored in wrap mode).",
    )
    parser.add_argument("--out", default="mem_trace.csv", help="CSV timeseries output path.")
    parser.add_argument(
        "--summary", default=None, help="Peak/mean summary path (also printed to stdout)."
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Optional max run time in seconds."
    )
    return parser.parse_args(argv)


def _write_summary(path, duration, metrics: _Metrics):
    lines = [f"duration_s: {duration:.1f}"]
    for i in sorted(metrics.gpu):
        mem_acc, util_acc = metrics.gpu[i]
        lines.append(f"peak_gpu{i}_used_mb: {mem_acc.peak / MB:.1f}")
        if util_acc.seen:
            lines.append(f"mean_gpu{i}_util_pct: {util_acc.mean:.1f}")
    lines.append(f"peak_sys_cpu_used_mb: {metrics.sys_cpu_mem.peak / MB:.1f}")
    lines.append(f"mean_sys_cpu_util_pct: {metrics.sys_cpu_util.mean:.1f}")
    if metrics.rss.seen:
        lines.append(f"peak_proc_rss_mb: {metrics.rss.peak / MB:.1f}")
        lines.append(f"mean_proc_cpu_util_pct: {metrics.proc_util.mean:.1f}")
    text = "\n".join(lines)
    print(text, flush=True)
    if path:
        Path(path).write_text(text + "\n")


def main() -> None:
    monitor_argv, command = _split_command(sys.argv[1:])
    args = parse_args(monitor_argv)
    gpu = GpuSampler(_resolve_gpu_indices(args.gpus))

    child = subprocess.Popen(command) if command else None
    target_pid = child.pid if child is not None else args.pid
    proc = psutil.Process(target_pid) if target_pid else None

    metrics = _Metrics(gpu.indices)

    stop = False

    def _request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # Prime cpu_percent so the first real sample reflects the interval, not 0.0.
    psutil.cpu_percent(None)
    if proc is not None:
        with contextlib.suppress(psutil.Error):
            proc.cpu_percent(None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "elapsed_s",
        *(f"gpu{i}_used_mb" for i in gpu.indices),
        *(f"gpu{i}_util_pct" for i in gpu.indices),
        "sys_cpu_used_mb",
        "sys_cpu_util_pct",
        "proc_rss_mb",
        "proc_cpu_util_pct",
    ]
    start = time.monotonic()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        while not stop:
            gpu_used = gpu.sample()
            cpu = _sample_cpu(proc)
            elapsed = time.monotonic() - start

            row = {"elapsed_s": round(elapsed, 3)}
            for i in gpu.indices:
                used, util = gpu_used.get(i, (None, None))
                mem_acc, util_acc = metrics.gpu[i]
                row[f"gpu{i}_used_mb"] = _cell(mem_acc, used, MB, 1)
                row[f"gpu{i}_util_pct"] = _cell(util_acc, util)
            row["sys_cpu_used_mb"] = _cell(metrics.sys_cpu_mem, cpu.sys_used, MB, 1)
            row["sys_cpu_util_pct"] = _cell(metrics.sys_cpu_util, cpu.sys_util)
            row["proc_rss_mb"] = _cell(metrics.rss, cpu.rss, MB, 1)
            row["proc_cpu_util_pct"] = _cell(metrics.proc_util, cpu.proc_util)
            writer.writerow(row)
            f.flush()

            if args.duration is not None and elapsed >= args.duration:
                break
            if child is not None and child.poll() is not None:
                break
            if child is None and proc is not None and not proc.is_running():
                break
            time.sleep(args.interval)

    if child is not None and child.poll() is None:
        child.terminate()
        child.wait()

    _write_summary(args.summary, time.monotonic() - start, metrics)
    if child is not None:
        rc = child.returncode
        sys.exit(rc if rc >= 0 else 128 - rc)  # 128+signal when the monitor stopped the child


if __name__ == "__main__":
    main()
