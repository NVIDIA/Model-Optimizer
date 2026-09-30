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

r"""Cross-process resource monitor: sample a workload's GPU/CPU usage from the outside.

Samples GPU (memory, utilization, power, temperature) and CPU (total/used/free memory,
utilization) at a fixed interval while a *separate* workload process (e.g. ``hf_ptq.py``)
runs, writes a CSV timeseries (overwriting any existing file), and prints a peak/mean/min
summary on exit. This keeps profiling out of the workload itself, so it can verify per-run
budgets (e.g. the single-GPU layerwise target of <=80 GB GPU / <=80 GB CPU) without
perturbing calibration.

Not to be confused with ``modelopt.torch.utils.memory_monitor.GPUMemoryMonitor``: that is
an *in-process* thread (import it inside your Python workload) that tracks device GPU memory
only. This is a standalone *cross-process* tool that wraps any command, adds CPU + utilization
+ power/temperature, writes a CSV/summary, and survives an OOM/kill of the monitored process.

GPU memory is read at the *device* level (via NVML, falling back to ``nvidia-smi``)
so the monitor observes the workload's usage against the physical device budget
even though it runs as its own process. If no GPU driver is reachable, GPU columns
are left blank and only CPU is reported. Indices are physical NVML/PCI indices; if a
node's CUDA device order differs from NVML order, set ``CUDA_DEVICE_ORDER=PCI_BUS_ID``
so ``--gpus`` matches the GPUs the workload actually uses.

Two ways to bind the monitor to a workload:

* **Wrap mode** (preferred) — pass the workload after ``--``; the monitor launches
  it, tracks its process tree (for RSS + CPU%), and exits with its return code::

      python tools/resource_monitor.py --gpus 2,3 --out mem.csv --summary peak.txt -- \
          python hf_ptq.py ...

* **Standalone** — run in the background and stop it with a signal, ``--duration``,
  or ``--pid`` (tracks that PID's tree and exits when it does)::

      python tools/resource_monitor.py --gpus 2,3 --out mem.csv & MON=$!
      python hf_ptq.py ...
      kill "$MON"
"""

import argparse
import contextlib
import csv
import shutil
import signal
import subprocess  # nosec B404 - wraps and monitors a user-provided workload command; no shell is used
import sys
import time
from collections import namedtuple
from pathlib import Path

import psutil

MB = 1024**2

GpuSample = namedtuple("GpuSample", ["used", "util", "power", "temp"])
_EMPTY_GPU = GpuSample(None, None, None, None)
CpuStat = namedtuple(
    "CpuStat", ["sys_total", "sys_used", "sys_free", "sys_util", "rss", "proc_util"]
)


def _to_number(value: str, cast):
    """Parse an ``nvidia-smi`` field, returning None for '[N/A]'/unsupported values."""
    try:
        return cast(value)
    except ValueError:
        return None


def _resolve_gpu_indices(spec) -> list[int] | None:
    """Parse ``--gpus``: ``none``/empty -> [], ``all`` -> None (all devices), else GPU indices.

    ``spec`` may be a single string (CSV like ``"2,3"``) or a list of argparse tokens
    (space-separated like ``["2", "3"]``); both are accepted. Non-integer tokens (e.g. the
    GPU-UUID / MIG ids that ``CUDA_VISIBLE_DEVICES`` may hold) cannot be mapped to NVML
    indices, so GPU monitoring is disabled with a warning rather than crashing the workload.
    """
    if isinstance(spec, (list, tuple)):
        spec = ",".join(spec)
    spec = (spec or "").strip()
    if spec.lower() in ("", "none"):
        return []
    if spec.lower() == "all":
        return None
    try:
        return [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        print(
            f"resource_monitor: --gpus={spec!r} is not integer indices "
            "(UUID/MIG ids unsupported); disabling GPU monitoring.",
            file=sys.stderr,
        )
        return []


class GpuSampler:
    """GPU sampler for memory/utilization/power/temperature via NVML (``nvidia-smi`` fallback).

    ``indices`` is a list of physical device indices, ``None`` for all devices, or
    an empty list to disable GPU sampling. ``self.indices`` holds the indices that
    were actually resolved (empty if no driver is reachable). ``sample()`` returns
    ``{index: GpuSample}``.
    """

    def __init__(self, indices: list[int] | None):
        """Resolve ``indices`` and open NVML/``nvidia-smi`` handles; see the class docstring."""
        self.indices: list[int] = []
        self._backend = None
        self._handles: dict[int, object] = {}
        self._wanted: set[int] = set()
        if indices == []:
            return
        self._init_nvml(indices) or self._init_smi(indices)

    def _init_nvml(self, indices: list[int] | None) -> bool:
        try:
            # Optional dependency: pynvml (nvidia-ml-py) may be absent; on any failure
            # the sampler falls back to parsing ``nvidia-smi`` in _init_smi.
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
                    f"resource_monitor: requested GPU indices {missing} exceed device "
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
            available = {i for i, *_ in self._query_smi()}
            self.indices = sorted(available if indices is None else available.intersection(indices))
            self._wanted = set(self.indices)
            self._backend = "smi"
            return True
        except Exception:
            return False

    @staticmethod
    def _query_smi() -> list[tuple]:
        out = subprocess.check_output(  # nosec B603 B607 - fixed nvidia-smi argv; no shell
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,  # never let a hung nvidia-smi block the sampling loop
        )
        rows = []
        for line in out.strip().splitlines():
            index, used_mb, util, power, temp = (part.strip() for part in line.split(","))
            used = _to_number(used_mb, int)  # "[N/A]" on some MIG / unsupported devices
            rows.append(
                (
                    int(index),
                    used * MB if used is not None else None,
                    _to_number(util, int),
                    _to_number(power, float),
                    _to_number(temp, int),
                )
            )
        return rows

    def sample(self) -> dict[int, GpuSample]:
        """Return ``{device_index: GpuSample}`` for the resolved indices.

        Each metric is read independently so an unsupported one (e.g. utilization or
        power on MIG/vGPU) doesn't drop the others.
        """
        if self._backend == "nvml":
            result = {}
            for i, handle in self._handles.items():
                used = util = power = temp = None
                with contextlib.suppress(self._pynvml.NVMLError):
                    used = self._pynvml.nvmlDeviceGetMemoryInfo(handle).used
                with contextlib.suppress(self._pynvml.NVMLError):
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                with contextlib.suppress(self._pynvml.NVMLError):
                    power = self._pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W
                with contextlib.suppress(self._pynvml.NVMLError):
                    temp = self._pynvml.nvmlDeviceGetTemperature(
                        handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                result[i] = GpuSample(used, util, power, temp)
            return result
        if self._backend == "smi":
            try:
                rows = self._query_smi()
            except Exception:
                # A transient nvidia-smi failure (nonzero exit, driver reset, timeout)
                # must not kill the monitor and orphan the wrapped workload; skip this sample.
                return {}
            return {
                i: GpuSample(used, util, power, temp)
                for i, used, util, power, temp in rows
                if i in self._wanted
            }
        return {}


def _sample_cpu(proc: psutil.Process | None) -> CpuStat:
    """System memory (total/used/free) + CPU%, and (if ``proc`` given) tree RSS + process CPU%.

    ``sys_free`` is ``virtual_memory().available`` (allocatable memory incl. reclaimable page
    cache), not raw ``.free`` -- the right headroom measure for a budget monitor.
    """
    vm = psutil.virtual_memory()
    sys_util = psutil.cpu_percent(None)
    if proc is None:
        return CpuStat(vm.total, vm.used, vm.available, sys_util, None, None)
    try:
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                rss += child.memory_info().rss
        return CpuStat(vm.total, vm.used, vm.available, sys_util, rss, proc.cpu_percent(None))
    except psutil.Error:
        return CpuStat(vm.total, vm.used, vm.available, sys_util, None, None)


class _Accumulator:
    """Tracks running peak, min, and mean of a metric."""

    def __init__(self):
        self.peak = 0
        self.min = None
        self._sum = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        self.peak = max(self.peak, value)
        self.min = value if self.min is None else min(self.min, value)
        self._sum += value
        self._count += 1

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def seen(self) -> bool:
        return self._count > 0


class _Metrics:
    """Time-aggregated accumulators for every sampled metric."""

    def __init__(self, gpu_indices: list[int]):
        self.gpu = {
            i: {k: _Accumulator() for k in ("used", "util", "power", "temp")} for i in gpu_indices
        }
        self.sys_total = _Accumulator()
        self.sys_used = _Accumulator()
        self.sys_free = _Accumulator()
        self.sys_util = _Accumulator()
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
    """Parse the monitor's CLI arguments (the tokens before any ``--`` separator)."""
    parser = argparse.ArgumentParser(
        description="Sidecar GPU/CPU memory, utilization, power & temperature monitor.",
        epilog="Append '-- <command>' to launch and monitor a workload, exiting with its return code.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["all"],
        metavar="GPU",
        help="GPUs to monitor: 'all', 'none', or physical indices as CSV ('2,3') or "
        "space-separated ('2 3').",
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
        acc = metrics.gpu[i]
        if acc["used"].seen:
            lines.append(f"peak_gpu{i}_used_mb: {acc['used'].peak / MB:.1f}")
        if acc["util"].seen:
            lines.append(f"mean_gpu{i}_util_pct: {acc['util'].mean:.1f}")
        if acc["power"].seen:
            lines.append(f"peak_gpu{i}_power_w: {acc['power'].peak:.1f}")
        if acc["temp"].seen:
            lines.append(f"peak_gpu{i}_temp_c: {acc['temp'].peak:.0f}")
    if metrics.sys_total.seen:
        lines.append(f"sys_cpu_total_mb: {metrics.sys_total.peak / MB:.1f}")
    if metrics.sys_used.seen:
        lines.append(f"peak_sys_cpu_used_mb: {metrics.sys_used.peak / MB:.1f}")
    if metrics.sys_free.min is not None:
        lines.append(f"min_sys_cpu_free_mb: {metrics.sys_free.min / MB:.1f}")
    if metrics.sys_util.seen:
        lines.append(f"mean_sys_cpu_util_pct: {metrics.sys_util.mean:.1f}")
    if metrics.rss.seen:
        lines.append(f"peak_proc_rss_mb: {metrics.rss.peak / MB:.1f}")
        lines.append(f"mean_proc_cpu_util_pct: {metrics.proc_util.mean:.1f}")
    text = "\n".join(lines)
    print(text, flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n")


def main() -> None:
    """Parse args, sample until the wrapped workload / duration ends, then write outputs."""
    monitor_argv, command = _split_command(sys.argv[1:])
    args = parse_args(monitor_argv)
    gpu = GpuSampler(_resolve_gpu_indices(args.gpus))

    child = subprocess.Popen(command) if command else None  # nosec B603 - runs the user-supplied command argv directly; no shell
    target_pid = child.pid if child is not None else args.pid
    try:
        proc = psutil.Process(target_pid) if target_pid else None
    except psutil.NoSuchProcess:
        print(f"resource_monitor: --pid {target_pid} is not a running process.", file=sys.stderr)
        sys.exit(1)

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
        *(f"gpu{i}_{m}" for i in gpu.indices for m in ("used_mb", "util_pct", "power_w", "temp_c")),
        "sys_cpu_total_mb",
        "sys_cpu_used_mb",
        "sys_cpu_free_mb",
        "sys_cpu_util_pct",
        "proc_rss_mb",
        "proc_cpu_util_pct",
    ]
    start = time.monotonic()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        while not stop:
            gpu_sample = gpu.sample()
            cpu = _sample_cpu(proc)
            elapsed = time.monotonic() - start

            row = {"elapsed_s": round(elapsed, 3)}
            for i in gpu.indices:
                s = gpu_sample.get(i, _EMPTY_GPU)
                acc = metrics.gpu[i]
                row[f"gpu{i}_used_mb"] = _cell(acc["used"], s.used, MB, 1)
                row[f"gpu{i}_util_pct"] = _cell(acc["util"], s.util)
                row[f"gpu{i}_power_w"] = _cell(acc["power"], s.power, 1, 1)
                row[f"gpu{i}_temp_c"] = _cell(acc["temp"], s.temp)
            row["sys_cpu_total_mb"] = _cell(metrics.sys_total, cpu.sys_total, MB, 1)
            row["sys_cpu_used_mb"] = _cell(metrics.sys_used, cpu.sys_used, MB, 1)
            row["sys_cpu_free_mb"] = _cell(metrics.sys_free, cpu.sys_free, MB, 1)
            row["sys_cpu_util_pct"] = _cell(metrics.sys_util, cpu.sys_util)
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
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    _write_summary(args.summary, time.monotonic() - start, metrics)
    if child is not None:
        rc = child.returncode
        sys.exit(rc if rc >= 0 else 128 - rc)  # 128+signal when the monitor stopped the child


if __name__ == "__main__":
    main()
