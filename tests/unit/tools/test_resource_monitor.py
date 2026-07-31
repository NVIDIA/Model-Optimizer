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
"""Unit tests for ``tools/resource_monitor.py``.

The module is a standalone script (not inside the ``modelopt`` package), so we add
``tools/`` to ``sys.path`` before importing it. These tests are CPU-only: GPU sampling
is exercised via the disabled (``none``) path.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import psutil

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import resource_monitor as mm

_SCRIPT = _TOOLS_DIR / "resource_monitor.py"


# ---------- helpers ---------------------------------------------------------


def test_resolve_gpu_indices():
    assert mm._resolve_gpu_indices("none") == []
    assert mm._resolve_gpu_indices("") == []
    assert mm._resolve_gpu_indices("all") is None
    assert mm._resolve_gpu_indices("2,3") == [2, 3]
    assert mm._resolve_gpu_indices(" 2 , 3 ") == [2, 3]
    # UUID / MIG ids can't map to NVML indices -> disable GPU monitoring, never crash.
    assert mm._resolve_gpu_indices("GPU-abcd,MIG-0") == []
    # argparse nargs tokens: space-separated (--gpus 2 3) and single CSV token both work.
    assert mm._resolve_gpu_indices(["2", "3"]) == [2, 3]
    assert mm._resolve_gpu_indices(["2,3"]) == [2, 3]
    assert mm._resolve_gpu_indices(["all"]) is None


def test_gpus_accepts_space_separated_and_csv():
    # --gpus 0 1 2 3 (space-separated) no longer crashes argparse; CSV still works.
    assert mm._resolve_gpu_indices(mm.parse_args(["--gpus", "0", "1", "2", "3"]).gpus) == [
        0,
        1,
        2,
        3,
    ]
    assert mm._resolve_gpu_indices(mm.parse_args(["--gpus", "2,3"]).gpus) == [2, 3]
    assert mm._resolve_gpu_indices(mm.parse_args([]).gpus) is None  # default 'all'


def test_gpu_sampler_disabled():
    sampler = mm.GpuSampler([])
    assert sampler.indices == []
    assert sampler.sample() == {}


def test_query_smi_parsing(monkeypatch):
    # nvidia-smi CSV parsing (incl. "[N/A]" -> None) is pure string logic; exercise it
    # without a GPU by faking check_output. Row order: index, mem_used, util, power, temp.
    fake = "0, 1024, 37, 250.5, 61\n1, [N/A], [N/A], [N/A], [N/A]\n"
    monkeypatch.setattr(mm.subprocess, "check_output", lambda *a, **k: fake)
    rows = mm.GpuSampler._query_smi()
    assert rows[0] == (0, 1024 * mm.MB, 37, 250.5, 61)
    assert rows[1] == (1, None, None, None, None)


def test_sample_smi_failure_yields_empty(monkeypatch):
    # A transient nvidia-smi failure during sampling must not raise (it would orphan the
    # wrapped workload); sample() returns {} so the row is written with blank GPU cells.
    sampler = mm.GpuSampler([])
    sampler._backend = "smi"
    sampler._wanted = {0}

    def _boom(*a, **k):
        raise mm.subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(mm.subprocess, "check_output", _boom)
    assert sampler.sample() == {}


def test_split_command():
    assert mm._split_command(["--gpus", "none"]) == (["--gpus", "none"], None)
    monitor_args, command = mm._split_command(["--out", "x.csv", "--", "python", "-c", "pass"])
    assert monitor_args == ["--out", "x.csv"]
    assert command == ["python", "-c", "pass"]


def test_sample_cpu_system_only():
    stat = mm._sample_cpu(None)
    assert stat.sys_total > 0
    assert stat.sys_used > 0
    assert stat.sys_free >= 0
    assert stat.sys_util >= 0.0
    assert stat.rss is None
    assert stat.proc_util is None


def test_sample_cpu_process_tree():
    stat = mm._sample_cpu(psutil.Process(os.getpid()))
    assert stat.rss is not None and stat.rss > 0


def test_accumulator():
    acc = mm._Accumulator()
    assert not acc.seen
    for v in (10, 30, 20):
        acc.add(v)
    assert acc.peak == 30
    assert acc.min == 10
    assert acc.mean == 20.0
    assert acc.seen


# ---------- end-to-end (subprocess) -----------------------------------------


def _run(args, **kwargs):
    kwargs.setdefault("timeout", 30)
    return subprocess.run([sys.executable, str(_SCRIPT), *args], **kwargs)


def test_standalone_writes_csv_and_summary(tmp_path):
    csv_path = tmp_path / "trace.csv"
    summary_path = tmp_path / "peak.txt"
    _run(
        [
            "--gpus",
            "none",
            "--interval",
            "0.05",
            "--duration",
            "0.2",
            "--out",
            str(csv_path),
            "--summary",
            str(summary_path),
        ],
        check=True,
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "expected at least one sample row"
    for col in (
        "elapsed_s",
        "sys_cpu_total_mb",
        "sys_cpu_used_mb",
        "sys_cpu_free_mb",
        "sys_cpu_util_pct",
        "proc_rss_mb",
        "proc_cpu_util_pct",
    ):
        assert col in rows[0]

    summary = summary_path.read_text()
    assert "sys_cpu_total_mb:" in summary
    assert "peak_sys_cpu_used_mb:" in summary
    assert "min_sys_cpu_free_mb:" in summary
    assert "mean_sys_cpu_util_pct:" in summary


def test_wrap_mode_propagates_exit_code(tmp_path):
    ok = _run(
        [
            "--gpus",
            "none",
            "--interval",
            "0.05",
            "--out",
            str(tmp_path / "a.csv"),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)",
        ],
    )
    assert ok.returncode == 0

    fail = _run(
        [
            "--gpus",
            "none",
            "--interval",
            "0.05",
            "--out",
            str(tmp_path / "b.csv"),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
        ],
    )
    assert fail.returncode == 3

    # Wrap mode tracks the child tree, so proc_rss is populated.
    with open(tmp_path / "a.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["proc_rss_mb"] != ""
