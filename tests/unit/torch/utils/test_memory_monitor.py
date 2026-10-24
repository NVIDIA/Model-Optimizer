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

"""Unit tests for launch_memory_monitor's NVML availability probe, against a fake NVML."""

import atexit
from types import SimpleNamespace

import pytest

import modelopt.torch.utils.memory_monitor as mm


class _FakeNVML:
    """Reference-counted like NVML: every call but init raises while uninitialized."""

    def __init__(self, memory_info_supported: bool = True):
        self.inits = 0
        self.memory_info_supported = memory_info_supported

    def install(self, monkeypatch):
        for name in ("Init", "Shutdown", "DeviceGetCount", "DeviceGetHandleByIndex"):
            monkeypatch.setattr(mm, f"nvml{name}", getattr(self, name))
        monkeypatch.setattr(mm, "nvmlDeviceGetMemoryInfo", self.DeviceGetMemoryInfo)

    def _require_init(self):
        if self.inits <= 0:
            raise RuntimeError("Uninitialized")

    def Init(self):  # noqa: N802
        self.inits += 1

    def Shutdown(self):  # noqa: N802
        self._require_init()
        self.inits -= 1

    def DeviceGetCount(self):  # noqa: N802
        self._require_init()
        return 1

    def DeviceGetHandleByIndex(self, index):  # noqa: N802
        self._require_init()
        return index

    def DeviceGetMemoryInfo(self, handle):  # noqa: N802
        self._require_init()
        if not self.memory_info_supported:
            raise RuntimeError("Not Supported")
        return SimpleNamespace(used=2 * 1024**3)


@pytest.fixture
def launched():
    monitors = []
    yield monitors
    for monitor in monitors:
        atexit.unregister(monitor.stop)
        monitor.stop()


@pytest.mark.parametrize("supported", [True, False], ids=["reports_memory", "cannot"])
def test_the_monitor_starts_only_where_nvml_can_report_memory(monkeypatch, launched, supported):
    """The probe used to run before NVML was initialized, so it always failed and the monitor
    never started on any host. Hosts where NVML cannot report memory still skip the monitor, and
    leave NVML shut down; elsewhere the probe's own init is balanced and the monitor holds one."""
    nvml = _FakeNVML(memory_info_supported=supported)
    nvml.install(monkeypatch)
    monitor = mm.launch_memory_monitor(monitor_interval=0.01)
    if monitor is not None:
        launched.append(monitor)

    assert (monitor is not None) is supported
    assert nvml.inits == (1 if supported else 0)
