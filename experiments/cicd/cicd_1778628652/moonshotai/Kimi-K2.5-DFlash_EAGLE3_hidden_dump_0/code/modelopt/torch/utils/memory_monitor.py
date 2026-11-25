# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU Memory Monitoring Utilities.

This module provides utilities for monitoring GPU memory usage in real-time using NVIDIA Management Library (NVML).
It includes a GPUMemoryMonitor class that tracks peak memory usage across all available GPUs and provides
functionality to start/stop monitoring in a separate thread.

Classes:
    GPUMemoryMonitor: A class that monitors GPU memory usage and tracks peak memory consumption.

Functions:
    launch_memory_monitor: Helper function to create and start a GPU memory monitor instance.

Example:
    >>> monitor = launch_memory_monitor(monitor_interval=1.0)
    >>> # Run your GPU operations
    >>> monitor.stop()  # Will print peak memory usage per GPU

Note:
    This module requires the NVIDIA Management Library (NVML) through the pynvml package.
    It automatically initializes NVML when creating a monitor instance and shuts it down
    when monitoring is stopped.

Dependencies:
    - pynvml: For accessing NVIDIA GPU metrics
    - threading: For running the monitor in a background thread
    - atexit: For ensuring proper cleanup when the program exits
"""

import atexit
import threading
import time

from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)


class GPUMemoryMonitor:
    """GPU Memory Monitor for tracking NVIDIA GPU memory usage.

    This class provides functionality to monitor and track peak memory usage across all available
    NVIDIA GPUs on the system. It runs in a separate thread and periodically samples memory usage.
    """

    def __init__(self, monitor_interval: float = 10.0):
        """Initialize a NVIDIA GPU memory monitor.

        This class monitors the memory usage of NVIDIA GPUs at specified intervals.
        It initializes NVIDIA Management Library (NVML) and gets the count of available GPUs.

        Args:
            monitor_interval (float, optional): Time interval in seconds between memory usage checks.
                Defaults to 10.0.

        Attributes:
            monitor_interval (float): Time interval between memory checks.
            peak_memory (dict): Dictionary mapping GPU indices to their peak memory usage.
            is_running (bool): Flag indicating if the monitor is currently running.
            monitor_thread: Thread object for memory monitoring.
            device_count (int): Number of NVIDIA GPUs available in the system.

        Raises:
            NVMLError: If NVIDIA Management Library initialization fails.
        """
        self.monitor_interval = monitor_interval
        self.peak_memory = {}  # GPU index to peak memory mapping
        self.is_running = False
        self.monitor_thread = None
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()

    def _monitor_loop(self):
        while self.is_running:
            for i in range(self.device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                gpu_memory = nvmlDeviceGetMemoryInfo(handle)
                used_memory_gb = gpu_memory.used / (1024 * 1024 * 1024)  # Convert to GB
                self.peak_memory[i] = max(self.peak_memory.get(i, 0), used_memory_gb)
            time.sleep(self.monitor_interval)

    def start(self):
        """Start the GPU memory monitoring in a separate daemon thread.

        This method initializes and starts a daemon thread that continuously monitors
        GPU memory usage at the specified interval. The thread will run until stop()
        is called or the program exits.
        """
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        assert self.monitor_thread is not None
        self.monitor_thread.start()

    def stop(self):
        """Stop the GPU memory monitoring and display peak memory usage.

        This method stops the monitoring thread, prints the peak memory usage for each
        GPU that was monitored, and properly shuts down the NVML interface. It will
        wait for the monitoring thread to complete before returning.

        The peak memory usage is displayed in GB for each GPU index.
        """
        self.is_running = False
        # Print peak memory usage
        print("########")
        for gpu_idx, peak_mem in self.peak_memory.items():
            print(
                f"GPU {gpu_idx}: Peak memory usage = {peak_mem:.2f} GB for all processes on the GPU"
            )
        print("########")
        if self.monitor_thread:
            self.monitor_thread.join()
        nvmlShutdown()


def launch_memory_monitor(monitor_interval: float = 1.0) -> GPUMemoryMonitor | None:
    """Launch a GPU memory monitor in a separate thread.

    Args:
        monitor_interval (float): Time interval between memory checks in seconds

    Returns:
        GPUMemoryMonitor: The monitor instance that was launched
    """
    try:
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    except Exception as e:
        print(f"Failed to get GPU memory info: {e}. Stopping GPU memory monitor.")
        return None
    monitor = GPUMemoryMonitor(monitor_interval)
    monitor.start()
    atexit.register(monitor.stop)  # Ensure the monitor stops when the program exits
    return monitor
