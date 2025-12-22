# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Python script to automatically collect system information for reporting Issues."""

import contextlib
import platform
import re
import subprocess


def get_nvidia_gpu_info():
    """Get NVIDIA GPU Information."""
    try:
        nvidia_smi = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader,nounits",
                shell=True,
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        if len(nvidia_smi) > 0:
            gpu_name = nvidia_smi[0].split(",")[0].strip()
            gpu_memory = round(float(nvidia_smi[0].split(",")[1].strip()) / 1024, 1)
            gpu_count = len(nvidia_smi)
            return gpu_name, f"{gpu_memory} GB", gpu_count
    except Exception:
        return "?", "?", "?"


def get_cuda_version():
    """Get CUDA Version."""
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", nvcc_output)
        if match:
            return match.group(1)
    except Exception:
        return "?"


def get_package_version(package):
    """Get package version."""
    try:
        return getattr(__import__(package), "__version__", "?")
    except Exception:
        return "?"


# Get system info
os_info = f"{platform.system()} {platform.release()}"
if platform.system() == "Linux":
    with contextlib.suppress(Exception):
        os_info = (
            subprocess.check_output(
                "cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2", shell=True
            )
            .decode("utf-8")
            .strip()
            .strip('"')
        )
elif platform.system() == "Windows":
    print("Please add the `windows` label to the issue.")

cpu_arch = platform.machine()
gpu_name, gpu_memory, gpu_count = get_nvidia_gpu_info()
cuda_version = get_cuda_version()

# Print system information in the format required for the issue template
print("=" * 70)
print("- Container used (if applicable): " + "?")
print("- OS (e.g., Ubuntu 22.04, CentOS 7, Windows 10): " + os_info)
print("- CPU architecture (x86_64, aarch64): " + cpu_arch)
print("- GPU name (e.g. H100, A100, L40S): " + gpu_name)
print("- GPU memory size: " + gpu_memory)
print("- Number of GPUs: " + str(gpu_count))
print("- Library versions (if applicable):")
print("  - Python: " + platform.python_version())
print("  - ModelOpt version or commit hash: " + get_package_version("modelopt"))
print("  - CUDA: " + cuda_version)
print("  - PyTorch: " + get_package_version("torch"))
print("  - Transformers: " + get_package_version("transformers"))
print("  - TensorRT-LLM: " + get_package_version("tensorrt_llm"))
print("  - ONNXRuntime: " + get_package_version("onnxruntime"))
print("  - TensorRT: " + get_package_version("tensorrt"))
print("- Any other details that may help: " + "?")
print("=" * 70)
