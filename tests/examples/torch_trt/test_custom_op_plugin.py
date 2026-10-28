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

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("nvcc") is None,
    reason="Requires CMake, CUDA Toolkit, and TensorRT development files",
)
def test_identity_conv_plugin_build(tmp_path):
    """Build the custom-op plugin against the installed TensorRT C++ interfaces."""
    source = Path(__file__).resolve().parents[3] / "examples/onnx_ptq/custom_op_plugin/plugin"
    subprocess.run(["cmake", "-S", str(source), "-B", str(tmp_path)], check=True, timeout=60)
    subprocess.run(["cmake", "--build", str(tmp_path), "--parallel", "2"], check=True, timeout=120)
    assert (tmp_path / "libidentity_conv_plugin.so").is_file()
