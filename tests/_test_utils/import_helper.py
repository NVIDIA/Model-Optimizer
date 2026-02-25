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
import ctypes
import importlib.metadata
import os
import shutil

import pytest
from packaging import version


def skip_if_no_tensorrt():
    from modelopt.onnx.quantization.ort_utils import _check_for_tensorrt

    try:
        _check_for_tensorrt()
    except (AssertionError, ImportError) as e:
        pytest.skip(f"{e}", allow_module_level=True)

    # Also verify that ORT's TensorRT EP can actually load its native library.
    # The tensorrt Python package may be installed, but ORT's provider shared library
    # (libonnxruntime_providers_tensorrt.so) could fail to load due to CUDA version
    # mismatches (e.g., ORT built for CUDA 12 running on a CUDA 13 system).
    try:
        import onnxruntime

        ort_capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        trt_provider_lib = os.path.join(ort_capi_dir, "libonnxruntime_providers_tensorrt.so")
        if os.path.isfile(trt_provider_lib):
            ctypes.CDLL(trt_provider_lib)
    except OSError as e:
        pytest.skip(
            f"ORT TensorRT EP native library cannot be loaded: {e}",
            allow_module_level=True,
        )


def skip_if_no_trtexec():
    if not shutil.which("trtexec"):
        pytest.skip("trtexec cmdline tool is not available", allow_module_level=True)


def skip_if_no_libcudnn():
    from modelopt.onnx.quantization.ort_utils import _check_for_libcudnn

    try:
        _check_for_libcudnn()
    except FileNotFoundError as e:
        pytest.skip(f"{e}!", allow_module_level=True)


def skip_if_no_megatron(*, te_required: bool = True, mamba_required: bool = False):
    try:
        import megatron  # noqa: F401
    except ImportError:
        pytest.skip("megatron not available", allow_module_level=True)

    try:
        import transformer_engine  # noqa: F401

        has_te = True
    except ImportError:
        has_te = False

    try:
        import mamba_ssm  # noqa: F401

        has_mamba = True
    except ImportError:
        has_mamba = False

    if te_required and not has_te:
        pytest.skip("TE required for Megatron test", allow_module_level=True)

    if mamba_required and not has_mamba:
        pytest.skip("Mamba required for Megatron test", allow_module_level=True)


def skip_if_onnx_version_above_1_18():
    package_name = "onnx"
    required_version = "1.18.0"

    try:
        installed_version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{package_name} is not installed")

    if version.parse(installed_version) > version.parse(required_version):
        pytest.skip(
            f"{package_name} version {installed_version} is greater than required {required_version}"
        )
