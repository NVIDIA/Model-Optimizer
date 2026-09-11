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

"""Module to load C++ / CUDA extensions."""

from pathlib import Path

from modelopt.torch.utils import load_cpp_extension

__all__ = [
    "get_cuda_ext",
    "get_cuda_ext_fp8",
    "get_cuda_ext_iq1_s",
    "get_cuda_ext_iq2_xs",
    "get_cuda_ext_mx",
    "precompile",
]

path = Path(__file__).parent
kernels_gemm = path.parent / "kernels" / "quantization" / "gemm"
kernels_iq1_s = path.parent / "kernels" / "quantization" / "iq1_s"
kernels_iq2_xs = path.parent / "kernels" / "quantization" / "iq2_xs"


def get_cuda_ext(raise_if_failed: bool = False):
    """Returns the cuda extension for tensor_quant."""
    if not hasattr(get_cuda_ext, "extension"):
        get_cuda_ext.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext",
            sources=[kernels_gemm / "tensor_quant.cpp", kernels_gemm / "tensor_quant_gpu.cu"],
            cuda_version_specifiers=">=11",
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext.extension  # type:ignore[attr-defined]


def get_cuda_ext_fp8(raise_if_failed: bool = False):
    """Returns the cuda extension for tensor_quant_fp8."""
    if not hasattr(get_cuda_ext_fp8, "extension"):
        get_cuda_ext_fp8.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext_fp8",
            sources=[kernels_gemm / "tensor_quant_gpu_fp8.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg=(
                "CUDA extension for FP8 quantization could not be built and loaded, FP8 simulated"
                " quantization will not be available."
            ),
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext_fp8.extension  # type:ignore[attr-defined]


def get_cuda_ext_mx(raise_if_failed: bool = False):
    """Returns the cuda extension for tensor_quant_mx."""
    if not hasattr(get_cuda_ext_mx, "extension"):
        get_cuda_ext_mx.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext_mx",
            sources=[
                kernels_gemm / "tensor_quant_mx.cu",
            ],
            cuda_version_specifiers=">=11.8",
            fail_msg=(
                "CUDA extension for MX quantization could not be built and loaded, MX simulated"
                " quantization will not be available."
            ),
            extra_cuda_cflags=["--use_fast_math"],
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext_mx.extension  # type:ignore[attr-defined]


def get_cuda_ext_iq1_s(raise_if_failed: bool = False):
    """Return the GGML-compatible IQ1_S packing extension."""
    if not hasattr(get_cuda_ext_iq1_s, "extension"):
        get_cuda_ext_iq1_s.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext_iq1_s",
            sources=[kernels_iq1_s / "iq1_s.cpp", kernels_iq1_s / "iq1_s.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="IQ1_S CUDA packing is unavailable; using the PyTorch reference encoder.",
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext_iq1_s.extension  # type:ignore[attr-defined]


def get_cuda_ext_iq2_xs(raise_if_failed: bool = False):
    """Return the GGML-compatible IQ2_XS packing extension."""
    if not hasattr(get_cuda_ext_iq2_xs, "extension"):
        get_cuda_ext_iq2_xs.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext_iq2_xs",
            sources=[kernels_iq2_xs / "iq2_xs.cpp", kernels_iq2_xs / "iq2_xs.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="IQ2_XS CUDA packing is unavailable; using the PyTorch reference encoder.",
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext_iq2_xs.extension  # type:ignore[attr-defined]


def __getattr__(name):
    if name == "cuda_ext":
        return get_cuda_ext()
    elif name == "cuda_ext_fp8":
        return get_cuda_ext_fp8()
    elif name == "cuda_ext_mx":
        return get_cuda_ext_mx()
    elif name == "cuda_ext_iq1_s":
        return get_cuda_ext_iq1_s()
    elif name == "cuda_ext_iq2_xs":
        return get_cuda_ext_iq2_xs()
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


def precompile():
    """Precompile the CUDA extensions."""
    print(get_cuda_ext())
    print(get_cuda_ext_fp8())
    print(get_cuda_ext_mx())
    print(get_cuda_ext_iq1_s())
    print(get_cuda_ext_iq2_xs())
