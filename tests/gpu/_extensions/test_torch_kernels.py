# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pre-compile ModelOpt torch CUDA kernels so the one-time JIT build cost is paid here
rather than landing on the first functional test that uses them (e.g. the conv3d
implicit-GEMM tests). ``tests/gpu/_extensions`` is collected before ``tests/gpu/torch``, so
the module-level kernel cache is warm by the time those tests run in the same process.
"""

import pytest

# Override default timeout as these tests JIT-compile the CUDA extensions, which is slow
pytestmark = pytest.mark.timeout(180)


def test_conv3d_implicit_gemm():
    """Compile the conv3d implicit-GEMM CUDA extension."""
    from modelopt.torch.kernels.quantization.conv.implicit_gemm_cuda import _get_cuda_module

    assert _get_cuda_module() is not None
