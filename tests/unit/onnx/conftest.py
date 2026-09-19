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

import contextlib

import pytest


@pytest.fixture(scope="session", autouse=True)
def _prebuild_onnx_round_and_pack_ext():
    """Build the ONNX round-and-pack extension before per-test timeouts start.

    ``modelopt/onnx/quantization/extensions.py`` runs ``cppimport.imp`` at module import, and
    that module is imported lazily from inside ``quant_utils.round_and_pack``. So the first test
    to need it pays a full C++ compile INSIDE its own per-test timeout -- on the Windows runner
    that is an MSVC build measured in minutes, and the test dies with pytest-timeout while
    ``compiler.compile`` is still running. Which test pays is down to collection order, so the
    failure appears to wander between runs.

    ``pyproject`` sets ``timeout_func_only``, so the per-test clock covers the call only; doing
    the import here in session setup puts the build outside it. This mirrors
    ``tests/gpu_megatron/conftest.py``, which prebuilds the quant CUDA extensions for the same
    reason -- but it cannot reuse that helper: ``load_cpp_extension`` skips every quant extension
    when CUDA is unavailable, which is exactly the case on the CPU-only Windows runner, so
    ``precompile()`` would warm nothing here.

    Best-effort. The extension is an optimisation with a Python fallback -- ``extensions.py``
    already swallows its own build failures -- so a failure to prebuild must not fail the session.
    """
    with contextlib.suppress(Exception):
        import modelopt.onnx.quantization.extensions  # noqa: F401
