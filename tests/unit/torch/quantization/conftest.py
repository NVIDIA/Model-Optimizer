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

import pytest
import torch


@pytest.fixture(scope="package", autouse=True)
def _warm_cuda_extensions():
    """Build the JIT CUDA quantization extensions once, before any test in this package runs.

    Some quantization tests (e.g. NVFP4/MX dynamic-block quant) trigger a first-time
    ``torch.utils.cpp_extension.load()`` build that can take ~60s+ on a cold cache. The
    per-test cap from ``tests/conftest.py`` (``unit`` → 60s) would otherwise fire on that
    first build and fail an otherwise-passing test.

    Running the build here, in fixture *setup*, keeps it off the clock: the repo sets
    ``timeout_func_only = true`` (see ``pyproject.toml``), so pytest-timeout only times the
    test ``call`` phase, not fixture setup/teardown. The compiled ``.so`` is cached (in
    memory for the session and on disk under ``~/.cache*/torch_extensions``), so subsequent
    loads are instant.

    Best-effort: ``precompile`` does not raise if an extension can't be built; tests that
    require a specific extension guard on it (``get_cuda_ext_*() is None``) themselves.
    """
    if torch.cuda.is_available():
        from modelopt.torch.quantization.extensions import precompile

        precompile()
