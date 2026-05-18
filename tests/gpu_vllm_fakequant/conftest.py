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

"""Shared fixtures for vLLM fakequant tests.

Tests in this directory boot a real ``vllm.LLM(...)`` engine; vLLM handles its
own distributed init, current-vllm-config context, and parallel-state setup, so
this conftest doesn't need to do any of that — it only gates GPU availability.

We also opt into ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` *before* importing vLLM
so that ``LLM.apply_model(callable)`` can ship our worker callables over the
engine IPC channel via pickle. Without this, the default msgpack encoder rejects
raw functions and ``apply_model`` raises ``TypeError: Object of type <class
'function'> is not serializable``. This is only safe in a controlled test
environment.
"""

import os

# Must precede any ``import vllm``: the env is read at module-import time.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import pytest
import torch

pytest.importorskip("vllm")


@pytest.fixture(scope="session")
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for vLLM fakequant tests")
