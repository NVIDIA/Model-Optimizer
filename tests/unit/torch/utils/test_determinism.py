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

import os
import random
from unittest import mock

import numpy as np
import pytest
import torch

from modelopt.torch.utils import enable_determinism


@pytest.fixture
def restore_global_state():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with mock.patch.dict(os.environ):
        yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    random.setstate(rng_states[0])
    np.random.set_state(rng_states[1])
    torch.set_rng_state(rng_states[2])


@pytest.mark.usefixtures("restore_global_state")
def test_enable_determinism():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    enable_determinism(7)
    draws = (random.random(), np.random.rand(), torch.rand(1).item())
    enable_determinism(7)

    assert (random.random(), np.random.rand(), torch.rand(1).item()) == draws
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    with pytest.warns(UserWarning, match="CUBLAS_WORKSPACE_CONFIG=:0:0"):
        enable_determinism(7)
