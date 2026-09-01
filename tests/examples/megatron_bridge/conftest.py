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
"""Run example steps without a ``torchrun`` launch; see ``_test_utils.examples.megatron_example_runner``."""

import os

import pytest
from _test_utils.examples.megatron_example_runner import (
    reset_megatron_global_state,
    run_example_step,
)
from _test_utils.examples.run_command import set_in_process_runner


@pytest.fixture(scope="session", autouse=True)
def _fast_example_runner():
    """Run example steps without shelling out to ``torchrun``."""
    set_in_process_runner(run_example_step)
    try:
        yield
    finally:
        set_in_process_runner(None)


@pytest.fixture(autouse=True)
def _isolate_megatron_global_state():
    """Reset shared state around every test so a failure cannot cascade into the next one.

    In-process steps share the interpreter. Besides Megatron's singletons, Transformer-Engine
    records its chosen attention backend in ``NVTE_*``, which failed a Mamba hybrid that ran after
    an attention model -- so the environment is restored wholesale rather than by naming variables.
    """
    env_before = os.environ.copy()
    reset_megatron_global_state()
    try:
        yield
    finally:
        reset_megatron_global_state()
        os.environ.clear()
        os.environ.update(env_before)
