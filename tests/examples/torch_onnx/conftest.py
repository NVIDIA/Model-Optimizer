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
"""Run the torch_onnx steps in one long-lived worker; see ``_test_utils.examples.example_runner``."""

import pytest
from _test_utils.examples.example_runner import ExampleRunner, keep_post_conversion_plugins
from _test_utils.examples.run_command import set_in_process_runner

# hf_embedding_quant_to_onnx.py has not been checked for repeated runs in one process, so it keeps
# running as a subprocess.
_SCRIPTS = {"torch_quant_to_onnx.py"}


@pytest.fixture(scope="session")
def _example_runner():
    runner = ExampleRunner(_SCRIPTS, hooks=(keep_post_conversion_plugins,))
    yield runner
    runner.close()


@pytest.fixture(autouse=True)
def _run_steps_in_the_worker(_example_runner):
    """Install the runner per test, not per session.

    The hook is a module global in ``run_command``, so leaving it installed would also reach any
    other example suite collected later in the same session (``pytest tests/examples``).
    """
    set_in_process_runner(_example_runner)
    try:
        yield
    finally:
        set_in_process_runner(None)
