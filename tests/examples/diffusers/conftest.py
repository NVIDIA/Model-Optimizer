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

import pytest
from _test_utils.examples.diffusers_example_runner import run_example_in_process
from _test_utils.examples.run_command import set_in_process_runner
from _test_utils.fs_utils import assert_unmodified_tree
from _test_utils.torch.diffusers_models import (
    create_tiny_pixart_pipeline_dir,
    create_tiny_qwen_image_pipeline_dir,
)


@pytest.fixture
def in_process_examples():
    """Run example scripts in the pytest process; see ``_test_utils.examples.diffusers_example_runner``.

    Opt-in per test module (``pytestmark``). Per test, not per session: the hook is a module-global
    in ``run_command`` and must not leak into example suites collected later in the session.
    """
    set_in_process_runner(run_example_in_process)
    try:
        yield
    finally:
        set_in_process_runner(None)


@pytest.fixture(scope="session")
def tiny_qwen_image_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("qwen_image")
    with assert_unmodified_tree(create_tiny_qwen_image_pipeline_dir(tmp_path)) as path:
        yield str(path)


@pytest.fixture(scope="session")
def tiny_pixart_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pixart")
    with assert_unmodified_tree(create_tiny_pixart_pipeline_dir(tmp_path)) as path:
        yield str(path)
