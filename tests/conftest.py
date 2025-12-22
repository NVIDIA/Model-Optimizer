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

import platform

import pytest


@pytest.fixture(scope="session")
def verbose(request):
    return request.config.getoption("verbose")


def pytest_addoption(parser):
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run manual tests",
    )
    parser.addoption(
        "--run-release",
        action="store_true",
        default=False,
        help="Run release tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests with specific markers unless their corresponding flag is provided."""
    skip_marks = [
        ("manual", "--run-manual"),
        ("release", "--run-release"),
    ]

    for mark_name, option_name in skip_marks:
        if not config.getoption(option_name):
            skipper = pytest.mark.skip(reason=f"Only run when {option_name} is given")
            for item in items:
                if mark_name in item.keywords:
                    item.add_marker(skipper)


@pytest.fixture
def skip_on_windows():
    if platform.system() == "Windows":
        pytest.skip("Skipping on Windows")
