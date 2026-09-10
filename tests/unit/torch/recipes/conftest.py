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

"""Shared test fixtures."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_modelopt: needs nvidia-modelopt installed")


def pytest_collection_modifyitems(config, items):
    try:
        import modelopt.torch.quantization.config  # noqa: F401

        return  # modelopt available, run all tests
    except ModuleNotFoundError:
        pass
    skip = pytest.mark.skip(reason="nvidia-modelopt not installed")
    for item in items:
        if "requires_modelopt" in item.keywords:
            item.add_marker(skip)
