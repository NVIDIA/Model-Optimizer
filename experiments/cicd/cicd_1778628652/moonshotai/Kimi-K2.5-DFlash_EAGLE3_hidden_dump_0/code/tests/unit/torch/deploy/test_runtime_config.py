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

from modelopt.torch._deploy._runtime import RuntimeRegistry


def sanitize_deployment_config(deployment):
    client = RuntimeRegistry.get(deployment)
    client.sanitize_deployment_config(deployment)


@pytest.mark.parametrize(
    ("invalid_deployment", "error_msg"),
    [
        ({"runtime": "invalid_runtime"}, "Runtime invalid_runtime is not supported."),
        ({}, "Runtime was not set."),
    ],
)
def test_invalid_device(invalid_deployment, error_msg) -> None:
    with pytest.raises((ValueError, KeyError), match=error_msg):
        sanitize_deployment_config(invalid_deployment)


@pytest.mark.parametrize(
    "invalid_deployment",
    [
        ({"runtime": "ORT", "accelerator": "GPU"}),
    ],
)
def test_accelerator_not_found(invalid_deployment) -> None:
    with pytest.raises(AssertionError, match=".*\\('accelerator', 'GPU'\\): .*"):
        sanitize_deployment_config(invalid_deployment)
