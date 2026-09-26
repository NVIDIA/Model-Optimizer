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


@pytest.fixture(scope="session")
def tiny_wan22_path(tmp_path_factory):
    """Create a tiny Wan 2.2 (14B-style) pipeline and return its path.

    Built once per session and shared across all tests that need it.
    """
    try:
        from _test_utils.torch.diffusers_models import create_tiny_wan22_pipeline_dir
    except ImportError:
        pytest.skip("Wan 2.2 diffusers models not available (requires diffusers with WanPipeline)")

    tmp_path = tmp_path_factory.mktemp("wan22")
    return str(create_tiny_wan22_pipeline_dir(tmp_path))
