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

"""Test that `uv run` installs the launcher as the editable `modelopt_launcher` package.

conftest.py only puts tools/launcher on sys.path, which does not make `modelopt_launcher`
importable, so this test passes only when the package itself is installed. CI runs it
from a clean checkout with plain `uv run` (no explicit `pip install -e .`), which covers
the [build-system] table in pyproject.toml.
"""

import os

_LAUNCHER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_package_dir_resolves_to_checkout():
    """PACKAGE_DIR must be the checkout, not a copy in site-packages.

    launch.py's dev mode derives MODELOPT_ROOT from PACKAGE_DIR, so a non-editable install
    would silently stop packaging the ModelOpt source tree.
    """
    import modelopt_launcher

    assert os.path.realpath(modelopt_launcher.PACKAGE_DIR) == os.path.realpath(_LAUNCHER_DIR)
