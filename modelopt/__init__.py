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

"""Nvidia Model Optimizer (modelopt)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("nvidia-modelopt")
except PackageNotFoundError:
    # No dist metadata — e.g. the modelopt source tree is mounted directly into a
    # vLLM / TRT-LLM container's site-packages (as the launcher does) instead of being
    # pip-installed. Importing modelopt must not crash in that case; downstream tools
    # (specdec_bench, collect_hidden_states) only need the package, not its version.
    __version__ = "0.0.0+unknown"
