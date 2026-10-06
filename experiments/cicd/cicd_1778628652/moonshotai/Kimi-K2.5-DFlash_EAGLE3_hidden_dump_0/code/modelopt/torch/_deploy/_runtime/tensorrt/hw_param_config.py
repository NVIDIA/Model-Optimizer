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

"""Hardware specific parameters.

All the hardware parameters will be treated as dictionary type for omniengine client APIs.
The key name and the suggested value range are listed with name TENSORRT_HW_PARAMS_SUGGESTED_OPTIONS.
The key name and the optimum value are listed with name TENSORRT_HW_PARAMS_OPT_OPTIONS.
"""

from .constants import (
    DEFAULT_AVG_TIMING,
    DEFAULT_MAX_WORKSPACE_SIZE,
    DEFAULT_MIN_TIMING,
    DEFAULT_TACTIC_SOURCES,
)

# Key names
# Workspace unit: MB
MAX_WORKSPACE_SIZE = [16, 32, 64, 128, 256, 512, 1024]
TACTIC_SOURCES = ["cublasLt", "cublas", "cudnn"]
ALL_TATIC_SOURCES_COMPONENT = []
for source in TACTIC_SOURCES:
    ALL_TATIC_SOURCES_COMPONENT.append("+" + source)
    ALL_TATIC_SOURCES_COMPONENT.append("-" + source)

TENSORRT_HW_PARAMS_SUGGESTED_OPTIONS = {
    "tacticSources": ALL_TATIC_SOURCES_COMPONENT,
    "minTiming": range(1, 5),
    "avgTiming": range(1, 5),
    "workspace": MAX_WORKSPACE_SIZE,
}

TENSORRT_HW_PARAMS_OPT_OPTIONS = {
    "tacticSources": DEFAULT_TACTIC_SOURCES,
    "minTiming": str(DEFAULT_MIN_TIMING),
    "avgTiming": str(DEFAULT_AVG_TIMING),
    "workspace": str(DEFAULT_MAX_WORKSPACE_SIZE),
}
