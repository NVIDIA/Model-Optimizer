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

"""Pattern-Based Q/DQ Autotuning for ONNX Models.

This package provides automated optimization of Quantize/Dequantize (Q/DQ) node placement
in ONNX computation graphs to minimize TensorRT inference latency. It uses pattern-based
region analysis to efficiently explore and optimize Q/DQ insertion strategies.
"""

# Core data structures
from .benchmark import TensorRTPyBenchmark, TrtExecBenchmark
from .common import (
    AutotunerError,
    AutotunerNotInitializedError,
    Config,
    InsertionScheme,
    InvalidSchemeError,
    PatternCache,
    PatternSchemes,
    Region,
    RegionType,
)
from .insertion_points import (
    ChildRegionInputInsertionPoint,
    ChildRegionOutputInsertionPoint,
    NodeInputInsertionPoint,
    ResolvedInsertionPoint,
)
from .region_pattern import RegionPattern
from .region_search import CombinedRegionSearch

__all__ = [
    "AutotunerError",
    "AutotunerNotInitializedError",
    "ChildRegionInputInsertionPoint",
    "ChildRegionOutputInsertionPoint",
    "CombinedRegionSearch",
    "Config",
    "InsertionScheme",
    "InvalidSchemeError",
    "NodeInputInsertionPoint",
    "PatternCache",
    "PatternSchemes",
    "Region",
    "RegionPattern",
    "RegionType",
    "ResolvedInsertionPoint",
    "TensorRTPyBenchmark",
    "TrtExecBenchmark",
]
