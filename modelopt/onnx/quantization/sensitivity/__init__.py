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

"""ONNX quantization sensitivity scan.

Ranks quantization targets (op types or individual nodes) by the accuracy impact they would have if
quantized. The core primitive, :func:`score`, mutates the graph with a properly calibrated single-
target Q/DQ pass (via the standard :func:`modelopt.onnx.quantization.quantize` entry point), runs
both the FP16 reference and the quantized model through ONNXRuntime, and reports a proxy metric per
target so a downstream picker can decide which ops or nodes to keep at higher precision.
"""

from modelopt.onnx.quantization.sensitivity.picker import suggest_exclusion, summarize_exclusion
from modelopt.onnx.quantization.sensitivity.score import (
    CalibrationSource,
    Granularity,
    Metric,
    score,
)

__all__ = [
    "CalibrationSource",
    "Granularity",
    "Metric",
    "score",
    "suggest_exclusion",
    "summarize_exclusion",
]
