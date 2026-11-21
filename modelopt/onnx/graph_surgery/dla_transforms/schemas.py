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

"""Normalize graph schemas before the DLA transformation phases."""

import onnx
from onnx import version_converter

__all__ = []


def upgrade_quantizers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return an opset-19 copy of older standard Q/DQ graphs for FP16 preparation."""
    imports = {item.domain: item.version for item in model.opset_import}
    if imports.get("", imports.get("ai.onnx", 0)) >= 19 or not any(
        node.domain in {"", "ai.onnx"} and node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in model.graph.node
    ):
        return model
    return version_converter.convert_version(model, target_version=19)
