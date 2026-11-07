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

"""Fixed activation Cast rules used by optional FP16 preparation."""

from typing import TYPE_CHECKING

from onnxscript import ir
from onnxscript.rewriter import pattern

if TYPE_CHECKING:
    from ..context import DlaGraph

__all__ = ["Fp16Cast"]


class Fp16Cast(pattern.RewriteRuleClassBase):
    """Retarget eligible internal ``Cast(to=FLOAT)`` operations to FLOAT16.

    Public FLOAT endpoints and the FP32 regions retained by schema/native
    analysis keep their original dtype. For example, an explicit input Cast
    feeding FP16 arithmetic becomes its existing FP16 boundary, avoiding an
    additional conversion. The target attribute prevents repeated rewriting.
    """

    def __init__(self, graph: "DlaGraph", retained: set[ir.Value]):
        """Bind the FP32 values retained by this conversion's precision analysis."""
        super().__init__(name="DlaFp16Cast")
        self.retained = retained

    def pattern(self, op, source):
        """Match an explicit FLOAT Cast eligible for FP16 retargeting."""
        return op.Cast(source, to=int(ir.DataType.FLOAT), _outputs=["value"])

    def check(self, context, source, value):
        """Keep public FLOAT outputs and required native regions unchanged."""
        return not value.is_graph_output() and value not in self.retained

    def rewrite(self, op, source, value):
        """Emit the corresponding FLOAT16 Cast through the SDK replacement path."""
        return op.Cast(source, to=int(ir.DataType.FLOAT16))
