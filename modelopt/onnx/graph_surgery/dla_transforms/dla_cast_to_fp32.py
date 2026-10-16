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

"""Redirect all Cast nodes to produce float32 output.

Skip conditions
---------------
A Cast node is left untouched when its output is consumed by an op that
requires a specific non-float32 dtype:

* **Gather / GatherElements** ``input[1]`` (indices) — must stay INT32.
* **GroupQueryAttention** inputs — require FP16.

These Cast nodes are inserted by earlier transforms
(``dla-convert-ops-to-4d`` for Gather/GatherElements indices, or
externally for GQA) and must not be overridden.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto

from ...logging_config import logger
from ._common import get_tensor_dtype_by_name, run_onnx_file_transform


def _apply_cast_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """Change every eligible Cast node's ``to`` attribute to FLOAT (float32).

    A Cast node is skipped when its output feeds:
    - ``input[1]`` (indices) of a Gather or GatherElements node, or
    - any input of a GroupQueryAttention node.
    """
    graph = model.graph

    # ── Build set of protected Cast outputs ──────────────────────────────────
    # Tensor names produced by Cast nodes that must keep their current target
    # dtype: INT32 for Gather/GatherElements indices, FP16 for GQA inputs.
    protected: set[str] = set()
    for node in graph.node:
        if node.op_type in ("Gather", "GatherElements"):
            # input[1] is the indices tensor — must stay INT32
            if (
                len(node.input) > 1
                and node.input[1]
                and get_tensor_dtype_by_name(model, node.input[1]) == TensorProto.INT32
            ):
                protected.add(node.input[1])
        elif node.op_type == "GroupQueryAttention":
            for inp in node.input:
                if inp:
                    protected.add(inp)

    # ── Redirect eligible Cast nodes to FLOAT ────────────────────────────────
    cnt = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        if not node.output or node.output[0] in protected:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i != TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT
                # Update value_info dtype for the Cast output tensor
                for vi in graph.value_info:
                    if vi.name == node.output[0]:
                        vi.type.tensor_type.elem_type = TensorProto.FLOAT
                        break
                cnt += 1
                break

    logger.debug("cast_to_fp32: redirected %d Cast node(s) to float32.", cnt)
    return model


def dla_cast_to_fp32(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, redirect Cast nodes to float32, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_cast_to_fp32,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
