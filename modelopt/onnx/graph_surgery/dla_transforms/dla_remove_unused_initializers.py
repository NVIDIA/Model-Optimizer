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

"""Legacy transform: remove unused ONNX graph initializers."""

import onnx

from ...logging_config import logger
from ._common import run_onnx_file_transform


def _apply_remove_unused_initializers(model):
    """Remove initializers that are not used as inputs to any node in the graph.

    Args:
        model: An ONNX ModelProto object

    Returns:
        The modified model with unused initializers removed

    """
    graph = model.graph

    # Collect all node inputs
    used_inputs = set()
    for node in graph.node:
        used_inputs.update(node.input)

    # Also consider graph outputs as used
    for output in graph.output:
        used_inputs.add(output.name)

    # Find initializers that are used
    used_initializers = []
    removed_count = 0

    for initializer in graph.initializer:
        if initializer.name in used_inputs:
            used_initializers.append(initializer)
        else:
            removed_count += 1

    # Clear and reset initializers
    graph.ClearField("initializer")
    graph.initializer.extend(used_initializers)

    logger.debug("Removed %d unused initializers", removed_count)
    return model


def dla_remove_unused_initializers(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, apply graph transform, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_remove_unused_initializers,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
