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

import os
import re
import stat
from pathlib import Path

import onnx

from modelopt.onnx.utils import topologically_sort_graph_nodes

__all__ = ["find_vovnet_nodes_to_exclude"]

MAX_ONNX_BYTES = 512 << 20


def _load_onnx_graph(onnx_path, max_onnx_bytes=MAX_ONNX_BYTES):
    if max_onnx_bytes < 1:
        raise ValueError("ONNX byte limit must be positive")
    path = Path(onnx_path)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Unable to open {path} as an ONNX file") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{path} is not a regular ONNX file")
        if file_stat.st_size > max_onnx_bytes:
            raise ValueError(f"{path} exceeds the ONNX byte limit")
        with os.fdopen(descriptor, "rb") as onnx_file:
            model_bytes = onnx_file.read(max_onnx_bytes + 1)
            descriptor = None
        if len(model_bytes) > max_onnx_bytes:
            raise ValueError(f"{path} exceeds the ONNX byte limit")
        return onnx.load_model_from_string(model_bytes).graph
    finally:
        if descriptor is not None:
            os.close(descriptor)


def find_vovnet_nodes_to_exclude(onnx_path, max_onnx_bytes=MAX_ONNX_BYTES):
    """Find the VoVNet OSA4_5 stage and nodes downstream of FPN lateral_convs."""
    graph = _load_onnx_graph(onnx_path, max_onnx_bytes)
    topologically_sort_graph_nodes(graph)

    excluded = set()
    downstream_tensors = set()
    for node in graph.node:
        is_osa = "OSA4_5" in node.name
        is_downstream = any(name in downstream_tensors for name in node.input)
        if is_osa or is_downstream:
            excluded.add(node.name)
        if "lateral_convs" in node.name or (is_downstream and not is_osa):
            downstream_tensors.update(node.output)

    if not excluded:
        raise ValueError(f"No accuracy-sensitive VoVNet nodes found in {onnx_path}")
    return [rf"^{re.escape(name)}$" for name in sorted(excluded)]
