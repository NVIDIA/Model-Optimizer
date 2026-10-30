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

import gc
import weakref

import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.graph_indexing import has_path_type


def test_path_matching_does_not_retain_graph_nodes():
    node = gs.Node(op="Relu", name="relu")
    graph = gs.Graph(nodes=[node])
    reference = weakref.ref(node)
    assert has_path_type(node, graph, ["Relu"], True)
    path = []
    assert has_path_type(node, graph, ["Relu"], True, path_nodes=path)
    assert path == [node]
    del node, graph, path
    gc.collect()
    assert reference() is None
