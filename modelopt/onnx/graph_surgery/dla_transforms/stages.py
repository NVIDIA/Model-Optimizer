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
"""Keep fixed rule batches and direct IR passes in their required phase order."""

from collections.abc import Collection

from onnxscript import ir

from . import cleanup as _cleanup
from .context import DlaGraph

__all__ = []


def cleanup(graph: DlaGraph, *, preserve_identities: Collection[ir.Node] | None = None) -> int:
    """Collapse fixed layout chains while preserving original/public identity ownership."""
    names = {}
    count = graph.apply_patterns(
        [
            _cleanup.ReshapeChain.rule(graph, names),
            _cleanup.IdentityReshape.rule(graph, names),
            _cleanup.InternalIdentity.rule(graph, names, preserve_identities=preserve_identities),
        ]
    )
    graph.apply_pass(_cleanup.RestoreValueNames(names))
    graph.apply_pass(ir.passes.common.RemoveUnusedNodesPass())
    graph.apply_pass(ir.passes.common.TopologicalSortPass())
    return count
