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

from collections.abc import Collection, Sequence

from onnxscript import ir
from onnxscript.rewriter import pattern

from . import cleanup as _cleanup
from .context import DlaGraph
from .operators import rules as operator_rules
from .preprocessing import passes as preprocessing_passes
from .preprocessing import rules as preprocessing_rules

__all__ = []


def apply_rules(
    graph: DlaGraph, definitions: Sequence[type[pattern.RewriteRuleClassBase]], **options
) -> int:
    """Bind per-run state to one fixed SDK rule batch without discovering operators."""
    return graph.apply_patterns([definition.rule(graph, **options) for definition in definitions])


def preprocessing(graph: DlaGraph, *, quantized_dtypes=None, remove_all_qdq=False) -> int:
    """Prepare quantization, constants and connected FP32 regions in their original order."""
    names = {}
    count = graph.apply_patterns(
        [
            preprocessing_rules.SelectedQdq.rule(
                graph,
                domain,
                quantized_dtypes=quantized_dtypes,
                remove_all_qdq=remove_all_qdq,
                source_names=names,
            )
            for domain in ("", "com.microsoft")
        ]
    )
    graph.apply_pass(_cleanup.RestoreValueNames(names))
    graph.apply_pass(ir.passes.common.RemoveUnusedNodesPass())
    count += graph.apply_pattern(preprocessing_rules.ConstantInitializer.rule(graph))
    analysis = preprocessing_passes.AnalyzeFp32Casts()
    graph.apply_pass(analysis)
    if analysis.candidates:
        count += graph.apply_pattern(preprocessing_rules.Fp32Cast.rule(graph, analysis))
        graph.apply_pass(preprocessing_passes.PromoteFp32Values(analysis))
    folded = {}
    count += graph.apply_patterns(
        [
            preprocessing_rules.StaticDequantization.rule(graph, domain, folded=folded)
            for domain in ("", "com.microsoft")
        ]
    )
    graph.apply_pass(ir.passes.common.RemoveUnusedNodesPass())
    return count


def individual_op_rewrites(graph: DlaGraph) -> int:
    """Lower native operators, coordinate layouts, then lower logical arithmetic."""
    count = 0
    for definitions in (
        operator_rules.INDEXING_RULES,
        operator_rules.MATRIX_RULES,
    ):
        count += apply_rules(graph, definitions, views={})
    graph.apply_pass(ir.passes.common.RemoveUnusedNodesPass())
    return count


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
