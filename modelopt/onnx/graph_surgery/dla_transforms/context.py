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

"""Per-run graph access and phase metadata shared by DLA rules and passes."""

from onnxscript import ir
from onnxscript.rewriter import pattern

from .common import initializer_factory

__all__ = ["DlaGraph"]


class DlaGraph:
    """Keep one live IR model and capture native layouts at layout-phase entry."""

    def __init__(self, model: ir.Model, *, force_fp16_activations: bool = False):
        """Bind live graph access and initializer naming to this conversion only."""
        self.model = model
        self.force_fp16_activations = force_fp16_activations
        self.add_initializer = initializer_factory(model.graph)
        self.batch_nodes = frozenset(model.graph)
        self.native_shapes = {}
        self.planned_shapes = {}
        self.public_outputs = []

    @property
    def initializers(self):
        """Return the current initializer table, including later additions/removals."""
        return self.model.graph.initializers

    @property
    def inputs(self):
        """Return current public input values."""
        return self.model.graph.inputs

    @property
    def outputs(self):
        """Return current public output values."""
        return self.model.graph.outputs

    def apply_pattern(self, rule: pattern.RewriteRule) -> int:
        """Apply a single-rule batch with its own snapshot of existing nodes."""
        return self.apply_patterns([rule])

    def apply_patterns(self, rules: list[pattern.RewriteRule]) -> int:
        """Apply one ordered SDK batch; generated nodes remain available to its rules."""
        self.batch_nodes = frozenset(self.model.graph)
        try:
            return pattern.RewriteRuleSet(rules).apply_to_model(self.model) if rules else 0
        finally:
            self.batch_nodes = frozenset()

    def apply_pass(self, transform: ir.passes.InPlacePass) -> ir.passes.PassResult:
        """Run a direct in-place pass through the SDK's contract checks."""
        return transform(self.model)
