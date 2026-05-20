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

"""Unit tests for ``sewing_kit.core.Needle`` graph construction and validation.

The bypass factory builds three ``Needle``\\s per rank (teacher train, teacher
val, student val) and calls ``Needle.knot()`` on each. ``knot()`` runs
``_validate_nodes`` first; a regression in that validation would either crash
with an opaque NoneType error during forward, or — worse — silently allow a
malformed graph that produces incorrect activations.

We test the validation contract on CPU without instantiating ``StitchedModule``
itself (which requires Module patching). ``_validate_nodes`` is a private
method but it's the unit of behavior worth pinning; ``knot()`` is essentially
``_validate_nodes() + StitchedModule(...)``.
"""

import pytest
import torch.nn as nn

from modelopt.torch.puzzletron.sewing_kit.core import (
    ExternalTarget,
    InputsLoopFoundException,
    ModuleTarget,
    Needle,
    Node,
    OnlyInternalNodesException,
    StitchDescriptor,
)

# ---------------------------------------------------------------------------
# get_node_for_target: lazy creation, cached lookup
# ---------------------------------------------------------------------------


def test_get_node_for_target_creates_node_on_first_call():
    needle = Needle()
    target = ModuleTarget("a", nn.Linear(2, 2))
    node = needle.get_node_for_target(target)
    assert isinstance(node, Node)
    assert node.target is target
    assert needle.nodes[target] is node


def test_get_node_for_target_returns_same_node_on_repeat_call():
    """Re-getting the same target must NOT create a duplicate node — every
    stitch involving that target must funnel into a single Node, otherwise
    the validation/forward graph fragments."""
    needle = Needle()
    target = ModuleTarget("a", nn.Linear(2, 2))
    node1 = needle.get_node_for_target(target)
    node2 = needle.get_node_for_target(target)
    assert node1 is node2
    assert len(needle.nodes) == 1


# ---------------------------------------------------------------------------
# stitch: adds StitchDescriptor to source.stitches_from and dest.stitches_to
# ---------------------------------------------------------------------------


def test_stitch_records_descriptor_on_both_endpoints():
    needle = Needle()
    target_a = ModuleTarget("a", nn.Linear(2, 2))
    target_b = ModuleTarget("b", nn.Linear(2, 2))

    needle.stitch(target_a.output("x"), target_b.input("y"))

    node_a = needle.nodes[target_a]
    node_b = needle.nodes[target_b]
    # Source endpoint: A has one outgoing stitch; B has one incoming stitch.
    assert len(node_a.stitches_from) == 1
    assert len(node_a.stitches_to) == 0
    assert len(node_b.stitches_from) == 0
    assert len(node_b.stitches_to) == 1
    # Same StitchDescriptor object on both lists.
    assert node_a.stitches_from[0] is node_b.stitches_to[0]
    assert isinstance(node_a.stitches_from[0], StitchDescriptor)


def test_stitch_returns_self_for_chaining():
    """Bypass factory chains ``.stitch(...).stitch(...)`` — the return type
    must be the Needle itself so the second call sees the same graph."""
    needle = Needle()
    target_a = ModuleTarget("a", nn.Linear(2, 2))
    target_b = ModuleTarget("b", nn.Linear(2, 2))

    out = needle.stitch(target_a.output("x"), target_b.input("y"))
    assert out is needle


# ---------------------------------------------------------------------------
# _validate_nodes: contract checks before knot() builds the StitchedModule
# ---------------------------------------------------------------------------


def test_validate_raises_when_only_internal_nodes_present():
    """A graph with no External and no Remote target has nothing for the
    runtime to feed inputs through — must raise loudly rather than build a
    dead StitchedModule."""
    needle = Needle()
    target_a = ModuleTarget("a", nn.Linear(2, 2))
    target_b = ModuleTarget("b", nn.Linear(2, 2))
    needle.stitch(target_a.output("x"), target_b.input("y"))

    with pytest.raises(OnlyInternalNodesException):
        needle._validate_nodes()


def test_validate_passes_with_external_plus_dag():
    """Happy path: ExternalTarget + a small linear DAG. Must not raise."""
    needle = Needle()
    ext = ExternalTarget()
    target_a = ModuleTarget("a", nn.Linear(2, 2))
    target_b = ModuleTarget("b", nn.Linear(2, 2))

    needle.stitch(ext.output("init"), target_a.input("entry"))
    needle.stitch(target_a.output("x"), target_b.input("y"))
    needle.stitch(target_b.output("z"), ext.input("final"))

    # No raise.
    needle._validate_nodes()


def test_validate_raises_on_input_cycle_among_internal_nodes():
    """Detect a 2-node cycle A→B→A among internal nodes.

    The validation uses ``_search_loops`` walking ``stitches_to`` (incoming
    edges); ExternalTarget short-circuits the recursion, so we add an
    external feed to A so ``_validate_nodes`` doesn't bail out early on the
    'no external' check.
    """
    needle = Needle()
    ext = ExternalTarget()
    target_a = ModuleTarget("a", nn.Linear(2, 2))
    target_b = ModuleTarget("b", nn.Linear(2, 2))

    # Anchor an external feed so we get past the OnlyInternalNodes check.
    needle.stitch(ext.output("init"), target_a.input("entry"))
    # Cycle: A -> B -> A.
    needle.stitch(target_a.output("x"), target_b.input("y"))
    needle.stitch(target_b.output("p"), target_a.input("q"))

    with pytest.raises(InputsLoopFoundException):
        needle._validate_nodes()


def test_validate_passes_when_external_node_has_self_referential_loop_via_external():
    """``_search_loops`` short-circuits at ExternalTarget. So a 'loop' that
    only goes through external (e.g. external→A and A→external) is fine —
    and indeed required for normal stitching, where external is both the
    input and output endpoint.
    """
    needle = Needle()
    ext = ExternalTarget()
    target_a = ModuleTarget("a", nn.Linear(2, 2))

    needle.stitch(ext.output("in"), target_a.input("entry"))
    needle.stitch(target_a.output("x"), ext.input("out"))

    # Despite the external→A→external pattern, this is the canonical bypass
    # shape and must validate clean.
    needle._validate_nodes()


# ---------------------------------------------------------------------------
# Sanity: ExternalTarget.input()/output() builds correctly typed descriptors
# ---------------------------------------------------------------------------


def test_module_target_descriptors_carry_target_and_name():
    """The ``.input("foo")`` and ``.output("bar")`` builders are what the
    bypass factory uses to construct stitches. They must propagate the
    target reference and the name into the resulting descriptor so the
    runtime can route values correctly."""
    target = ModuleTarget("a", nn.Linear(2, 2))
    in_desc = target.input("foo")
    out_desc = target.output("bar")
    assert in_desc.target is target
    assert in_desc.input_name == "foo"
    assert out_desc.target is target
    assert out_desc.output_name == "bar"
