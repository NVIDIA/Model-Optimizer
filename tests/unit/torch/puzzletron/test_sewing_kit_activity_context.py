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

"""Unit tests for ``sewing_kit.utils.ActivityContext``.

``ActivityContext`` is the stack the ``Passage`` machinery uses to track which
passages are currently active inside a ``StitchedModule.forward`` call. A bug
in push/pop ordering or in the exception-safe cleanup would leak state across
forward passes — every subsequent block would see a stale "active passage"
and route inputs/outputs to the wrong module.
"""

import pytest

from modelopt.torch.puzzletron.sewing_kit.utils import (
    ActivityContext,
    ActivityContextDuplicateException,
    ActivityContextMaxDepthException,
)


# ---------------------------------------------------------------------------
# Basic push/pop semantics via the ``with ctx(value):`` form
# ---------------------------------------------------------------------------


def test_starts_empty_and_inactive():
    ctx: ActivityContext[str] = ActivityContext()
    assert len(ctx) == 0
    assert not ctx.is_active()
    assert ctx.get_active() is None


def test_with_block_pushes_and_pops_value():
    ctx: ActivityContext[str] = ActivityContext()
    with ctx("a"):
        assert ctx.is_active()
        assert ctx.get_active() == "a"
        assert "a" in ctx
        assert len(ctx) == 1
    # After the block: stack must be back to empty.
    assert len(ctx) == 0
    assert ctx.get_active() is None


def test_nested_pushes_track_lifo_order():
    """``get_active`` returns the *most recent* push (LIFO) — Passage relies on
    this to find the innermost active passage during forward."""
    ctx: ActivityContext[str] = ActivityContext()
    with ctx("outer"):
        assert ctx.get_active() == "outer"
        with ctx("inner"):
            assert ctx.get_active() == "inner"
            assert ctx[0] == "outer"
            assert ctx[1] == "inner"
        # Inner pop returns to outer.
        assert ctx.get_active() == "outer"


# ---------------------------------------------------------------------------
# max_depth: limits stack height
# ---------------------------------------------------------------------------


def test_max_depth_one_allows_single_push():
    ctx: ActivityContext[str] = ActivityContext(max_depth=1)
    with ctx("a"):
        assert ctx.get_active() == "a"


def test_max_depth_one_rejects_second_push():
    ctx: ActivityContext[str] = ActivityContext(max_depth=1)
    with ctx("a"):
        with pytest.raises(ActivityContextMaxDepthException):
            with ctx("b"):
                pass
    # Stack must have unwound to empty even after the exception.
    assert len(ctx) == 0


# ---------------------------------------------------------------------------
# no_duplicates: same value can't appear twice
# ---------------------------------------------------------------------------


def test_no_duplicates_rejects_repeat_value():
    ctx: ActivityContext[str] = ActivityContext(no_duplicates=True)
    with ctx("x"):
        with pytest.raises(ActivityContextDuplicateException):
            with ctx("x"):
                pass
    # Stack unwound; the still-active "x" was preserved through the failed push.
    assert len(ctx) == 0


def test_no_duplicates_allows_distinct_values():
    ctx: ActivityContext[str] = ActivityContext(no_duplicates=True)
    with ctx("x"):
        with ctx("y"):
            assert "x" in ctx and "y" in ctx


# ---------------------------------------------------------------------------
# reversed=True: insert at front, pop from front
# ---------------------------------------------------------------------------


def test_reversed_pushes_to_front_and_pops_from_front():
    """``Passage.active_passages_context`` uses ``reversed=True`` so the
    *first* active passage in iteration order is the innermost. Pin both
    insert position and pop position."""
    ctx: ActivityContext[str] = ActivityContext(reversed=True)
    with ctx("a"):
        with ctx("b"):
            # b inserted at front of stack.
            assert ctx[0] == "b"
            assert ctx[1] == "a"
        # Pop from front: only "a" left.
        assert list(ctx[:]) == ["a"]


# ---------------------------------------------------------------------------
# Exception safety: stack unwinds even if the caller's body raises
# ---------------------------------------------------------------------------


def test_stack_unwinds_when_body_raises():
    """A bug here would leak stack frames — the next forward pass would see
    a stale active passage. This is the silent-failure scenario."""
    ctx: ActivityContext[str] = ActivityContext()
    with pytest.raises(ValueError, match="boom"):
        with ctx("a"):
            assert ctx.get_active() == "a"
            raise ValueError("boom")
    assert len(ctx) == 0


# ---------------------------------------------------------------------------
# is_submodule_of / is_submodule_or_same — string predicates used by passage.py
# ---------------------------------------------------------------------------


from modelopt.torch.puzzletron.sewing_kit.utils import (  # noqa: E402
    is_submodule_of,
    is_submodule_or_same,
)


def test_is_submodule_of_proper_descendant():
    assert is_submodule_of("model.layers.0.self_attn", "model.layers.0")
    assert is_submodule_of("model.layers.0", "model")
    # Empty string parent matches any non-empty name (root-of-everything case).
    assert is_submodule_of("model", "")


def test_is_submodule_of_rejects_self_and_unrelated():
    assert not is_submodule_of("model.layers.0", "model.layers.0")
    assert not is_submodule_of("model.layers.0", "model.layers.1")
    # Empty == empty is not a submodule relationship.
    assert not is_submodule_of("", "")
    # Prefix collision: "model.layers" is NOT a submodule of "model.lay" — the
    # predicate requires a literal "." separator after the parent.
    assert not is_submodule_of("model.layers", "model.lay")


def test_is_submodule_or_same_includes_equality():
    assert is_submodule_or_same("model.layers.0", "model.layers.0")
    assert is_submodule_or_same("model.layers.0.attn", "model.layers.0")
    assert not is_submodule_or_same("model.layers.0", "model.layers.1")
