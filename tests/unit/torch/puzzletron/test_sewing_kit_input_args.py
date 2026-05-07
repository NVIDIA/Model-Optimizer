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

"""Unit tests for ``sewing_kit.passage.InputArgs``.

``InputArgs`` is the workhorse args/kwargs container the bypass distillation
factory uses inside its stitching reducers — see ``bypass_factory_fn`` calls
like ``lambda acc, override, orig, *args: override + orig.drop_args(0)``.
A regression in ``__add__`` or ``drop_args`` would silently corrupt the
inputs passed into per-block forward passes, producing wrong loss values
without any loud failure.
"""

import pytest

from modelopt.torch.puzzletron.sewing_kit.passage import InputArgs


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_accepts_positional_and_keyword_args():
    ia = InputArgs(1, 2, foo="bar")
    assert ia.args == [1, 2]
    assert ia.kwargs == {"foo": "bar"}


def test_init_with_no_args_is_empty():
    ia = InputArgs()
    assert ia.args == []
    assert ia.kwargs == {}


# ---------------------------------------------------------------------------
# __add__: concatenates args, merges kwargs (right wins on collision)
# ---------------------------------------------------------------------------


def test_add_concatenates_positional_args_in_order():
    a = InputArgs(1, 2)
    b = InputArgs(3, 4)
    result = a + b
    assert result.args == [1, 2, 3, 4]
    assert result.kwargs == {}


def test_add_merges_kwargs_with_right_winning():
    """Bypass reducers chain ``override + orig.drop_args(0)`` — when both sides
    happen to set the same kwarg, the right-side value (the original input)
    must win, otherwise the override silently displaces the original kwarg."""
    a = InputArgs(foo="from_a", bar="only_a")
    b = InputArgs(foo="from_b", baz="only_b")
    result = a + b
    assert result.kwargs == {"foo": "from_b", "bar": "only_a", "baz": "only_b"}


def test_add_does_not_mutate_operands():
    a = InputArgs(1, 2, x="a")
    b = InputArgs(3, y="b")
    _ = a + b
    assert a.args == [1, 2] and a.kwargs == {"x": "a"}
    assert b.args == [3] and b.kwargs == {"y": "b"}


def test_add_rejects_non_input_args():
    with pytest.raises(AssertionError):
        InputArgs(1) + [2]  # type: ignore[operator]


# ---------------------------------------------------------------------------
# drop_args: clears all positional args (default) or one by index/slice
# ---------------------------------------------------------------------------


def test_drop_args_default_clears_all_positional():
    """The ``drop_args(0)`` and ``drop_args()`` forms are both used by bypass
    stitches — the default-no-arg form must wipe the entire positional tuple
    (kwargs untouched)."""
    ia = InputArgs(1, 2, 3, foo="bar")
    out = ia.drop_args()
    assert out.args == []
    assert out.kwargs == {"foo": "bar"}
    # And the original is unmodified.
    assert ia.args == [1, 2, 3]


def test_drop_args_with_index_drops_one():
    ia = InputArgs(10, 20, 30)
    out = ia.drop_args(0)
    assert out.args == [20, 30]
    # Source preserved.
    assert ia.args == [10, 20, 30]


def test_drop_args_with_slice_drops_range():
    ia = InputArgs(10, 20, 30, 40)
    out = ia.drop_args(slice(1, 3))
    assert out.args == [10, 40]


# ---------------------------------------------------------------------------
# drop_kwargs: clears all kwargs (default) or specific keys
# ---------------------------------------------------------------------------


def test_drop_kwargs_default_clears_all():
    ia = InputArgs(1, foo="bar", baz="qux")
    out = ia.drop_kwargs()
    assert out.args == [1]
    assert out.kwargs == {}


def test_drop_kwargs_with_keys_drops_only_those():
    ia = InputArgs(1, foo="bar", baz="qux", keep="this")
    out = ia.drop_kwargs(["foo", "baz"])
    assert out.kwargs == {"keep": "this"}


def test_drop_kwargs_silently_ignores_missing_keys():
    """A key listed in ``drop_kwargs`` that isn't present must not raise —
    bypass calls this against args from arbitrary upstream stitches and may
    pass keys that only some sources produce."""
    ia = InputArgs(foo="bar")
    out = ia.drop_kwargs(["nonexistent"])  # must not KeyError
    assert out.kwargs == {"foo": "bar"}


# ---------------------------------------------------------------------------
# from_value: lifts assorted values into InputArgs
# ---------------------------------------------------------------------------


def test_from_value_passes_through_existing_input_args():
    ia = InputArgs(1, foo="bar")
    out = InputArgs.from_value(ia)
    assert out is ia


def test_from_value_lifts_sequence_to_positional_args():
    out = InputArgs.from_value([1, 2, 3])
    assert out.args == [1, 2, 3]
    assert out.kwargs == {}


def test_from_value_lifts_scalar_to_single_positional():
    out = InputArgs.from_value(42)
    assert out.args == [42]
    assert out.kwargs == {}
