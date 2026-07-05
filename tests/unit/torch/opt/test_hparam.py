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

"""Unit tests for the ``Hparam`` primitive in ``modelopt.torch.opt.hparam``."""

import pickle

import pytest
import torch

from modelopt.torch.opt.hparam import Hparam

# -----------------------------------------------------------------------------------------------
# Construction / basic properties
# -----------------------------------------------------------------------------------------------


def test_init_defaults_original_to_max():
    hp = Hparam([8, 16, 32])
    assert hp.original == 32
    assert hp.active == 32
    assert hp.choices == [8, 16, 32]


def test_init_with_explicit_original():
    hp = Hparam([8, 16, 32], original=16)
    assert hp.original == 16
    assert hp.active == 16
    assert hp.choices == [8, 16, 32]


def test_init_adds_original_to_choices():
    # the original value is always folded into the (sorted) choices
    hp = Hparam([1, 2, 3], original=5)
    assert hp.choices == [1, 2, 3, 5]
    assert hp.original == 5
    assert hp.active == 5


def test_init_deduplicates_and_sorts_choices():
    hp = Hparam([3, 1, 2, 3, 1, 2])
    assert hp.choices == [1, 2, 3]


def test_iter_yields_choices_in_order():
    hp = Hparam([32, 8, 16])
    assert list(hp) == [8, 16, 32]


def test_min_max():
    hp = Hparam([16, 4, 8])
    assert hp.min == 4
    assert hp.max == 16


def test_tuple_valued_choices():
    hp = Hparam([(1, 1), (2, 2), (3, 3)])
    assert hp.original == (3, 3)
    assert hp.min == (1, 1)
    assert hp.max == (3, 3)
    hp.active = (2, 2)
    assert hp.active == (2, 2)


def test_repr():
    hp = Hparam([1, 2, 3])
    assert repr(hp) == "Hparam(choices=[1, 2, 3], active=3, original=3)"


# -----------------------------------------------------------------------------------------------
# Configurability
# -----------------------------------------------------------------------------------------------


def test_is_configurable_multiple_choices():
    assert Hparam([1, 2]).is_configurable


def test_is_configurable_single_choice():
    assert not Hparam([7]).is_configurable


def test_is_configurable_manual_override():
    hp = Hparam([1, 2, 3])
    hp._is_configurable = False
    assert not hp.is_configurable


def test_force_configurable_context_manager():
    hp = Hparam([1, 2, 3])
    hp._is_configurable = False
    with hp._force_configurable():
        assert hp.is_configurable
        hp.active = 1
    assert not hp.is_configurable
    assert hp.active == 1


# -----------------------------------------------------------------------------------------------
# Active value
# -----------------------------------------------------------------------------------------------


def test_active_setter_valid_choice():
    hp = Hparam([8, 16, 32])
    hp.active = 8
    assert hp.active == 8


def test_active_setter_none_resets_to_original():
    hp = Hparam([8, 16, 32], original=16)
    hp.active = 8
    hp.active = None
    assert hp.active == 16


def test_active_setter_invalid_choice():
    hp = Hparam([8, 16, 32])
    with pytest.raises(AssertionError, match=r"val = 5, choices = \[8, 16, 32\]"):
        hp.active = 5


def test_active_setter_non_configurable():
    hp = Hparam([8, 16, 32])
    hp.active = 16
    hp._is_configurable = False
    hp.active = 16  # same value is allowed
    assert hp.active == 16
    with pytest.raises(AssertionError):
        hp.active = 8  # different value is not


# -----------------------------------------------------------------------------------------------
# Active slice and order enforcement
# -----------------------------------------------------------------------------------------------


def test_active_slice_default_is_slice():
    hp = Hparam([1, 2, 3, 4])
    hp.active = 2
    assert hp.active_slice == slice(2)


def test_active_slice_requires_int_active():
    hp = Hparam([(1, 1), (2, 2)])
    with pytest.raises(AssertionError, match="active_slice only supported for int hparams"):
        _ = hp.active_slice


def test_enforce_order_changes_active_slice():
    hp = Hparam([1, 2, 3, 4])
    hp.enforce_order(torch.tensor([3, 1, 0, 2]))
    hp.active = 2
    active_slice = hp.active_slice
    assert isinstance(active_slice, torch.Tensor)
    assert torch.equal(active_slice, torch.tensor([3, 1]))


def test_enforce_order_converts_to_long_cpu():
    hp = Hparam([1, 2, 3])
    hp.enforce_order(torch.tensor([2.0, 0.0, 1.0]))  # float order is converted
    assert hp._slice_order.dtype == torch.long
    assert hp._slice_order.device.type == "cpu"
    assert torch.equal(hp.active_slice, torch.tensor([2, 0, 1]))


def test_enforce_order_invalid_permutation():
    hp = Hparam([1, 2, 3, 4])
    with pytest.raises(AssertionError, match="order must be a permutation"):
        hp.enforce_order(torch.tensor([0, 1, 1, 2]))


def test_enforce_order_wrong_length_is_invalid():
    hp = Hparam([1, 2, 3, 4])
    with pytest.raises(AssertionError, match="order must be a permutation"):
        hp.enforce_order(torch.tensor([1, 0]))


def test_enforce_order_none_resets_to_slice():
    hp = Hparam([1, 2, 3])
    hp.enforce_order(torch.tensor([2, 1, 0]))
    hp.enforce_order(None)
    assert hp.active_slice == slice(3)


# -----------------------------------------------------------------------------------------------
# Choices setter
# -----------------------------------------------------------------------------------------------


def test_choices_setter_shrinks_to_subset():
    hp = Hparam([8, 16, 32])
    hp.choices = [16, 32]
    assert hp.choices == [16, 32]
    assert hp.min == 16
    assert hp.max == 32


def test_choices_setter_requires_original():
    hp = Hparam([8, 16, 32])  # original == 32
    with pytest.raises(AssertionError, match="Original choice not in choices"):
        hp.choices = [8, 16]


def test_choices_setter_requires_active():
    hp = Hparam([8, 16, 32])
    hp.active = 8
    with pytest.raises(AssertionError, match="Active choice not in choices"):
        hp.choices = [16, 32]


def test_choices_setter_rejects_new_values():
    hp = Hparam([8, 16, 32])
    with pytest.raises(AssertionError, match="New choices must be a subset"):
        hp.choices = [8, 16, 32, 64]


def test_choices_setter_non_configurable_requires_equality():
    hp = Hparam([8, 16, 32])
    hp._is_configurable = False
    hp.choices = [32, 16, 8]  # same set is fine
    assert hp.choices == [8, 16, 32]
    with pytest.raises(AssertionError, match="Cannot update choices"):
        hp.choices = [16, 32]


def test_choices_setter_single_choice_hparam():
    hp = Hparam([5])  # not configurable since only one choice
    hp.choices = [5]
    assert hp.choices == [5]
    with pytest.raises(AssertionError, match="Cannot update choices"):
        hp.choices = [5, 6]


# -----------------------------------------------------------------------------------------------
# Importance
# -----------------------------------------------------------------------------------------------


def test_importance_default_is_none():
    hp = Hparam([1, 2, 3])
    assert hp.is_sortable
    assert hp.importance is None


def test_importance_requires_configurable():
    hp = Hparam([3])
    with pytest.raises(AssertionError):
        _ = hp.importance


def test_register_importance_normalizes():
    hp = Hparam([1, 2, 3])
    hp.register_importance(lambda: torch.tensor([1.0, 2.0, 4.0]))
    imp = hp.importance
    assert imp.shape == (3,)
    assert torch.allclose(imp, torch.tensor([0.25, 0.5, 1.0]), atol=1e-6)


def test_register_importance_sums_multiple_estimators():
    hp = Hparam([1, 2, 3])
    hp.register_importance(lambda: torch.tensor([1.0, 2.0, 4.0]))
    hp.register_importance(lambda: torch.tensor([3.0, 3.0, 3.0]))
    imp = hp.importance
    # each estimator is normalized by its max before summation
    assert torch.allclose(imp, torch.tensor([1.25, 1.5, 2.0]), atol=1e-6)


def test_importance_length_must_match_max():
    hp = Hparam([1, 2, 3])
    hp.register_importance(lambda: torch.ones(2))
    with pytest.raises(AssertionError, match="Length of importance must be equal to max choice!"):
        _ = hp.importance


def test_register_importance_unsortable_raises():
    hp = Hparam([1, 2, 3])
    hp._importance_estimators = None
    assert not hp.is_sortable
    with pytest.raises(RuntimeError, match="Cannot register importance for unsortable hparams"):
        hp.register_importance(lambda: torch.ones(3))


def test_importance_as_order_returns_raw_estimate():
    hp = Hparam([1, 2, 3])
    hp._importance_is_order = True
    order = torch.tensor([2.0, 0.0, 1.0])
    hp.register_importance(lambda: order)
    assert torch.equal(hp.importance, order)  # no normalization applied


def test_importance_as_order_single_estimator_only():
    hp = Hparam([1, 2, 3])
    hp._importance_is_order = True
    hp.register_importance(lambda: torch.ones(3))
    hp.register_importance(lambda: torch.ones(3))
    with pytest.raises(AssertionError, match="Only one importance estimator is supported"):
        _ = hp.importance


# -----------------------------------------------------------------------------------------------
# Merging (__iand__)
# -----------------------------------------------------------------------------------------------


def test_iand_merges_choices_to_intersection():
    hp1 = Hparam([1, 2, 3])
    hp2 = Hparam([2, 3, 4], original=3)
    hp1 &= hp2
    assert hp1.choices == [2, 3]
    assert hp1.original == 3
    assert hp2.choices == [2, 3, 4]  # other hparam is untouched


def test_iand_removes_slice_order():
    hp1 = Hparam([1, 2, 3])
    hp1.enforce_order(torch.tensor([2, 1, 0]))
    hp1 &= Hparam([1, 2, 3])
    assert hp1._slice_order is None
    assert hp1.active_slice == slice(3)


def test_iand_merges_importance_estimators():
    hp1 = Hparam([1, 2, 3])
    hp2 = Hparam([1, 2, 3])
    hp2.register_importance(lambda: torch.tensor([1.0, 2.0, 4.0]))
    hp1 &= hp2
    assert torch.allclose(hp1.importance, torch.tensor([0.25, 0.5, 1.0]), atol=1e-6)


def test_iand_with_unsortable_makes_unsortable():
    hp1 = Hparam([1, 2, 3])
    hp2 = Hparam([1, 2, 3])
    hp2._importance_estimators = None
    hp1 &= hp2
    assert not hp1.is_sortable
    assert hp1.importance is None


def test_iand_rejects_non_hparam():
    hp = Hparam([1, 2, 3])
    with pytest.raises(AssertionError, match="Cannot merge"):
        hp &= 5


def test_iand_fails_if_original_dropped():
    hp1 = Hparam([1, 2, 3])  # original == 3
    hp2 = Hparam([1, 2, 4])
    with pytest.raises(AssertionError, match="Original choice not in choices"):
        hp1 &= hp2


# -----------------------------------------------------------------------------------------------
# Pickling (required for distributed broadcasting, see NOTE in hparam.py)
# -----------------------------------------------------------------------------------------------


def test_pickle_roundtrip():
    hp = Hparam([1, 2, 3], original=2)
    hp.active = 1
    hp.enforce_order(torch.tensor([2, 0, 1]))
    hp2 = pickle.loads(pickle.dumps(hp))
    assert hp2.choices == [1, 2, 3]
    assert hp2.original == 2
    assert hp2.active == 1
    assert hp2.is_configurable
    assert torch.equal(hp2._slice_order, torch.tensor([2, 0, 1]))
    assert torch.equal(hp2.active_slice, torch.tensor([2]))
