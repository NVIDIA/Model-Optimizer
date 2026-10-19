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

import pytest
import torch.nn as nn

from modelopt.torch.opt.dynamic import (
    Hparam,
    _DMAttributeManager,
    _FoldedCallback,
    _pytorch_managed,
)


def test_dm_attribute_manager():
    """A simple unit test to test the _DMAttributeManager class."""
    manager = _DMAttributeManager()
    assert not manager, "manager should be empty!"
    assert manager.level == 0, "level should be 0!"

    # add a new layer
    manager.append_level(nn.Linear)
    assert manager.og_cls is nn.Linear, "og_cls should be nn.Linear!"
    assert manager.level == 1, "level should be 1!"

    # add an hparam
    hp = Hparam([1, 2, 3])
    manager.set_hp("in_features", hp)
    assert manager.get_hp("in_features") is hp, "hp should be the same!"
    assert len(list(manager.named_hps())) == 1, "should have 1 hp!"
    assert manager.hp_keys() == {"in_features"}, "should have in_features!"

    # set a dynamic attribute
    def cb1(m, w):
        return w * 2.0

    manager.set_da("weight", _pytorch_managed, cb1)
    assert manager.da_keys() == {"weight"}, "should have weight!"

    # set an attribute
    manager.set_attr("amax", None, None)
    assert manager.attr_keys() == {"amax"}, "should have amax!"

    # add another level
    manager.append_level(nn.ReLU)

    # try adding an hp
    with pytest.raises(AssertionError, match="Hparam already exists in the base cls!"):
        manager.set_hp("in_features", hp)  # already exist in another level
    hp2 = Hparam([4, 5, 6])
    manager.set_hp("out_features", hp2)
    assert manager.get_hp("in_features") is hp, "hp should be the same!"
    assert manager.hp_keys() == {"in_features", "out_features"}, "should have both hps!"
    assert manager.hp_keys(all=False) == {"out_features"}, "should have only out_features!"

    # try adding dynamic attribute
    def cb2(m, w):
        return w * 3.0

    manager.set_da("weight", _pytorch_managed, cb2)
    manager.set_da("weight", cb=cb2)  # should both work...
    assert manager.da_keys() == {"weight"}, "should have weight only!"

    # try adding an attribute
    with pytest.raises(AssertionError, match="Attribute already exists!"):
        manager.set_attr("amax", None, None)  # already exists
    manager.set_attr("bmax", None, None)

    # remove current level
    with pytest.raises(AssertionError, match="Some hparams were not removed properly!"):
        manager.pop_level()  # have not cleaned out hps and callbacks
    assert bool(manager)  # this will do a sanity check on the data structure
    manager.pop_hp("out_features")
    with pytest.raises(
        AssertionError, match="Some dynamic attribute callbacks were not removed properly!"
    ):
        manager.pop_level()  # have not cleaned out callbacks
    assert bool(manager)  # this will do a sanity check on the data structure
    manager.fold_cbs()  # fold callbacks to clean them up
    manager.pop_level()
    assert bool(manager)  # this will do a sanity check on the data structure

    # some more sanity checks after removal
    assert manager.level == 1, "level should be 1!"
    assert manager.da_keys() == {"weight"}, "should have weight only!"
    assert manager.hp_keys() == {"in_features"}, "should have in_features!"
    assert manager.attr_keys() == {"amax", "bmax"}, "should still have amax and bmax!"

    # check folded callbacks now
    cb_folded = manager.get_da_cb("weight")
    assert isinstance(cb_folded, _FoldedCallback), "should be a folded callback!"
    assert len(cb_folded) == 2, "should have 2 callback!"
    assert cb_folded.callback is cb1, "should be cb1!"
    assert cb_folded._callbacks == [cb1, cb2], "should be cb2!"

    # do another pop
    manager.pop_hp("in_features")
    manager.fold_cbs()
    with pytest.raises(
        AssertionError, match="Some dynamic attribute callbacks were not removed properly!"
    ):
        manager.pop_level()  # folding doesn't help for last level
    assert bool(manager)  # this will do a sanity check on the data structure
    manager.pop_da("weight")
    with pytest.raises(AssertionError, match="Some attributes were not removed properly!"):
        manager.pop_level()  # have not cleaned up attributes for last level
    assert bool(manager)  # this will do a sanity check on the data structure
    manager.pop_attr("amax")
    manager.pop_attr("bmax")
    manager.pop_level()
    assert not manager, "manager should be empty!"
