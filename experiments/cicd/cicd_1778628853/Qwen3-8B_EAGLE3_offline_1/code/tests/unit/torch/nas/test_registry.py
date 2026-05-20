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

import copy

import pytest
import torch.nn as nn

from modelopt.torch.nas.modules import _DynamicLinear
from modelopt.torch.nas.registry import DMRegistry

# Create a new registry as not to interfere with the global one.
DMRegistry2 = copy.deepcopy(DMRegistry)


class Linear(nn.Linear):
    pass


class Linear2(nn.Linear):
    pass


class RandomModule(nn.Module):
    pass


def test_register_unregister():
    # check if linear is registered (can be done via type or instance)
    assert nn.Linear in DMRegistry2
    assert nn.Linear(1, 1) in DMRegistry2

    def _test_new_cls(cls, name):
        # register cls with tests
        assert cls in DMRegistry2
        dyn_cls = DMRegistry2[cls]  # this should not throw an error
        assert cls in DMRegistry2._dynamic_classes
        assert dyn_cls.__name__ == name

    # test nn.Linear
    _test_new_cls(nn.Linear, "DynamicLinear")

    # test linear (has name clash!)
    _test_new_cls(Linear, "test_registry_DynamicLinear")

    # test linear2
    _test_new_cls(Linear2, "DynamicLinear2")

    # check key for nn.Linear, linear, linear2
    assert DMRegistry2.get_key(nn.Linear) == "nn.Linear"
    assert DMRegistry2.get_key(Linear) == "nn.Linear"
    assert DMRegistry2.get_key(Linear2) == "nn.Linear"

    # check key from dynamic modules
    assert DMRegistry2.get_key_from_dm(DMRegistry2[nn.Linear]) == "nn.Linear"
    assert DMRegistry2.get_key_from_dm(DMRegistry2[Linear]) == "nn.Linear"
    assert DMRegistry2.get_key_from_dm(DMRegistry2[Linear2]) == "nn.Linear"

    # unregister nn.Linear (only works with truly registered class)
    with pytest.raises(KeyError):
        DMRegistry2.unregister(Linear)
    with pytest.raises(KeyError):
        DMRegistry2.unregister(Linear2)
    DMRegistry2.unregister(nn.Linear)

    # all linear classes should be unregistered
    assert nn.Linear not in DMRegistry2
    assert Linear not in DMRegistry2
    assert Linear2 not in DMRegistry2
    assert nn.Linear not in DMRegistry2._registry
    assert nn.Linear not in DMRegistry2._dynamic_classes
    assert Linear not in DMRegistry2._dynamic_classes
    assert Linear2 not in DMRegistry2._dynamic_classes

    # test Linear first (now nn.Linear has the name clash...)
    DMRegistry2.register({nn.Linear: "nn.Linear"})(_DynamicLinear)

    # test Linear
    _test_new_cls(Linear, "DynamicLinear")

    # test nn.Linear (has name clash!)
    _test_new_cls(nn.Linear, "torch_nn_modules_linear_DynamicLinear")

    # test linear2
    _test_new_cls(Linear2, "DynamicLinear2")

    # check that RandomModule is not registered
    assert RandomModule not in DMRegistry2
    assert DMRegistry2.get(RandomModule) is None
    with pytest.raises(KeyError):
        DMRegistry2[RandomModule]
    with pytest.raises(KeyError):
        DMRegistry2.unregister(RandomModule)


def test_convert():
    def _test(mod):
        og_type = type(mod)
        mod = DMRegistry2.convert(mod)
        dyn_type = type(mod)
        assert dyn_type == DMRegistry2[og_type], f"{dyn_type} != {DMRegistry2[og_type]}"

    # test nn.Linear
    _test(nn.Linear(1, 1))

    # test Linear
    _test(Linear(1, 1))

    # test Linear2
    _test(Linear2(1, 1))

    # test RandomModule
    with pytest.raises(KeyError):
        _test(RandomModule())
