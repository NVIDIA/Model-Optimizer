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

from modelopt.torch.trace import RobustTracer, Symbol, SymMap
from modelopt.torch.trace.modules.nn import get_conv_sym_info, get_linear_sym_info


def test_symbol_cls():
    sym = Symbol(elastic_dims={1, 2}, cl_type=Symbol.CLType.INCOMING)

    assert sym.is_free, "Symbol should be free"
    assert sym.elastic_dims == {1, 2}, "Symbol should have elastic dims {1, 2}"

    sym._is_searchable = True
    assert sym.is_searchable, "Symbol should be searchable"

    # check if we can set a parent
    sym2 = Symbol(elastic_dims={1, 2}, cl_type=Symbol.CLType.INCOMING)
    sym.link_to(sym2)
    assert sym.parent == sym2, "Symbol should have a parent"
    assert sym.is_dynamic, "Symbol should be dynamic"

    # checking that linking doesn't work anymore
    with pytest.raises(AssertionError):
        sym.link_to(Symbol())
    with pytest.raises(AssertionError):
        sym2.link_to(sym)

    # check if we can disable everything in one go
    sym.disable()
    assert sym.is_constant, "Symbol should be constant"
    assert sym2.is_constant, "Symbol 2 should be constant"

    # checking that linking doesn't work anymore
    with pytest.raises(AssertionError):
        sym.link_to(Symbol())


class ExampleModel(nn.Module):
    conv_class = nn.Conv2d

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.ModuleList(
            [
                self.conv_class(3, 3, 3, padding="same"),
                self.conv_class(3, 3, 3, padding="same"),
                self.conv_class(3, 3, 3, padding="same"),
            ]
        )

    def forward(self, x):
        for mod in self.main:
            x = mod(x)
        return x


class ExampleModel2(ExampleModel):
    class CustomConv2d(nn.Conv2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._another_private_attr = None

    conv_class = CustomConv2d


@pytest.mark.parametrize("model", [ExampleModel(), ExampleModel2()])
def test_sym_map(model):
    num_mods_with_syms = len(model.main)
    syms_per_mod = 3  # in_c + out_c + ks
    num_syms = num_mods_with_syms * syms_per_mod
    sym_map = SymMap(model)

    def assert_num_symbols():
        assert len(sym_map) == num_mods_with_syms, f"{num_mods_with_syms} mods in sym_map expected!"
        assert len(list(sym_map.named_symbols())) == num_syms, f"{num_syms} symbols expected!"

    # check # of symbols
    assert_num_symbols()

    # check popping off a module
    sym_map.pop(model.main[0])
    num_mods_with_syms -= 1
    num_syms -= syms_per_mod
    assert_num_symbols()

    # try pruning the sym_map after setting all symbols of a module to constant
    syms = sym_map[model.main[1]]
    for sym in syms.values():
        sym.disable()
    sym_map.prune()
    num_mods_with_syms -= 1
    num_syms -= syms_per_mod
    assert_num_symbols()


def test_sym_map_registry():
    mods_in_registry = {
        nn.Linear,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.Sequential,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    }

    try:
        from transformers.models.bert.modeling_bert import BertAttention
        from transformers.models.gptj.modeling_gptj import GPTJAttention

        mods_in_registry.add(BertAttention)
        mods_in_registry.add(GPTJAttention)
    except ImportError:
        pass

    not_a_leaf = {nn.Sequential}
    dependent_registry = set()

    def check_registry():
        # check sym map registry
        assert SymMap._registry.keys() == mods_in_registry, "Mismatch in registry!"
        assert dependent_registry <= SymMap._dependent_registry.keys(), "Mismatch in dep registry!"

        # check tracer registry
        all_leaves = (mods_in_registry | dependent_registry) - not_a_leaf
        assert all_leaves <= RobustTracer._leaves, "Mismatch in trace registry!"

    # check if registry matches our expectations
    check_registry()

    # check if we can properly unregister and re-register module...
    mods_in_registry.remove(nn.Linear)
    SymMap.unregister(nn.Linear)
    check_registry()

    SymMap.register(nn.Linear)(get_linear_sym_info)
    mods_in_registry.add(nn.Linear)
    check_registry()

    # now check if dependent registry works as expected
    model = ExampleModel2()
    SymMap(model)
    dependent_registry.add(model.CustomConv2d)
    check_registry()

    # note that we cannot remove dependencies from the registry directly
    with pytest.raises(KeyError):
        SymMap.unregister(model.CustomConv2d)

    # now try removing the conv layer --> will also remove dependent layer!
    mods_in_registry.remove(nn.Conv2d)
    dependent_registry.remove(model.CustomConv2d)
    SymMap.unregister(nn.Conv2d)
    check_registry()

    # put linear layer back in --> dependent layer will still be gone!
    SymMap.register(nn.Conv2d)(get_conv_sym_info)
    mods_in_registry.add(nn.Conv2d)
    check_registry()
