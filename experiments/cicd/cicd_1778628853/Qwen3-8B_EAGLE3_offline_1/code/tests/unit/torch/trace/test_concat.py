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

import torch
import torch.nn as nn

from modelopt.torch.trace import ConcatSymbol, Symbol, analyze_symbols


class FakeSymbol(Symbol):
    pass


def test_concat_symbol():
    oc1 = ConcatSymbol.Input.convert(
        Symbol(is_searchable=True, cl_type=Symbol.CLType.OUTGOING, elastic_dims={-1}), -1
    )
    oc2 = ConcatSymbol.Input.convert(
        FakeSymbol(is_searchable=True, cl_type=Symbol.CLType.OUTGOING, elastic_dims={-1}), -1
    )

    concat_sym = ConcatSymbol([oc1, oc2])
    assert concat_sym.input_syms == [oc1, oc2]

    # concat should be searchable as long as one of its input syms is searchable
    assert concat_sym.is_searchable
    oc1.disable()
    assert concat_sym.is_searchable
    oc2.disable()
    assert not concat_sym.is_searchable and concat_sym.is_constant

    # Disabling concat hparam should disable all individual hparams
    concat_sym.disable()
    for sym in [oc1, oc2, concat_sym]:
        assert sym.is_constant
        assert not sym.is_searchable

    # check conversion back from Input symbol
    oc1_converted = oc1.create_linked_copy()
    assert type(oc1_converted) is Symbol
    assert oc1_converted.is_dynamic

    oc2_converted = oc2.create_linked_copy()
    assert type(oc2_converted) is FakeSymbol
    assert oc2_converted.is_dynamic


class ExampleModel(nn.Module):
    """A model with a concat input linked to another concat input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.conv2 = nn.Conv2d(3, 32, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o3 = self.conv3(
            torch.cat([o1, o2], dim=1)
        )  # original conv1 and conv2 linked to their ConcatInpSymbol copies
        o4 = self.conv4(o1 + o2)  # both ConcatInpSymbols disabled
        return o3 + o4


def test_concat_input_linking():
    model = ExampleModel()
    sym_map = analyze_symbols(model)
    assert len(sym_map) == 4, f"No cross-layer symbols should be created for this model: {sym_map}"
