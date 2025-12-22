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

import itertools

import pytest
from _test_utils.torch.nas_prune.models import BaseExampleModel, get_example_models

from modelopt.torch.trace import Symbol, SymDepth, SymMap, analyze_symbols

benchmarks = get_example_models()


def _get_dependencies(sym: Symbol, sym_map: SymMap, sym_to_name: dict[Symbol, str] | None = None):
    """Get all the dependencies of a symbol."""
    if sym_to_name is None:
        sym_to_name = {sym: name for name, sym in sym_map.named_symbols()}

    # grab all dependencies in a DFS fashion (including input_syms from concat symbol)
    deps = {}
    for dep in itertools.chain(sym._dependencies, getattr(sym, "input_syms", [])):
        if dep.is_constant:
            # potentially true for input syms...
            continue
        if dep in sym_to_name:
            # not true for input syms...
            deps[sym_to_name[dep]] = dep
        deps.update(_get_dependencies(dep, sym_map, sym_to_name))

    return deps


def _test_util(model: BaseExampleModel):
    # test tracing
    sym_map = analyze_symbols(model)

    # check the number of searchable symbols
    searchable_symbols = dict(sym_map.named_symbols(searchable=True))
    assert len(searchable_symbols) == model.get_num_searchable_symbols(), searchable_symbols

    # check that there are no free symbols left
    free_symbols = dict(sym_map.named_symbols(free=True))
    assert len(free_symbols) == 0, free_symbols

    # check expected dependencies between symbols (if any...)
    expected_deps = model.get_expected_dependencies()
    for name, sym in searchable_symbols.items():
        # get dependencies
        deps = _get_dependencies(sym, sym_map)
        assert deps.keys() == expected_deps.pop(name, set()), f"{name}: {deps.keys()}"
    assert not expected_deps, expected_deps

    # check depth results...
    expected_skippables = model.get_skippable_for_depths()
    for name, sym in searchable_symbols.items():
        if not isinstance(sym, SymDepth):
            continue
        max_depth = len(model.get_submodule(name.rpartition(".")[0]))
        skippable_expected = expected_skippables.pop(name, [])
        min_depth_expected = max_depth - len(skippable_expected)
        assert sym.min_depth == min_depth_expected, f"{name}: {sym.min_depth}"
        assert skippable_expected == sym.skippable_idxs, f"{name}: {sym.skippable_idxs}"
    assert not expected_skippables, expected_skippables

    # check sortability
    expected_unsortable = model.get_unsortable_searchable_symbols()
    unsortable = {name for name, sym in searchable_symbols.items() if not sym.is_sortable}
    assert expected_unsortable == unsortable


# TODO: consider bringing fixture for experimental mode back eventually
@pytest.mark.parametrize("model", benchmarks.values(), ids=benchmarks.keys())
def test_benchmarks_trace(model):
    _test_util(model)
