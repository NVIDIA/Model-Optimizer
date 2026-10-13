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
import torch
from _test_utils.torch.misc import compare_outputs
from _test_utils.torch.nas_prune.models import BaseExampleModel, get_example_models

from modelopt.torch.nas.search_space import generate_search_space
from modelopt.torch.utils.random import _set_deterministic_seed

benchmarks = get_example_models()
benchmarks2 = get_example_models()


@pytest.mark.parametrize("model", benchmarks.values(), ids=benchmarks.keys())
def test_benchmarks_dynamic(model: BaseExampleModel):
    search_space = generate_search_space(model)
    # check print representation to make sure modules are printable after search space conversion
    str(model)
    # set default seed
    _set_deterministic_seed()

    config = search_space.sample(sample_func=model.get_sample_func())
    assert isinstance(config, dict)

    # Check the number of configurable hparams
    configurable_hps = dict(search_space.named_hparams(configurable=True))
    assert len(configurable_hps) == model.get_num_configurable_hparams(), configurable_hps

    # Retrieve args
    args = model.get_args()

    # Check forward when we have a search space
    out1 = model(*args)

    # Check export
    subnet = search_space.export()

    # Check subnet forward
    out2 = subnet(*args)
    compare_outputs(out1, out2)


@pytest.mark.parametrize("model", benchmarks2.values(), ids=benchmarks2.keys())
def test_dynamic_sorting(model: BaseExampleModel):
    model.eval()
    search_space = generate_search_space(model)

    # Retrieve args
    args = model.get_args()

    # Get the output of the original model
    with torch.no_grad():
        y1 = model(*args)

    search_space.sort_parameters()

    # sanity check if the model functionality is preserved after sorting
    with torch.no_grad():
        y2 = model(*args)

    # # check if the inference results after sorting is the same
    compare_outputs(y1, y2, atol=1e-5, rtol=1e-5)

    # recompute importance and check if order is trivial
    num_sortable = 0
    for hp_name, hp in search_space.named_hparams(configurable=True):
        num_sortable += hp.importance is not None
        model.assert_order(hp_name, hp)

    # check if all hparam groups have been sorted
    assert num_sortable == model.get_num_sortable_hparams()
