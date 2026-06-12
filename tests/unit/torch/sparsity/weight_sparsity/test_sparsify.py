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
from _test_utils.torch.misc import compare_outputs
from _test_utils.torch.nas_prune.models import BaseExampleModel, get_example_models

import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts
from modelopt.torch.opt.utils import is_dynamic
from modelopt.torch.sparsity.weight_sparsity.magnitude import MagnitudeSearcher
from modelopt.torch.utils.random import _set_deterministic_seed

try:
    import transformers
except ImportError:
    transformers = None

benchmarks = get_example_models()
algos = [("sparse_magnitude", {}), ("sparsegpt", {"device": "cpu"})]

skip_models = ["ExampleModel18", "LinearModel11"]
benchmarks = {k: v for k, v in benchmarks.items() if k not in skip_models}


@pytest.mark.parametrize("specs", algos, ids=[x[0] for x in algos])
@pytest.mark.parametrize("model", benchmarks.values(), ids=benchmarks.keys())
def test_save_restore_whole(model: BaseExampleModel, specs):
    _set_deterministic_seed()

    model = copy.deepcopy(model)
    model.eval()
    model_restored = copy.deepcopy(model)

    args = model.get_args()

    # sparsify model in-place
    model = mts.sparsify(
        model,
        mode=specs[0],
        config={"data_loader": [args], "collect_func": lambda x: x, **specs[1]},
    )

    assert is_dynamic(model), "Model should be dynamic after sparsification."

    modelopt_state = mto.modelopt_state(model)
    model_state = model.state_dict()

    out = model(*args)

    model_restored = mto.restore_from_modelopt_state(model_restored, modelopt_state)
    model_restored.load_state_dict(model_state)

    assert is_dynamic(model_restored), "Model should be dynamic after restore."

    out_restored = model_restored(*args)

    compare_outputs(out, out_restored)

    # export is in-place, so we can just do it like this...
    mts.export(model)

    # check that it's not dynamic anymore when we use export
    assert not is_dynamic(model), "Model should be static after export."

    out_export = model(*args)

    compare_outputs(out, out_export)


@pytest.mark.skipif(transformers is None, reason="transformers is not installed.")
def test_specify_forward_loop():
    # setup
    model = copy.deepcopy(benchmarks["GPTJModel"])
    model.eval()
    args = model.get_args()
    searcher = MagnitudeSearcher()

    # test different configs for forward loop
    config_data_loader = {"data_loader": [args], "collect_func": lambda x: x}
    config_loop = {"forward_loop": lambda m: m(*args)}

    # try specifying data_loader
    searcher.search(model, {}, (), config_data_loader)
    assert searcher.forward_loop is not None
    searcher.forward_loop(model)

    # try specifying forward_loop
    searcher.search(model, {}, (), config_loop)
    assert searcher.forward_loop is not None
    searcher.forward_loop(model)

    # try specifying None
    searcher.search(model, {}, (), {})
    assert searcher.forward_loop is None

    # try specifying both
    with pytest.raises(AssertionError, match="Only provide `data_loader` or `forward_loop`!"):
        searcher.search(model, {}, (), {**config_data_loader, **config_loop})
