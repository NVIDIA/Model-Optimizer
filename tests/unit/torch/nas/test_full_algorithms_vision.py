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
import math
from contextlib import nullcontext

import pytest
import torch.nn as nn
from _test_utils.torch.vision_models import get_tiny_mobilenet_and_input, get_tiny_resnet_and_input

pytest.importorskip("torchprofile")

import modelopt.torch.nas as mtn
from modelopt.torch.nas.algorithms import ConstraintsFunc
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import random


@pytest.mark.parametrize(
    ("get_model_and_input", "variant"),
    [
        (get_tiny_mobilenet_and_input, 0),
        (get_tiny_resnet_and_input, 1),
        (get_tiny_resnet_and_input, 2),
    ],
)
def test_searched_model_constraints(get_model_and_input, variant):
    """Tests to see if subnets found satisfy constraints specified"""

    def _initialize_test_case(get_model_and_input, variant):
        model, args, kwargs = get_model_and_input()
        dummy_input = (*args, kwargs)
        mode = ["autonas", "fastnas", "autonas"]
        model = mtn.convert(model, mode=mode[variant])
        _, stats = mtn.profile(model, dummy_input, use_centroid=True)

        constraints = [
            {"flops": stats["max"]["flops"], "params": stats["max"]["params"]},
            {"flops": stats["max"]["flops"]},
            {"flops": 0, "params": 0},
        ]
        satisfiable = [True, True, False]

        return (
            model,
            dummy_input,
            constraints[variant].keys(),
            constraints[variant],
            satisfiable[variant],
        )

    model, dummy_input, interp_keys, constraints, satisfiable = _initialize_test_case(
        get_model_and_input, variant
    )
    flops_max = mtn.utils.inference_flops(model, dummy_input)

    def check_constraint(val: float, limits: float | tuple[float, float]) -> bool:
        """Check if val falls within limits. `None` indicates open-ended intervals."""
        return val >= limits[0] and val <= limits[1] if isinstance(limits, tuple) else val <= limits

    def fake_score(subnet: nn.Module) -> float:
        return mtn.utils.inference_flops(subnet, dummy_input) / flops_max

    def run_search():
        return mtn.search(
            model=model,
            constraints=constraints,
            dummy_input=dummy_input,
            config={
                "score_func": fake_score,  # fake score proportional to flops
                "num_iters": 10,
            },
        )

    # subnet found should not satisfy any un-satisfiable constraints
    if not satisfiable:
        with pytest.raises(ValueError):
            run_search()

    else:
        _, detailed_stats = run_search()

        for key in interp_keys:
            subnets_constraint_history = detailed_stats["history"]["constraints"][key]
            limits = constraints[key]
            for subnet_stat in subnets_constraint_history:
                # subnet found should satisfy all satisfiable constraints
                assert check_constraint(subnet_stat, limits)


@pytest.mark.parametrize("flops_only", [True, False])
@pytest.mark.parametrize("bounded_latency", [True, False])
def test_search_constraints(flops_only: bool, bounded_latency: bool):
    def _fake_latency(self, model, precomputed=None):
        coeff_params = 0.00 if flops_only else 0.75
        if not hasattr(_fake_latency, "call_counter"):
            _fake_latency.call_counter = 0
        _fake_latency.call_counter += 1  # counter to keep track of how often it is called
        print("Calling fake latency")
        return (1.8 * self._get_flops(model) + coeff_params * self._get_params(model)) / 1.0e6

    model, args, kwargs = get_tiny_resnet_and_input()
    dummy_input = (*args, kwargs)
    model = mtn.convert(model, mode="autonas")

    # get fake latency and spoof ConstraintsFunc class
    ConstraintsFunc._get_true_latency = _fake_latency

    # now setup constraints functor
    constraints_func = ConstraintsFunc(model, {}, dummy_input, {"device": None})

    # we want to some (un-)reasonable limits here for constraints_func
    if bounded_latency:
        mtn.sample(model, random.centroid)
    else:
        mtn.sample(model, max)
    latency = _fake_latency(constraints_func, model)
    constraints_func.limits = {"latency": (1.01 if bounded_latency else 0.1) * latency}

    # run profile to fill with default options and check that it should pass if we set
    # reasonable latency
    with nullcontext() if bounded_latency else pytest.raises(ValueError):
        mtn.profile(model, constraints=constraints_func, strict=True, use_centroid=True)

    # at this point we should have _fake_latency counter at 4 (1 explicit call + 9 in profile)
    expected_counter = 10
    assert _fake_latency.call_counter == expected_counter

    # cannot continue test with wrong latency
    if not bounded_latency:
        return

    # now try random sampling, this doesn't have to always call the latency function
    num_random = 3
    configs = []
    for _ in range(num_random):
        configs.append(mtn.sample(model, random.choice))
        _, vals = constraints_func()
        latency = _fake_latency(constraints_func, model)

        # here it is clear how "accurate" the linear estimate is
        if not flops_only:
            continue

        # for FLOPS only we can check if the linear interpolation matches (up to 0.5%)
        assert vals["latency"] / latency == pytest.approx(1.0, rel=0.005)

    # now check counter
    current_counter = _fake_latency.call_counter
    assert current_counter <= expected_counter + 2 * num_random
    assert current_counter >= expected_counter + num_random

    print(constraints_func.fast_eval)

    # now use same configs again --> shouldn't call latency again!
    for config in configs:
        mtn.select(model, config)
        constraints_func()

    # check counter one final time
    # TODO: differs by 1 sometimes non-deterministically. Need to debug
    assert current_counter <= _fake_latency.call_counter <= current_counter + 1


@pytest.mark.parametrize(
    "get_model_and_input", [get_tiny_resnet_and_input, get_tiny_mobilenet_and_input]
)
def test_profile_same_max(get_model_and_input):
    """checks if the max/original subnet of the profiled model is the same as the original model."""

    def _initialize_test_case_profile(get_model_and_input):
        model, args, kwargs = get_model_and_input()
        dummy_input = (*args, kwargs)
        return (
            model,
            dummy_input,
            ["flops", "params"],
            {"full"},
            {"min", "centroid", "max", "max/min ratio"},
        )

    model_og, dummy_input, interp_keys, keys_og, keys_converted = _initialize_test_case_profile(
        get_model_and_input
    )

    # generate converted model from a deepcopy
    model = mtn.convert(copy.deepcopy(model_og), mode="fastnas")

    is_all_sat_original, stats_original = mtn.profile(model_og, dummy_input, use_centroid=True)
    is_all_sat_converted, stats_converted = mtn.profile(model, dummy_input, use_centroid=True)

    # original model stats should only contain info about the full subnet
    for key in keys_og:
        assert key in stats_original

    should_not_contain = keys_converted.difference(keys_og)
    for key in should_not_contain:
        assert key not in keys_og

    # converted model stats should contain info about min, centroid, and max
    for key in keys_converted:
        assert key in keys_converted

    # since we didn't specify constraints, all constraints should be satisfiable
    assert is_all_sat_original
    assert is_all_sat_converted

    # max subnet stats should be the same between the original model and the converted model
    for key in interp_keys:
        assert stats_original["full"][key] == stats_converted["max"][key]


@pytest.mark.parametrize(
    ("get_model_and_input", "mode"),
    [
        (get_tiny_resnet_and_input, "autonas"),
        (get_tiny_mobilenet_and_input, "fastnas"),
    ],
)
def test_profile_search_space(get_model_and_input, mode):
    """checks if the search space found by nas.profile is consistent."""
    model, args, kwargs = get_model_and_input()
    dummy_input = (*args, kwargs)
    model = mtn.convert(model, mode=mode)
    search_space_size = math.prod([len(hp.choices) for _, hp in named_hparams(model, True)])

    _, stats = mtn.profile(model, dummy_input, use_centroid=True)
    assert stats["search space size"] == search_space_size
