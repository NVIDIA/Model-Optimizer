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
import torch.nn as nn

pytest.importorskip("torchprofile")

import modelopt.torch.nas as mtn
from modelopt.torch.nas.algorithms import ConstraintsFunc


class Conv2dResidual(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        return x + self.conv(x)


def _get_model_and_inputs(residual=False):
    conv_cls = Conv2dResidual if residual else nn.Conv2d
    model = nn.Sequential(nn.Conv2d(1, 8, 1), conv_cls(8, 8, 1), conv_cls(8, 8, 1))
    dummy_input = torch.randn(1, 1, 4, 4)
    return model, dummy_input


sample_points_dict = {
    None: {"full"},
    ("flops",): {"min", "centroid", "max"},
    ("flops", "flops_min_depth"): ConstraintsFunc._sample_points_dict[
        ("flops", "flops_min_depth")
    ].keys(),
    ("flops_min_depth", "flops"): ConstraintsFunc._sample_points_dict[
        ("flops", "flops_min_depth")
    ].keys(),
    ("flops", "params"): {"min", "centroid", "max"},
}


@pytest.fixture
def fake_latency():
    def _fake_latency(self, model, precomputed=None):
        if not hasattr(_fake_latency, "call_counter"):
            _fake_latency.call_counter = 0
        _fake_latency.call_counter += 1  # counter to keep track of how often it is called
        print("Calling fake latency")
        return (self._get_flops(model)) / 1.0e6

    _get_true_latency = ConstraintsFunc._get_true_latency
    ConstraintsFunc._get_true_latency = _fake_latency
    yield _fake_latency
    ConstraintsFunc._get_true_latency = _get_true_latency


@pytest.mark.parametrize(
    (
        "benchmark",
        "mode",
        "keys_for_interpol",
        "expected_latency_calls",
        "expected_max_search_latency_calls",
    ),
    [
        (_get_model_and_inputs, "fastnas", ("flops",), 3, 4),
        (lambda: _get_model_and_inputs(True), "autonas", ("flops", "flops_min_depth"), 9, 10),
        (_get_model_and_inputs, "autonas", ("flops",), 3, 10),
    ],
)
def test_profile_interpolator(
    benchmark,
    mode,
    keys_for_interpol,
    expected_latency_calls,
    expected_max_search_latency_calls,
    fake_latency,
    use_channel_div_4,
):
    model, dummy_input = benchmark()
    model.eval()

    model = mtn.convert(model, mode=mode)

    ignore_keys = {"max/min ratio", "number of configurable hparams", "search space size"}
    # keys_for_default_interpol = ("flops",) if mode else None

    fake_latency.call_counter = 0
    # NOTE: revert when we bring back deployment
    # _, stats = mtn.profile(model, dummy_input, deployment={"runtime": None}, use_centroid=True)
    # assert fake_latency.call_counter == (3 if mode else 1)
    _, stats = mtn.profile(model, dummy_input, use_centroid=True)
    assert (stats.keys() - ignore_keys) == sample_points_dict[keys_for_interpol]

    fake_latency.call_counter = 0
    constraintsfunc = ConstraintsFunc(
        model,
        constraints={"latency": float("inf")},
        dummy_input=dummy_input,
        deployment={"runtime": None},
    )
    _, stats = mtn.profile(model, constraints=constraintsfunc, use_centroid=True)
    assert fake_latency.call_counter == expected_latency_calls
    assert (stats.keys() - ignore_keys) == sample_points_dict[keys_for_interpol]

    fake_latency.call_counter = 0
    _, _ = mtn.search(
        model,
        constraints={
            # NOTE: revert when we bring back deployment
            # "latency": stats["max"]["latency"] if "max" in stats else stats["full"]["latency"]
            "flops": stats["max"]["flops"] if "max" in stats else stats["full"]["flops"]
        },
        dummy_input=dummy_input,
        # deployment={"runtime": None},  # NOTE: revert when we bring back deployment
        config={"score_func": lambda _: 1, "num_iters": 1},
    )
    assert fake_latency.call_counter <= expected_max_search_latency_calls
