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

import contextlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import modelopt.torch.nas as mtn
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.prune.gradnas import GradientBinarySearcher

try:
    from _test_utils.torch.deploy.runtime import FAKE_DEPLOYMENT, fake_latency

    import modelopt.torch._deploy  # noqa: F401

    CONSTRAINTS = {"flops": "90%", "latency": "90%"}
except ImportError:
    print("Skipping latency constraint for gradnas tests")
    FAKE_DEPLOYMENT = None
    fake_latency = lambda x: contextlib.nullcontext()  # noqa: E731
    CONSTRAINTS = {"flops": "90%"}


@pytest.mark.parametrize(
    ("model", "dummy_input", "is_error_expected"),
    [
        (
            nn.Sequential(nn.Linear(1, 8), nn.Linear(8, 8), nn.Linear(8, 1)),
            torch.randn(1, 1),
            False,
        ),
        (
            nn.Sequential(nn.Conv2d(1, 8, 1), nn.Conv2d(8, 8, 1), nn.Conv2d(8, 1, 1)),
            torch.randn(1, 1, 8, 8),
            True,
        ),
        (
            nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.Conv2d(8, 1, 1),
                nn.Flatten(),
                nn.Linear(16, 8),
                nn.Linear(8, 1),
            ),
            torch.randn(1, 1, 4, 4),
            False,
        ),
    ],
)
def test_gradnas(model, dummy_input, is_error_expected, use_channel_div_4):
    modelopt_model = mtn.convert(model, "gradnas")

    data_loader = [(dummy_input,)]

    def loss_func(x, batch):
        label = x + 1
        return F.mse_loss(x, label)

    # make sure the forward patching has been removed
    for _, module in model.named_children():
        assert "forward" not in vars(module)

    # make sure the model can be exported
    with fake_latency(100):
        mtn.profile(modelopt_model, dummy_input, use_centroid=True, deployment=FAKE_DEPLOYMENT)

    with (
        pytest.raises(AssertionError, match="GradientBinarySearcher: no searchable hparams found")
        if is_error_expected
        else contextlib.nullcontext()
    ):
        with fake_latency([100, 75]):
            searched_model, search_history = mtn.search(
                modelopt_model,
                CONSTRAINTS,
                dummy_input=dummy_input,
                config={
                    "num_iters": 5,
                    "data_loader": data_loader,
                    "loss_func": loss_func,
                    "deployment": FAKE_DEPLOYMENT,
                },
            )

        # Test if all hparams have score tensors and whether they are sorted
        for hp_name, hparam in named_hparams(modelopt_model, configurable=True):
            suffix = hp_name.rpartition(".")[-1]

            if (
                suffix not in GradientBinarySearcher().hparam_names_for_search
                or len(hparam.choices) == 1
            ):
                assert not hasattr(hparam, "score_tensor")
                assert search_history["best"]["config"][hp_name] == hparam.max
                continue

            assert hparam.score_tensor is not None
            assert torch.all(
                torch.sort(hparam.score_tensor, descending=True)[0] == hparam.score_tensor
            )
