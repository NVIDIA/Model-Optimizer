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
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.prune.gradnas import GradientBinarySearcher, _setup_grad_manager_linear

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


class _CountingDataLoader:
    def __init__(self, batch, max_batches):
        self._batch = batch
        self._max_batches = max_batches
        self.num_batches = 0

    def __iter__(self):
        for _ in range(self._max_batches):
            self.num_batches += 1
            yield self._batch


def _make_gradnas_model():
    model = nn.Sequential(nn.Linear(4, 16, bias=False), nn.Linear(16, 8, bias=False))
    with torch.no_grad():
        for layer in model:
            layer.weight.fill_(0.1)
    return mtn.convert(model, "gradnas")


def _estimate_gradnas_scores(
    modelopt_model,
    dummy_input,
    *,
    average_scores,
    convergence_tol,
    convergence_patience,
    convergence_min_updates,
    max_batches,
    max_iter_data_loader=None,
):
    def loss_func(output, _batch):
        return output.pow(2).mean()

    data_loader = _CountingDataLoader((dummy_input,), max_batches=max_batches)
    searcher = GradientBinarySearcher()
    searcher.model = modelopt_model
    searcher.config = {
        **searcher.default_search_config,
        "average_scores": average_scores,
        "score_convergence_tol": convergence_tol,
        "score_convergence_patience": convergence_patience,
        "score_convergence_min_updates": convergence_min_updates,
    }
    had_setup = hasattr(GradientBinarySearcher, "SETUP_GRADIENT_FUNC")
    prev_setup = getattr(GradientBinarySearcher, "SETUP_GRADIENT_FUNC", None)
    GradientBinarySearcher.SETUP_GRADIENT_FUNC = {DMRegistry[nn.Linear]: _setup_grad_manager_linear}
    try:
        hps = searcher._estimate_gradient_scores(
            data_loader,
            loss_func,
            max_iter_data_loader=max_batches
            if max_iter_data_loader is None
            else max_iter_data_loader,
        )
    finally:
        if had_setup:
            GradientBinarySearcher.SETUP_GRADIENT_FUNC = prev_setup
        else:
            delattr(GradientBinarySearcher, "SETUP_GRADIENT_FUNC")
    return hps, data_loader.num_batches


def test_gradnas_score_averaging_and_convergence(use_channel_div_4):
    modelopt_model = _make_gradnas_model()
    dummy_input = torch.ones(2, 4)

    hps_avg, avg_batches = _estimate_gradnas_scores(
        modelopt_model,
        dummy_input,
        average_scores=True,
        convergence_tol=1e-6,
        convergence_patience=1,
        convergence_min_updates=1,
        max_batches=20,
    )
    avg_scores = {hp_name: hparam.score_tensor.clone() for hp_name, hparam in hps_avg.items()}
    hps_sum, sum_batches = _estimate_gradnas_scores(
        modelopt_model,
        dummy_input,
        average_scores=False,
        convergence_tol=1e-6,
        convergence_patience=1,
        convergence_min_updates=1,
        max_batches=20,
    )

    assert avg_batches == 2
    assert sum_batches == 2

    for hp_name, hparam in hps_sum.items():
        assert torch.allclose(
            hparam.score_tensor,
            avg_scores[hp_name] * sum_batches,
        )


@pytest.mark.parametrize(
    ("convergence_patience", "convergence_min_updates"),
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (2, 3),
    ],
)
def test_gradnas_convergence_patience_and_min_updates(
    use_channel_div_4,
    convergence_patience,
    convergence_min_updates,
):
    modelopt_model = _make_gradnas_model()
    dummy_input = torch.ones(2, 4)
    max_batches = 20

    _, num_batches = _estimate_gradnas_scores(
        modelopt_model,
        dummy_input,
        average_scores=True,
        convergence_tol=1e-6,
        convergence_patience=convergence_patience,
        convergence_min_updates=convergence_min_updates,
        max_batches=max_batches,
    )

    expected_batches = max(convergence_min_updates, 2) + convergence_patience - 1
    assert num_batches == expected_batches


@pytest.mark.parametrize(
    ("convergence_tol", "convergence_patience"),
    [
        (None, 1),
        (1e-6, 0),
    ],
)
def test_gradnas_convergence_disabled_runs_full_loader(
    use_channel_div_4,
    convergence_tol,
    convergence_patience,
):
    modelopt_model = _make_gradnas_model()
    dummy_input = torch.ones(2, 4)
    max_batches = 5

    _, num_batches = _estimate_gradnas_scores(
        modelopt_model,
        dummy_input,
        average_scores=True,
        convergence_tol=convergence_tol,
        convergence_patience=convergence_patience,
        convergence_min_updates=1,
        max_batches=max_batches,
    )

    assert num_batches == max_batches
